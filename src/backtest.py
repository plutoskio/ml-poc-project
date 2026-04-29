"""Backtesting helpers for long/cash index timing strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def make_long_cash_strategy_frame(
    predictions: pd.DataFrame,
    threshold: float,
    cost_per_trade: float = 0.0001,
) -> pd.DataFrame:
    """Return row-level strategy returns from prediction probabilities.

    ``predictions`` must include ``Date``, ``index_name``, ``next_return``, and
    ``p_up``.  A signal of 1 means long for the next return; 0 means cash.
    """

    frame = predictions.copy()
    frame = frame.sort_values(["index_name", "Date"]).reset_index(drop=True)
    frame["signal"] = (frame["p_up"] >= threshold).astype(int)
    frame["trade"] = (
        frame.groupby("index_name")["signal"].diff().abs().fillna(frame["signal"])
    )
    frame["gross_strategy_return"] = frame["signal"] * frame["next_return"]
    frame["strategy_return"] = (
        frame["gross_strategy_return"] - cost_per_trade * frame["trade"]
    )
    frame["threshold"] = threshold
    frame["cost_per_trade"] = cost_per_trade
    return frame


def make_rule_strategy_frame(
    dataset: pd.DataFrame,
    signal: pd.Series,
    strategy_key: str,
    cost_per_trade: float = 0.0001,
) -> pd.DataFrame:
    """Return row-level strategy returns for a non-model baseline signal."""

    frame = dataset[["Date", "index_name", "next_return"]].copy()
    frame["strategy_key"] = strategy_key
    frame["signal"] = signal.astype(int).to_numpy()
    frame = frame.sort_values(["index_name", "Date"]).reset_index(drop=True)
    frame["trade"] = (
        frame.groupby("index_name")["signal"].diff().abs().fillna(frame["signal"])
    )
    frame["gross_strategy_return"] = frame["signal"] * frame["next_return"]
    frame["strategy_return"] = (
        frame["gross_strategy_return"] - cost_per_trade * frame["trade"]
    )
    frame["threshold"] = np.nan
    frame["cost_per_trade"] = cost_per_trade
    return frame


def portfolio_daily_returns(strategy_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level index signals into equal-weight daily returns."""

    daily = (
        strategy_frame.groupby("Date", as_index=False)
        .agg(
            strategy_return=("strategy_return", "mean"),
            gross_strategy_return=("gross_strategy_return", "mean"),
            exposure=("signal", "mean"),
            turnover=("trade", "mean"),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )
    daily["equity"] = (1 + daily["strategy_return"]).cumprod()
    daily["gross_equity"] = (1 + daily["gross_strategy_return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1
    return daily


def strategy_metrics(strategy_frame: pd.DataFrame) -> dict[str, float]:
    """Compute portfolio-level annualized return, Sharpe, and drawdown metrics."""

    daily = portfolio_daily_returns(strategy_frame)
    returns = daily["strategy_return"]
    gross_returns = daily["gross_strategy_return"]

    annualized_return = returns.mean() * TRADING_DAYS_PER_YEAR
    annualized_volatility = returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = (
        annualized_return / annualized_volatility
        if annualized_volatility and not np.isnan(annualized_volatility)
        else 0.0
    )

    gross_annualized_return = gross_returns.mean() * TRADING_DAYS_PER_YEAR
    gross_annualized_volatility = gross_returns.std(ddof=1) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    gross_sharpe_ratio = (
        gross_annualized_return / gross_annualized_volatility
        if gross_annualized_volatility and not np.isnan(gross_annualized_volatility)
        else 0.0
    )

    return {
        "days": float(len(daily)),
        "total_return": float(daily["equity"].iloc[-1] - 1),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(daily["drawdown"].min()),
        "win_rate": float((returns > 0).mean()),
        "exposure": float(daily["exposure"].mean()),
        "turnover": float(daily["turnover"].mean()),
        "gross_total_return": float(daily["gross_equity"].iloc[-1] - 1),
        "gross_annualized_return": float(gross_annualized_return),
        "gross_sharpe_ratio": float(gross_sharpe_ratio),
    }


def choose_threshold(
    predictions: pd.DataFrame,
    thresholds: list[float],
    cost_per_trade: float = 0.0001,
    min_exposure: float = 0.10,
) -> tuple[float, dict[str, float]]:
    """Choose the validation threshold with the best Sharpe ratio."""

    best_threshold = thresholds[0]
    best_metrics: dict[str, float] | None = None

    for threshold in thresholds:
        strategy_frame = make_long_cash_strategy_frame(
            predictions=predictions,
            threshold=threshold,
            cost_per_trade=cost_per_trade,
        )
        metrics = strategy_metrics(strategy_frame)
        if metrics["exposure"] < min_exposure:
            continue

        best_sharpe = best_metrics["sharpe_ratio"] if best_metrics is not None else None
        if best_sharpe is None or metrics["sharpe_ratio"] > best_sharpe:
            best_threshold = threshold
            best_metrics = metrics

    if best_metrics is None:
        strategy_frame = make_long_cash_strategy_frame(
            predictions=predictions,
            threshold=thresholds[0],
            cost_per_trade=cost_per_trade,
        )
        best_metrics = strategy_metrics(strategy_frame)

    return best_threshold, best_metrics


def add_strategy_identity(
    strategy_frame: pd.DataFrame,
    strategy_key: str,
    model_key: str | None = None,
    split: str | None = None,
) -> pd.DataFrame:
    """Attach identifying columns to a strategy return frame."""

    frame = strategy_frame.copy()
    frame["strategy_key"] = strategy_key
    frame["model_key"] = model_key if model_key is not None else strategy_key
    if split is not None:
        frame["split"] = split
    return frame
