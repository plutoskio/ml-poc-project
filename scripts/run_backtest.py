"""Run validation-threshold selection and test-period backtests."""

from __future__ import annotations

import joblib
import pandas as pd

from src.backtest import (
    add_strategy_identity,
    choose_threshold,
    make_long_cash_strategy_frame,
    make_rule_strategy_frame,
    portfolio_daily_returns,
    strategy_metrics,
)
from src.config import MODELS, RESULTS_DIR
from src.data import feature_columns, load_modeling_dataset
from src.modeling import build_models, predict_positive_probability

THRESHOLDS = [0.50, 0.52, 0.55, 0.57, 0.60]
COST_PER_TRADE = 0.0001


def _prediction_frame(
    dataset: pd.DataFrame,
    model: object,
    columns: list[str],
    split: str,
) -> pd.DataFrame:
    split_df = dataset.loc[dataset["split"] == split].copy()
    p_up = predict_positive_probability(model, split_df[columns])
    return pd.DataFrame(
        {
            "Date": split_df["Date"],
            "index_name": split_df["index_name"],
            "next_return": split_df["next_return"],
            "p_up": p_up,
        },
        index=split_df.index,
    )


def _baseline_strategy_frames(
    dataset: pd.DataFrame,
    split: str,
) -> list[pd.DataFrame]:
    split_df = dataset.loc[dataset["split"] == split].copy()
    return [
        add_strategy_identity(
            make_rule_strategy_frame(
                split_df,
                signal=pd.Series(1, index=split_df.index),
                strategy_key="buy_and_hold",
                cost_per_trade=COST_PER_TRADE,
            ),
            strategy_key="buy_and_hold",
            split=split,
        ),
        add_strategy_identity(
            make_rule_strategy_frame(
                split_df,
                signal=pd.Series(0, index=split_df.index),
                strategy_key="cash",
                cost_per_trade=COST_PER_TRADE,
            ),
            strategy_key="cash",
            split=split,
        ),
        add_strategy_identity(
            make_rule_strategy_frame(
                split_df,
                signal=(split_df["return_1d"] > 0).astype(int),
                strategy_key="previous_day_direction",
                cost_per_trade=COST_PER_TRADE,
            ),
            strategy_key="previous_day_direction",
            split=split,
        ),
        add_strategy_identity(
            make_rule_strategy_frame(
                split_df,
                signal=(split_df["price_vs_trailing_ema_20"] > 0).astype(int),
                strategy_key="price_above_trailing_ema_20",
                cost_per_trade=COST_PER_TRADE,
            ),
            strategy_key="price_above_trailing_ema_20",
            split=split,
        ),
        add_strategy_identity(
            make_rule_strategy_frame(
                split_df,
                signal=(split_df["return_20d"] > 0).astype(int),
                strategy_key="positive_20d_momentum",
                cost_per_trade=COST_PER_TRADE,
            ),
            strategy_key="positive_20d_momentum",
            split=split,
        ),
    ]


def main() -> None:
    """Run long/cash backtests and write strategy result artifacts."""

    dataset = load_modeling_dataset()
    columns = feature_columns(dataset)

    train_df = dataset.loc[dataset["split"] == "train"]
    X_train = train_df[columns]
    y_train = train_df["target"]

    rows: list[dict[str, float | str]] = []
    equity_frames: list[pd.DataFrame] = []
    strategy_frames: list[pd.DataFrame] = []

    for split in ["validation", "test"]:
        for frame in _baseline_strategy_frames(dataset, split):
            metrics = strategy_metrics(frame)
            strategy_key = frame["strategy_key"].iloc[0]
            rows.append(
                {
                    "strategy_key": strategy_key,
                    "model_key": strategy_key,
                    "split": split,
                    "threshold": None,
                    "cost_per_trade": COST_PER_TRADE,
                    **metrics,
                }
            )

            daily = portfolio_daily_returns(frame)
            daily["strategy_key"] = strategy_key
            daily["model_key"] = strategy_key
            daily["split"] = split
            equity_frames.append(daily)
            strategy_frames.append(frame)

    for model_key, model_config in MODELS.items():
        train_only_model = build_models()[model_key]
        train_only_model.fit(X_train, y_train)
        validation_predictions = _prediction_frame(
            dataset,
            train_only_model,
            columns,
            split="validation",
        )

        threshold, validation_threshold_metrics = choose_threshold(
            validation_predictions,
            thresholds=THRESHOLDS,
            cost_per_trade=COST_PER_TRADE,
        )
        rows.append(
            {
                "strategy_key": f"{model_key}_long_cash",
                "model_key": model_key,
                "split": "validation_threshold_selection",
                "threshold": threshold,
                "cost_per_trade": COST_PER_TRADE,
                **validation_threshold_metrics,
            }
        )

        final_model = joblib.load(model_config["path"])
        predictions = _prediction_frame(dataset, final_model, columns, split="test")
        strategy_frame = make_long_cash_strategy_frame(
            predictions,
            threshold=threshold,
            cost_per_trade=COST_PER_TRADE,
        )
        strategy_key = f"{model_key}_long_cash"
        strategy_frame = add_strategy_identity(
            strategy_frame,
            strategy_key=strategy_key,
            model_key=model_key,
            split="test",
        )

        metrics = strategy_metrics(strategy_frame)
        rows.append(
            {
                "strategy_key": strategy_key,
                "model_key": model_key,
                "split": "test",
                "threshold": threshold,
                "cost_per_trade": COST_PER_TRADE,
                **metrics,
            }
        )

        daily = portfolio_daily_returns(strategy_frame)
        daily["strategy_key"] = strategy_key
        daily["model_key"] = model_key
        daily["split"] = "test"
        daily["threshold"] = threshold
        equity_frames.append(daily)
        strategy_frames.append(strategy_frame)

    RESULTS_DIR.mkdir(exist_ok=True)
    strategy_metrics_df = pd.DataFrame(rows)
    equity_curves_df = pd.concat(equity_frames, ignore_index=True)
    strategy_returns_df = pd.concat(strategy_frames, ignore_index=True)

    strategy_metrics_path = RESULTS_DIR / "strategy_metrics.csv"
    equity_curves_path = RESULTS_DIR / "equity_curves.csv"
    strategy_returns_path = RESULTS_DIR / "strategy_returns.csv"

    strategy_metrics_df.to_csv(strategy_metrics_path, index=False)
    equity_curves_df.to_csv(equity_curves_path, index=False)
    strategy_returns_df.to_csv(strategy_returns_path, index=False)

    print("Backtest complete.")
    print(f"Saved strategy metrics to: {strategy_metrics_path}")
    print(f"Saved equity curves to: {equity_curves_path}")
    print(f"Saved row-level strategy returns to: {strategy_returns_path}")
    print("\nTest strategy metrics:")
    test_cols = [
        "strategy_key",
        "threshold",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "exposure",
        "turnover",
    ]
    test_report = strategy_metrics_df.loc[
        strategy_metrics_df["split"] == "test", test_cols
    ]
    print(test_report.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
