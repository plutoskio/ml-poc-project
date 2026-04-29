"""Streamlit dashboard for reviewing the quant strategy results."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import MODEL_METRICS_FILE, RESULTS_DIR

STRATEGY_LABELS = {
    "buy_and_hold": "Buy & Hold",
    "cash": "Cash",
    "previous_day_direction": "Previous Day Direction",
    "price_above_trailing_ema_20": "Price > Trailing EMA 20",
    "positive_20d_momentum": "Positive 20D Momentum",
    "log_reg_long_cash": "Logistic Regression",
    "random_forest_long_cash": "Random Forest",
    "hist_gradient_boosting_long_cash": "Hist Gradient Boosting",
}

MODEL_LABELS = {
    "log_reg": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Hist Gradient Boosting",
}

MODEL_STRATEGIES = [
    "log_reg_long_cash",
    "random_forest_long_cash",
    "hist_gradient_boosting_long_cash",
]

DEFAULT_BASELINES = [
    "buy_and_hold",
    "price_above_trailing_ema_20",
    "positive_20d_momentum",
]

TRADING_DAYS_PER_YEAR = 252


def _percent(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def _number(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _strategy_label(strategy_key: str) -> str:
    return STRATEGY_LABELS.get(strategy_key, strategy_key.replace("_", " ").title())


def _style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #d9dee7;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            background: #ffffff;
        }
        div[data-testid="stMetric"] label {
            color: #3a4454;
        }
        .method-box {
            border: 1px solid #d9dee7;
            border-radius: 8px;
            padding: 1rem;
            background: #fbfcfe;
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _require_files(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


@st.cache_data
def _load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    equity = pd.read_csv(RESULTS_DIR / "equity_curves.csv", parse_dates=["Date"])
    strategy_metrics = pd.read_csv(RESULTS_DIR / "strategy_metrics.csv")
    strategy_returns = pd.read_csv(
        RESULTS_DIR / "strategy_returns.csv",
        parse_dates=["Date"],
    )
    model_metrics = pd.read_csv(MODEL_METRICS_FILE)
    return equity, strategy_metrics, strategy_returns, model_metrics


def _add_labels(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.copy()
    if "strategy_key" in labeled.columns:
        labeled["strategy"] = labeled["strategy_key"].map(_strategy_label)
    if "model_key" in labeled.columns:
        labeled["model"] = labeled["model_key"].map(
            lambda key: MODEL_LABELS.get(key, _strategy_label(str(key)))
        )
    return labeled


def _rolling_sharpe(equity: pd.DataFrame, window: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for strategy_key, group in equity.groupby("strategy_key"):
        group = group.sort_values("Date").copy()
        rolling_mean = group["strategy_return"].rolling(window).mean()
        rolling_std = group["strategy_return"].rolling(window).std()
        group["rolling_sharpe"] = (
            rolling_mean
            / rolling_std.replace(0, np.nan)
            * math.sqrt(TRADING_DAYS_PER_YEAR)
        )
        group["strategy_key"] = strategy_key
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def _metrics_from_daily(daily: pd.DataFrame) -> dict[str, float]:
    returns = daily["strategy_return"]
    annualized_return = returns.mean() * TRADING_DAYS_PER_YEAR
    annualized_volatility = returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = (
        annualized_return / annualized_volatility
        if annualized_volatility and not np.isnan(annualized_volatility)
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
    }


def _simulate_model_strategy(
    rows: pd.DataFrame,
    threshold: float,
    cost_per_trade: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    frame = rows.sort_values(["index_name", "Date"]).copy()
    frame["signal"] = (frame["p_up"] >= threshold).astype(int)
    frame["trade"] = (
        frame.groupby("index_name")["signal"].diff().abs().fillna(frame["signal"])
    )
    frame["gross_strategy_return"] = frame["signal"] * frame["next_return"]
    frame["strategy_return"] = (
        frame["gross_strategy_return"] - cost_per_trade * frame["trade"]
    )

    daily = (
        frame.groupby("Date", as_index=False)
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
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1
    return frame, daily, _metrics_from_daily(daily)


def _metric_row(
    strategy_metrics: pd.DataFrame,
    split: str,
    strategy_key: str,
) -> pd.Series | None:
    row = strategy_metrics.loc[
        (strategy_metrics["split"] == split)
        & (strategy_metrics["strategy_key"] == strategy_key)
    ]
    if row.empty:
        return None
    return row.iloc[0]


def _kpi_grid(metrics: pd.Series | dict[str, float]) -> None:
    cols = st.columns(6)
    values = dict(metrics)
    cols[0].metric("Total Return", _percent(values.get("total_return")))
    cols[1].metric("Sharpe", _number(values.get("sharpe_ratio")))
    cols[2].metric("Max Drawdown", _percent(values.get("max_drawdown")))
    cols[3].metric("Exposure", _percent(values.get("exposure")))
    cols[4].metric("Win Rate", _percent(values.get("win_rate")))
    cols[5].metric("Turnover", _percent(values.get("turnover")))


def _plot_equity(equity: pd.DataFrame, title: str) -> go.Figure:
    plot_df = _add_labels(equity)
    fig = px.line(
        plot_df,
        x="Date",
        y="equity",
        color="strategy",
        title=title,
        labels={"equity": "Growth of $1", "strategy": ""},
    )
    fig.add_hline(y=1, line_dash="dash", line_color="#737b86")
    fig.update_layout(hovermode="x unified", legend_title_text="")
    return fig


def _plot_drawdown(equity: pd.DataFrame, title: str) -> go.Figure:
    plot_df = _add_labels(equity)
    fig = px.line(
        plot_df,
        x="Date",
        y="drawdown",
        color="strategy",
        title=title,
        labels={"drawdown": "Drawdown", "strategy": ""},
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(hovermode="x unified", legend_title_text="")
    return fig


def _plot_metric_bars(
    strategy_metrics: pd.DataFrame,
    split: str,
    strategies: list[str],
    metric: str,
) -> go.Figure:
    plot_df = strategy_metrics.loc[
        (strategy_metrics["split"] == split)
        & (strategy_metrics["strategy_key"].isin(strategies))
    ].copy()
    plot_df["strategy"] = plot_df["strategy_key"].map(_strategy_label)
    plot_df = plot_df.sort_values(metric, ascending=False)
    fig = px.bar(
        plot_df,
        x=metric,
        y="strategy",
        orientation="h",
        title=metric.replace("_", " ").title(),
        labels={metric: "", "strategy": ""},
    )
    if metric != "sharpe_ratio":
        fig.update_xaxes(tickformat=".0%")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def _render_overview(
    equity: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    split: str,
    selected_strategy: str,
    comparison_strategies: list[str],
) -> None:
    selected_row = _metric_row(strategy_metrics, split, selected_strategy)
    if selected_row is not None:
        _kpi_grid(selected_row)

    plot_strategies = list(dict.fromkeys([selected_strategy, *comparison_strategies]))
    selected_equity = equity.loc[
        (equity["split"] == split) & (equity["strategy_key"].isin(plot_strategies))
    ]

    st.plotly_chart(
        _plot_equity(selected_equity, "Equity Curves"),
        width="stretch",
    )
    st.plotly_chart(
        _plot_drawdown(selected_equity, "Drawdowns"),
        width="stretch",
    )

    metric_cols = st.columns(3)
    for col, metric in zip(
        metric_cols,
        ["total_return", "sharpe_ratio", "max_drawdown"],
        strict=True,
    ):
        with col:
            st.plotly_chart(
                _plot_metric_bars(strategy_metrics, split, plot_strategies, metric),
                width="stretch",
            )

    table = strategy_metrics.loc[
        (strategy_metrics["split"] == split)
        & (strategy_metrics["strategy_key"].isin(plot_strategies))
    ].copy()
    table["strategy"] = table["strategy_key"].map(_strategy_label)
    display_cols = [
        "strategy",
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
    st.dataframe(
        table[display_cols].sort_values("sharpe_ratio", ascending=False),
        width="stretch",
        hide_index=True,
        column_config={
            "total_return": st.column_config.NumberColumn(format="%.2f"),
            "annualized_return": st.column_config.NumberColumn(format="%.2f"),
            "annualized_volatility": st.column_config.NumberColumn(format="%.2f"),
            "max_drawdown": st.column_config.NumberColumn(format="%.2f"),
            "win_rate": st.column_config.NumberColumn(format="%.2f"),
            "exposure": st.column_config.NumberColumn(format="%.2f"),
            "turnover": st.column_config.NumberColumn(format="%.2f"),
        },
    )


def _render_threshold_lab(
    strategy_returns: pd.DataFrame,
    model_metrics: pd.DataFrame,
    split: str,
) -> None:
    available_models = sorted(
        strategy_returns.loc[
            (strategy_returns["split"] == split) & strategy_returns["p_up"].notna(),
            "model_key",
        ]
        .dropna()
        .unique()
    )
    if not available_models:
        st.info("No saved model probabilities are available for this split.")
        return

    controls = st.columns([1.2, 1.0, 1.0])
    model_key = controls[0].selectbox(
        "Model",
        available_models,
        format_func=lambda key: MODEL_LABELS.get(key, str(key)),
    )
    threshold = controls[1].slider(
        "Probability threshold",
        min_value=0.40,
        max_value=0.75,
        value=0.60,
        step=0.01,
    )
    cost_bps = controls[2].number_input(
        "Cost per signal change, bps",
        min_value=0.0,
        max_value=25.0,
        value=1.0,
        step=0.5,
    )

    rows = strategy_returns.loc[
        (strategy_returns["split"] == split)
        & (strategy_returns["model_key"] == model_key)
        & strategy_returns["p_up"].notna()
    ].copy()

    simulated_rows, daily, metrics = _simulate_model_strategy(
        rows,
        threshold=threshold,
        cost_per_trade=cost_bps / 10_000,
    )

    _kpi_grid(metrics)

    daily_plot = daily.copy()
    daily_plot["strategy"] = (
        f"{MODEL_LABELS.get(model_key, model_key)} @ {threshold:.2f}"
    )
    st.plotly_chart(
        _plot_equity(
            daily_plot.assign(strategy_key=daily_plot["strategy"]), "Simulated Equity"
        ),
        width="stretch",
    )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(
            _plot_drawdown(
                daily_plot.assign(strategy_key=daily_plot["strategy"]),
                "Simulated Drawdown",
            ),
            width="stretch",
        )
    with chart_cols[1]:
        exposure_fig = px.area(
            daily,
            x="Date",
            y="exposure",
            title="Daily Portfolio Exposure",
            labels={"exposure": "Exposure"},
        )
        exposure_fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(exposure_fig, width="stretch")

    distribution_cols = st.columns(2)
    with distribution_cols[0]:
        fig = px.histogram(
            simulated_rows,
            x="p_up",
            color="signal",
            nbins=40,
            title="Predicted Probability Distribution",
            labels={"p_up": "Predicted P(up)", "signal": "Signal"},
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="#b23b3b")
        st.plotly_chart(fig, width="stretch")
    with distribution_cols[1]:
        by_index = (
            simulated_rows.groupby("index_name", as_index=False)
            .agg(
                exposure=("signal", "mean"),
                avg_p_up=("p_up", "mean"),
                avg_return=("strategy_return", "mean"),
                trades=("trade", "sum"),
            )
            .sort_values("exposure", ascending=False)
        )
        fig = px.bar(
            by_index,
            x="index_name",
            y="exposure",
            title="Exposure By Index",
            labels={"index_name": "Index", "exposure": "Exposure"},
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig, width="stretch")

    st.subheader("Classification Metrics")
    model_row = model_metrics.loc[model_metrics["model_key"] == model_key]
    if not model_row.empty:
        metric_df = model_row.melt(
            id_vars=["model_key", "model_name", "model_path"],
            var_name="metric",
            value_name="value",
        )
        metric_df = metric_df.loc[metric_df["metric"] != "model_path"]
        fig = px.bar(
            metric_df,
            x="metric",
            y="value",
            title=(
                f"{MODEL_LABELS.get(model_key, model_key)} "
                "Test Classification Metrics"
            ),
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, width="stretch")

    with st.expander("Inspect row-level signals"):
        display_cols = [
            "Date",
            "index_name",
            "p_up",
            "signal",
            "trade",
            "next_return",
            "strategy_return",
        ]
        st.dataframe(
            simulated_rows[display_cols].sort_values(["Date", "index_name"]),
            width="stretch",
            hide_index=True,
        )


def _render_model_comparison(
    model_metrics: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    split: str,
) -> None:
    st.subheader("Classifier Quality")
    metric_cols = [
        col
        for col in ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
        if col in model_metrics.columns
    ]
    model_plot = model_metrics.melt(
        id_vars=["model_key", "model_name"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="score",
    )
    fig = px.bar(
        model_plot,
        x="metric",
        y="score",
        color="model_name",
        barmode="group",
        title="Test Classification Metrics",
        labels={"model_name": "", "score": "Score"},
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#737b86")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, width="stretch")

    st.subheader("Strategy Quality")
    model_strategy_metrics = strategy_metrics.loc[
        (strategy_metrics["split"] == split)
        & (strategy_metrics["strategy_key"].isin(MODEL_STRATEGIES))
    ].copy()
    model_strategy_metrics["strategy"] = model_strategy_metrics["strategy_key"].map(
        _strategy_label
    )
    strategy_plot = model_strategy_metrics.melt(
        id_vars=["strategy"],
        value_vars=["total_return", "sharpe_ratio", "max_drawdown", "exposure"],
        var_name="metric",
        value_name="value",
    )
    fig = px.bar(
        strategy_plot,
        x="strategy",
        y="value",
        color="metric",
        barmode="group",
        title=f"{split.title()} Model Strategy Metrics",
        labels={"strategy": "", "value": ""},
    )
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        model_strategy_metrics[
            [
                "strategy",
                "threshold",
                "total_return",
                "annualized_return",
                "sharpe_ratio",
                "max_drawdown",
                "exposure",
                "turnover",
            ]
        ].sort_values("sharpe_ratio", ascending=False),
        width="stretch",
        hide_index=True,
    )


def _render_risk(equity: pd.DataFrame, split: str, strategies: list[str]) -> None:
    selected_equity = equity.loc[
        (equity["split"] == split) & (equity["strategy_key"].isin(strategies))
    ].copy()
    selected_equity["strategy"] = selected_equity["strategy_key"].map(_strategy_label)

    window = st.slider("Rolling Sharpe window, trading days", 21, 126, 63, 21)
    rolling = _rolling_sharpe(selected_equity, window)
    rolling["strategy"] = rolling["strategy_key"].map(_strategy_label)
    fig = px.line(
        rolling,
        x="Date",
        y="rolling_sharpe",
        color="strategy",
        title=f"{window}-Day Rolling Sharpe",
        labels={"rolling_sharpe": "Rolling Sharpe", "strategy": ""},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#737b86")
    st.plotly_chart(fig, width="stretch")

    cols = st.columns(2)
    with cols[0]:
        fig = px.line(
            selected_equity,
            x="Date",
            y="exposure",
            color="strategy",
            title="Exposure Through Time",
            labels={"exposure": "Exposure", "strategy": ""},
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig, width="stretch")
    with cols[1]:
        fig = px.line(
            selected_equity,
            x="Date",
            y="turnover",
            color="strategy",
            title="Turnover Through Time",
            labels={"turnover": "Turnover", "strategy": ""},
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")


def _render_methodology() -> None:
    st.markdown(
        """
        <div class="method-box">
        <b>Goal.</b> Predict whether the next trading day is positive, then
        convert predicted probability into a long/cash index strategy.
        </div>
        <div class="method-box">
        <b>Leakage controls.</b> Removed future-looking momentum, ROC, and raw
        EMA columns. Recomputed trailing EMA features from past prices only and
        shifted all model features by one trading day.
        </div>
        <div class="method-box">
        <b>Split.</b> Train on 2010-2019, select trading thresholds on 2020-2021
        validation, and evaluate final results on 2022-2023 test data.
        </div>
        <div class="method-box">
        <b>Strategy.</b> Go long when predicted P(up) is at least the selected
        threshold, otherwise stay in cash. Current production backtests selected
        0.60 for all model strategies.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("Key leakage columns removed:")
    st.code(
        "EMA_10, EMA_20, EMA_50, EMA_200, "
        "mom, mom1, mom2, mom3, ROC_5, ROC_10, ROC_15, ROC_20",
        language="text",
    )
    st.write("Main generated artifacts:")
    st.code(
        "\n".join(
            [
                "results/model_metrics.csv",
                "results/strategy_metrics.csv",
                "results/equity_curves.csv",
                "results/strategy_returns.csv",
                "plots/test_equity_curves.png",
                "plots/test_drawdowns.png",
                "plots/test_model_exposure.png",
            ]
        ),
        language="text",
    )


def build_app() -> None:
    """Render the Streamlit strategy review dashboard."""

    st.set_page_config(
        page_title="Quant Strategy Review",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _style()

    required = [
        MODEL_METRICS_FILE,
        RESULTS_DIR / "strategy_metrics.csv",
        RESULTS_DIR / "equity_curves.csv",
        RESULTS_DIR / "strategy_returns.csv",
    ]
    missing = _require_files(required)
    if missing:
        st.error("Missing result files. Run the training and backtest scripts first.")
        for path in missing:
            st.code(str(path), language="text")
        return

    equity, strategy_metrics, strategy_returns, model_metrics = _load_results()
    available_splits = sorted(equity["split"].dropna().unique())
    default_split = "test" if "test" in available_splits else available_splits[0]

    st.title("Quant Strategy Review Dashboard")
    st.caption(
        "Review supervised model predictions as long/cash trading strategies, "
        "including equity evolution, drawdowns, Sharpe, exposure, "
        "and threshold behavior."
    )

    with st.sidebar:
        st.header("Controls")
        split = st.selectbox(
            "Evaluation split",
            available_splits,
            index=available_splits.index(default_split),
        )

        split_strategies = sorted(
            equity.loc[equity["split"] == split, "strategy_key"].unique()
        )
        default_strategy = (
            "log_reg_long_cash"
            if "log_reg_long_cash" in split_strategies
            else split_strategies[0]
        )
        selected_strategy = st.selectbox(
            "Primary strategy",
            split_strategies,
            index=split_strategies.index(default_strategy),
            format_func=_strategy_label,
        )

        baseline_defaults = [
            key for key in DEFAULT_BASELINES if key in split_strategies
        ]
        comparison_strategies = st.multiselect(
            "Compare with",
            split_strategies,
            default=baseline_defaults,
            format_func=_strategy_label,
        )

    overview, threshold_lab, comparison, risk, methodology = st.tabs(
        [
            "Overview",
            "Threshold Lab",
            "Model Comparison",
            "Risk And Exposure",
            "Methodology",
        ]
    )

    with overview:
        _render_overview(
            equity,
            strategy_metrics,
            split,
            selected_strategy,
            comparison_strategies,
        )

    with threshold_lab:
        _render_threshold_lab(strategy_returns, model_metrics, split)

    with comparison:
        _render_model_comparison(model_metrics, strategy_metrics, split)

    with risk:
        risk_strategies = list(
            dict.fromkeys([selected_strategy, *comparison_strategies])
        )
        _render_risk(equity, split, risk_strategies)

    with methodology:
        _render_methodology()


if __name__ == "__main__":
    build_app()
