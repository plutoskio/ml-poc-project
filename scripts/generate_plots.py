"""Generate static PNG plots from corrected model and strategy results."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import PLOTS_DIR, RESULTS_DIR

PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D5D9E0",
    "axes.labelcolor": "#202833",
    "xtick.color": "#394150",
    "ytick.color": "#394150",
    "grid.color": "#E6E9EF",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
}

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

PRIMARY_STRATEGIES = [
    "buy_and_hold",
    "price_above_trailing_ema_20",
    "positive_20d_momentum",
    "log_reg_long_cash",
    "random_forest_long_cash",
    "hist_gradient_boosting_long_cash",
]


def _save(fig: plt.Figure, filename: str) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    output_path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def _load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    equity = pd.read_csv(RESULTS_DIR / "equity_curves.csv", parse_dates=["Date"])
    strategy_metrics = pd.read_csv(RESULTS_DIR / "strategy_metrics.csv")
    strategy_returns = pd.read_csv(
        RESULTS_DIR / "strategy_returns.csv",
        parse_dates=["Date"],
    )
    model_metrics = pd.read_csv(RESULTS_DIR / "model_metrics.csv")
    return equity, strategy_metrics, strategy_returns, model_metrics


def plot_test_equity_curves(equity: pd.DataFrame) -> None:
    test_equity = equity[
        (equity["split"] == "test") & (equity["strategy_key"].isin(PRIMARY_STRATEGIES))
    ].copy()
    test_equity["strategy_label"] = test_equity["strategy_key"].map(STRATEGY_LABELS)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(
        data=test_equity,
        x="Date",
        y="equity",
        hue="strategy_label",
        linewidth=2,
        ax=ax,
    )
    ax.axhline(1, color="#7A8494", linewidth=1, linestyle="--")
    ax.set_title("Test Period Equity Curves: Long/Cash Strategies vs Baselines")
    ax.set_xlabel("")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, axis="y")
    ax.legend(title="", loc="best")
    _save(fig, "test_equity_curves.png")


def plot_test_drawdowns(equity: pd.DataFrame) -> None:
    test_equity = equity[
        (equity["split"] == "test") & (equity["strategy_key"].isin(PRIMARY_STRATEGIES))
    ].copy()
    test_equity["strategy_label"] = test_equity["strategy_key"].map(STRATEGY_LABELS)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(
        data=test_equity,
        x="Date",
        y="drawdown",
        hue="strategy_label",
        linewidth=2,
        ax=ax,
    )
    ax.axhline(0, color="#7A8494", linewidth=1)
    ax.set_title("Test Period Drawdowns")
    ax.set_xlabel("")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(True, axis="y")
    ax.legend(title="", loc="lower left")
    _save(fig, "test_drawdowns.png")


def plot_strategy_metric_bars(strategy_metrics: pd.DataFrame) -> None:
    test_metrics = strategy_metrics[
        (strategy_metrics["split"] == "test")
        & (strategy_metrics["strategy_key"].isin(PRIMARY_STRATEGIES))
    ].copy()
    test_metrics["strategy_label"] = test_metrics["strategy_key"].map(STRATEGY_LABELS)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ("total_return", "Total Return", "{:.0%}"),
        ("sharpe_ratio", "Sharpe Ratio", "{:.2f}"),
        ("max_drawdown", "Max Drawdown", "{:.0%}"),
    ]

    for ax, (column, title, formatter) in zip(axes, metrics, strict=True):
        plot_df = test_metrics.sort_values(column, ascending=False)
        sns.barplot(
            data=plot_df,
            x=column,
            y="strategy_label",
            color="#3B6EA8",
            ax=ax,
        )
        ax.axvline(0, color="#7A8494", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        for container in ax.containers:
            ax.bar_label(
                container,
                labels=[formatter.format(value) for value in container.datavalues],
                fontsize=8,
                padding=3,
            )
        ax.grid(True, axis="x")

    _save(fig, "test_strategy_metric_bars.png")


def plot_model_metric_bars(model_metrics: pd.DataFrame) -> None:
    metrics = ["accuracy", "balanced_accuracy", "f1"]
    plot_df = model_metrics.melt(
        id_vars=["model_key"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    plot_df["model_label"] = plot_df["model_key"].map(MODEL_LABELS)
    plot_df["metric"] = plot_df["metric"].replace(
        {
            "accuracy": "Accuracy",
            "balanced_accuracy": "Balanced Accuracy",
            "f1": "F1",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=plot_df,
        x="metric",
        y="value",
        hue="model_label",
        ax=ax,
    )
    ax.axhline(0.5, color="#B44747", linewidth=1, linestyle="--", label="Chance")
    ax.set_title("Corrected Test Classification Metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y")
    ax.legend(title="")
    _save(fig, "test_model_metric_bars.png")


def plot_strategy_return_distribution(strategy_returns: pd.DataFrame) -> None:
    test_returns = strategy_returns[
        (strategy_returns["split"] == "test")
        & (strategy_returns["strategy_key"].isin(["buy_and_hold", "log_reg_long_cash"]))
    ].copy()
    test_returns["strategy_label"] = test_returns["strategy_key"].map(STRATEGY_LABELS)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        data=test_returns,
        x="strategy_return",
        hue="strategy_label",
        bins=45,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        linewidth=2,
        ax=ax,
    )
    ax.axvline(0, color="#7A8494", linewidth=1)
    ax.set_title("Daily Return Distribution: Logistic Strategy vs Buy & Hold")
    ax.set_xlabel("Daily strategy return")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(True, axis="y")
    _save(fig, "test_return_distribution.png")


def plot_model_exposure(equity: pd.DataFrame) -> None:
    test_equity = equity[
        (equity["split"] == "test")
        & (
            equity["strategy_key"].isin(
                [
                    "log_reg_long_cash",
                    "random_forest_long_cash",
                    "hist_gradient_boosting_long_cash",
                ]
            )
        )
    ].copy()
    test_equity["strategy_label"] = test_equity["strategy_key"].map(STRATEGY_LABELS)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(
        data=test_equity,
        x="Date",
        y="exposure",
        hue="strategy_label",
        linewidth=2,
        ax=ax,
    )
    ax.set_title("Test Period Daily Portfolio Exposure")
    ax.set_xlabel("")
    ax.set_ylabel("Average long exposure")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y")
    ax.legend(title="")
    _save(fig, "test_model_exposure.png")


def main() -> None:
    plt.rcParams.update(PLOT_STYLE)
    sns.set_theme(style="whitegrid")
    equity, strategy_metrics, strategy_returns, model_metrics = _load_results()

    plot_test_equity_curves(equity)
    plot_test_drawdowns(equity)
    plot_strategy_metric_bars(strategy_metrics)
    plot_model_metric_bars(model_metrics)
    plot_strategy_return_distribution(strategy_returns)
    plot_model_exposure(equity)


if __name__ == "__main__":
    main()
