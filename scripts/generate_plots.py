from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import PLOTS_DIR, RESULTS_DIR  # noqa: E402


def save_barplot(data: pd.DataFrame, x: str, y: str, title: str, output_name: str) -> None:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=data, x=x, y=y, color="#3A6EA5")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=160)
    plt.close()


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    monthly_path = RESULTS_DIR / "monthly_sales.csv"
    if monthly_path.exists():
        monthly_sales = pd.read_csv(monthly_path)
        plt.figure(figsize=(12, 5))
        sns.lineplot(data=monthly_sales, x="month_period", y="total_units", marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title("Monthly Unit Sales")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "monthly_unit_sales.png", dpi=160)
        plt.close()

    top_items_path = RESULTS_DIR / "top_items.csv"
    if top_items_path.exists():
        top_items = pd.read_csv(top_items_path).head(12)
        save_barplot(top_items, "item_nbr", "total_units", "Top Items by Total Units", "top_items.png")

    top_stores_path = RESULTS_DIR / "top_stores.csv"
    if top_stores_path.exists():
        top_stores = pd.read_csv(top_stores_path).head(12)
        save_barplot(top_stores, "store_nbr", "total_units", "Top Stores by Total Units", "top_stores.png")

    missing_path = RESULTS_DIR / "weather_missingness.csv"
    if missing_path.exists():
        missing = (
            pd.read_csv(missing_path)
            .sort_values("missing_or_coded_share", ascending=False)
            .head(12)
        )
        save_barplot(
            missing,
            "column",
            "missing_or_coded_share",
            "Weather Missing or Coded Value Share",
            "weather_missingness.png",
        )

    metrics_path = RESULTS_DIR / "model_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        metric_long = metrics.melt(
            id_vars=["model_name"],
            value_vars=["mae", "rmse", "rmsle", "positive_sales_mae"],
            var_name="metric",
            value_name="value",
        )
        plt.figure(figsize=(11, 6))
        sns.barplot(data=metric_long, x="metric", y="value", hue="model_name")
        plt.title("Model Metrics on Chronological Test Split")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "model_metric_bars.png", dpi=160)
        plt.close()

    predictions_path = RESULTS_DIR / "model_predictions_sample.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        plt.figure(figsize=(7, 7))
        sns.scatterplot(
            data=predictions,
            x="actual_units",
            y="predicted_units",
            alpha=0.35,
            edgecolor=None,
        )
        limit = max(predictions["actual_units"].max(), predictions["predicted_units"].max())
        plt.plot([0, limit], [0, limit], color="#B23A48", linewidth=1)
        plt.title("Actual vs Predicted Units, Sample")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "actual_vs_predicted_sample.png", dpi=160)
        plt.close()

    print(f"Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
