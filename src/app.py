from pathlib import Path

import pandas as pd
import streamlit as st

from config import PLOTS_DIR, RESULTS_DIR, WALMART_RAW_DIR


def build_app() -> None:
    """Build the Streamlit presentation for the ML proof of concept."""
    st.set_page_config(
        page_title="Walmart Sales Forecasting POC",
        layout="wide",
    )

    st.title("Walmart Sales Forecasting POC")
    st.markdown(
        "Forecast item-level unit sales for Walmart stores using historical sales "
        "and local weather observations."
    )

    st.header("Business Objective")
    st.write(
        "The objective is to help a retailer anticipate daily demand at the "
        "store-item level. Better short-term sales forecasts can support inventory "
        "planning, replenishment, and staffing decisions, especially when weather "
        "events may affect demand."
    )

    st.header("Dataset and Validation Setup")
    st.write(f"Raw data folder: `{WALMART_RAW_DIR}`")

    overview_path = Path(RESULTS_DIR) / "data_overview.csv"
    if overview_path.exists():
        overview = pd.read_csv(overview_path)
        values = dict(zip(overview["metric"], overview["value"], strict=False))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{int(values.get('rows', 0)):,}")
        col2.metric("Stores", values.get("stores", ""))
        col3.metric("Items", values.get("items", ""))
        zero_share = float(values.get("zero_sales_share", 0))
        col4.metric("Zero Sales Share", f"{zero_share:.1%}")

        st.caption(
            "Chronological split: train before "
            f"{values.get('chronological_test_start')} and test from that date onward."
        )
    else:
        st.info("Run `python scripts/prepare_data.py` to generate dataset summaries.")

    st.header("Exploratory Insights")
    plot_columns = st.columns(2)
    plot_files = [
        ("monthly_unit_sales.png", "Monthly sales volume"),
        ("weather_missingness.png", "Weather data quality"),
        ("top_items.png", "Top selling items"),
        ("top_stores.png", "Top selling stores"),
    ]
    for index, (filename, caption) in enumerate(plot_files):
        path = Path(PLOTS_DIR) / filename
        if path.exists():
            with plot_columns[index % 2]:
                st.image(str(path), caption=caption, use_container_width=True)

    metrics_path = Path(RESULTS_DIR) / "model_metrics.csv"
    if metrics_path.exists():
        st.header("Model Comparison")
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics, use_container_width=True)

        metric_plot = Path(PLOTS_DIR) / "model_metric_bars.png"
        if metric_plot.exists():
            st.image(str(metric_plot), caption="Regression metrics by model", use_container_width=True)

        pca_path = Path(RESULTS_DIR) / "pca_ab_test.csv"
        if pca_path.exists():
            st.subheader("PCA Experiment")
            st.dataframe(pd.read_csv(pca_path), use_container_width=True)
            pca_plot = Path(PLOTS_DIR) / "pca_ab_test.png"
            if pca_plot.exists():
                st.image(str(pca_plot), caption="Numeric ridge with and without PCA", use_container_width=True)

        best = metrics.sort_values("rmsle").iloc[0]
        st.success(
            f"Best model by RMSLE: {best['model_name']} "
            f"(RMSLE={best['rmsle']:.4f}, MAE={best['mae']:.4f})."
        )
    else:
        st.info("Model metrics will appear here after running `python scripts/main.py`.")

    st.header("Prediction Sample")
    predictions_path = Path(RESULTS_DIR) / "model_predictions_sample.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        store_options = sorted(predictions["store_nbr"].unique())
        selected_store = st.selectbox("Store", store_options)
        filtered = predictions[predictions["store_nbr"] == selected_store]
        st.dataframe(
            filtered[
                [
                    "store_nbr",
                    "item_nbr",
                    "month",
                    "dayofweek",
                    "actual_units",
                    "predicted_units",
                    "best_model_id",
                ]
            ].head(200),
            use_container_width=True,
        )

        actual_plot = Path(PLOTS_DIR) / "actual_vs_predicted_sample.png"
        if actual_plot.exists():
            st.image(str(actual_plot), caption="Actual vs predicted sample", use_container_width=True)
    else:
        st.info("Prediction examples will appear after training the models.")


if __name__ == "__main__":
    build_app()
