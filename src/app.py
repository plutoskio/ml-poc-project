from pathlib import Path

import pandas as pd
import plotly.express as px
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
    st.caption(
        "The historical same-day weather fields are used as a proxy for same-day "
        "weather forecasts. In production, the model inputs for future dates would "
        "come from weather forecasts available before the sales day."
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
        best = metrics.sort_values("rmsle").iloc[0]

        metric_cols = st.columns(4)
        metric_cols[0].metric("Best Model", best["model_name"])
        metric_cols[1].metric("R^2", f"{best['r2']:.3f}")
        metric_cols[2].metric("RMSLE", f"{best['rmsle']:.3f}")
        metric_cols[3].metric("MAE", f"{best['mae']:.3f}")

        st.dataframe(
            metrics[
                [
                    "model_name",
                    "mae",
                    "rmse",
                    "rmsle",
                    "r2",
                    "positive_sales_mae",
                    "zero_sales_mae",
                    "mean_actual_units",
                    "mean_predicted_units",
                ]
            ],
            use_container_width=True,
        )

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

        st.success(
            f"Best model by RMSLE: {best['model_name']} "
            f"(RMSLE={best['rmsle']:.4f}, R^2={best['r2']:.4f}, MAE={best['mae']:.4f})."
        )
    else:
        st.info("Model metrics will appear here after running `python scripts/main.py`.")

    st.header("Realized vs Predicted Sales")
    predictions_path = Path(RESULTS_DIR) / "best_model_test_predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path, parse_dates=["date"])

        plot_sample = predictions.sample(
            n=min(15_000, len(predictions)),
            random_state=42,
        )
        scatter = px.scatter(
            plot_sample,
            x="actual_units",
            y="predicted_units",
            hover_data=["date", "store_nbr", "item_nbr", "absolute_error"],
            title="Actual vs predicted units on the chronological test set",
            labels={
                "actual_units": "Realized units",
                "predicted_units": "Predicted units",
            },
            opacity=0.45,
        )
        axis_limit = float(
            max(
                plot_sample["actual_units"].quantile(0.995),
                plot_sample["predicted_units"].quantile(0.995),
                1,
            )
        )
        scatter.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=axis_limit,
            y1=axis_limit,
            line={"color": "#B23A48", "width": 2},
        )
        scatter.update_xaxes(range=[0, axis_limit])
        scatter.update_yaxes(range=[0, axis_limit])
        st.plotly_chart(scatter, use_container_width=True)

        top_items_path = Path(RESULTS_DIR) / "top_items.csv"
        if top_items_path.exists():
            top_item = int(pd.read_csv(top_items_path).iloc[0]["item_nbr"])
        else:
            top_item = int(
                predictions.groupby("item_nbr")["actual_units"].sum().idxmax()
            )

        st.subheader("Top Product, One Store, One Week")
        st.caption(
            f"Item {top_item} is the highest-selling product in the historical data. "
            "Use the controls below to inspect one realized-vs-predicted week for that item."
        )

        top_item_predictions = predictions[predictions["item_nbr"] == top_item].copy()
        store_ranking = (
            top_item_predictions.groupby("store_nbr")["actual_units"]
            .sum()
            .sort_values(ascending=False)
        )
        store_options = store_ranking.index.astype(int).tolist()
        selected_store = st.selectbox(
            "Store",
            store_options,
            index=0,
            help="Default is the store with the highest realized test-period sales for the top item.",
        )

        item_store_predictions = top_item_predictions[
            top_item_predictions["store_nbr"] == selected_store
        ].copy()
        item_store_predictions["week_start"] = (
            item_store_predictions["date"]
            - pd.to_timedelta(item_store_predictions["date"].dt.dayofweek, unit="D")
        )
        week_summary = (
            item_store_predictions.groupby("week_start")
            .agg(total_realized_units=("actual_units", "sum"), days=("date", "nunique"))
            .sort_values("total_realized_units", ascending=False)
        )
        week_options = sorted(item_store_predictions["week_start"].drop_duplicates())
        default_week = week_summary.index[0]
        default_week_index = week_options.index(default_week)
        selected_week = st.selectbox(
            "Week",
            week_options,
            index=default_week_index,
            format_func=lambda value: (
                f"{value.date()} to {(value + pd.Timedelta(days=6)).date()}"
            ),
            help="Default is the week with the highest realized sales for the selected store and top item.",
        )

        week_predictions = (
            item_store_predictions[
                item_store_predictions["week_start"] == selected_week
            ]
            .sort_values("date")
            .copy()
        )
        weekly_long = week_predictions.melt(
            id_vars=["date"],
            value_vars=["actual_units", "predicted_units"],
            var_name="series",
            value_name="units",
        )
        weekly_long["series"] = weekly_long["series"].map(
            {
                "actual_units": "Realized sales",
                "predicted_units": "Predicted sales",
            }
        )
        line = px.line(
            weekly_long,
            x="date",
            y="units",
            color="series",
            markers=True,
            title=f"Item {top_item}, store {selected_store}: one-week sales forecast",
            labels={"date": "Date", "units": "Units", "series": ""},
        )
        st.plotly_chart(line, use_container_width=True)
        st.caption(
            f"This selected calendar week has {week_predictions['date'].nunique()} "
            "available sales dates in the test data."
        )

        st.dataframe(
            week_predictions[
                [
                    "date",
                    "store_nbr",
                    "item_nbr",
                    "actual_units",
                    "predicted_units",
                    "absolute_error",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.info(
            "Dated realized-vs-predicted plots will appear after running "
            "`python scripts/main.py` or `python scripts/train_models.py`."
        )


if __name__ == "__main__":
    build_app()
