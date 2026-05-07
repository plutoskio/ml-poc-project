from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config import RESULTS_DIR, WALMART_RAW_DIR


FEATURE_LABELS = {
    "store_nbr": "Store",
    "item_nbr": "Product",
    "station_nbr": "Weather station",
    "dayofweek": "Day of week",
    "dayofmonth": "Day of month",
    "weekofyear": "Week of year",
    "is_weekend": "Weekend flag",
    "is_month_start": "Month-start flag",
    "is_month_end": "Month-end flag",
    "days_since_start": "Sales history age",
    "lag_1": "Previous-day units",
    "lag_7": "Same weekday last week",
    "lag_28": "Same weekday four weeks ago",
    "rolling_mean_7": "7-day average units",
    "rolling_mean_28": "28-day average units",
    "rolling_mean_90": "90-day average units",
    "tmax": "Maximum temperature",
    "tmin": "Minimum temperature",
    "tavg": "Average temperature",
    "dewpoint": "Dew point",
    "wetbulb": "Wet bulb temperature",
    "preciptotal": "Precipitation",
    "stnpressure": "Station pressure",
    "sealevel": "Sea-level pressure",
    "resultspeed": "Wind speed",
    "resultdir": "Wind direction",
    "avgspeed": "Average wind speed",
    "has_rain": "Rain flag",
    "has_snow": "Snow flag",
    "has_fog": "Fog flag",
    "has_thunder": "Thunder flag",
    "has_freezing": "Freezing flag",
    "weather_event_count": "Weather event count",
}


def _format_units(value: float) -> str:
    return f"{value:,.0f}"


def _feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def build_app() -> None:
    """Build the Streamlit management dashboard for the ML proof of concept."""
    st.set_page_config(
        page_title="Walmart Demand Forecasting",
        layout="wide",
    )

    st.title("Walmart Demand Forecasting")
    st.markdown(
        "Management view of item-level demand forecasts, model credibility, and "
        "store-product forecast performance."
    )

    st.header("Executive Objective")
    st.write(
        "The model estimates daily unit demand for each store and product. The "
        "operational use case is inventory planning: identifying where demand is "
        "likely to occur, how accurate the forecast is, and which store-product "
        "combinations deserve management attention."
    )
    st.caption(
        "The historical same-day weather fields are used as a proxy for same-day "
        "weather forecasts. In production, the model inputs for future dates would "
        "come from weather forecasts available before the sales day."
    )

    overview_path = Path(RESULTS_DIR) / "data_overview.csv"
    metrics_path = Path(RESULTS_DIR) / "model_metrics.csv"
    predictions_path = Path(RESULTS_DIR) / "best_model_test_predictions.csv"
    feature_importance_path = Path(RESULTS_DIR) / "feature_importance.csv"

    st.header("Data Coverage")
    if overview_path.exists():
        overview = pd.read_csv(overview_path)
        values = dict(zip(overview["metric"], overview["value"], strict=False))
        coverage_cols = st.columns(5)
        coverage_cols[0].metric("Historical Rows", f"{int(values.get('rows', 0)):,}")
        coverage_cols[1].metric("Stores", values.get("stores", ""))
        coverage_cols[2].metric("Items", values.get("items", ""))
        coverage_cols[3].metric("Weather Stations", values.get("weather_stations", ""))
        zero_share = float(values.get("zero_sales_share", 0))
        coverage_cols[4].metric("Zero-sales Rows", f"{zero_share:.1%}")

        st.caption(
            f"Raw data folder: `{WALMART_RAW_DIR}`. Backtest uses a chronological split: "
            f"train before {values.get('chronological_test_start')} and test from that date onward."
        )
    else:
        st.info("Run `python scripts/prepare_data.py` to generate dataset summaries.")

    predictions = None
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path, parse_dates=["date"])

    st.header("Forecast Credibility")
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        best = metrics.sort_values("rmsle").iloc[0]

        credibility_cols = st.columns(4)
        credibility_cols[0].metric("Selected Model", best["model_name"])
        credibility_cols[1].metric("R^2", f"{best['r2']:.3f}")
        credibility_cols[2].metric("MAE", f"{best['mae']:.3f}")
        credibility_cols[3].metric("Positive-sales MAE", f"{best['positive_sales_mae']:.2f}")

        st.write(
            "Credibility comes from a chronological backtest: models are trained on "
            "past dates and evaluated on later dates. The selected model explains "
            f"{best['r2']:.1%} of test-period variance and materially beats the "
            "transparent lag baseline."
        )
        st.dataframe(
            metrics[
                [
                    "model_name",
                    "mae",
                    "rmse",
                    "r2",
                    "positive_sales_mae",
                    "zero_sales_mae",
                    "mean_actual_units",
                    "mean_predicted_units",
                ]
            ],
            width="stretch",
        )

        if feature_importance_path.exists():
            importance = pd.read_csv(feature_importance_path)
            top_importance = (
                importance[importance["importance_mae"] > 0]
                .head(15)
                .sort_values("importance_mae", ascending=True)
                .copy()
            )
            top_importance["feature_label"] = top_importance["feature"].map(_feature_label)
            importance_chart = px.bar(
                top_importance,
                x="importance_mae",
                y="feature_label",
                orientation="h",
                error_x="importance_mae_std",
                title="Top forecast drivers: increase in MAE when each feature is shuffled",
                labels={
                    "importance_mae": "Increase in MAE",
                    "feature_label": "",
                    "importance_mae_std": "Repeated-shuffle variation",
                },
            )
            st.plotly_chart(importance_chart, width="stretch")
            sample_rows = int(importance["sample_rows"].iloc[0])
            positive_share = float(importance["positive_sales_share"].iloc[0])
            st.caption(
                "Permutation importance is measured on a chronological test sample. "
                "Higher bars mean the model's error increases more when that input is "
                f"randomly shuffled. Sample rows: {sample_rows:,}; positive-sales share: "
                f"{positive_share:.1%}."
            )
        else:
            st.info(
                "Feature importance will appear after running `python scripts/train_models.py` "
                "or `python scripts/main.py`."
            )

        if predictions is not None:
            daily = (
                predictions.groupby("date", as_index=False)
                .agg(
                    realized_units=("actual_units", "sum"),
                    predicted_units=("predicted_units", "sum"),
                    absolute_error=("absolute_error", "sum"),
                )
                .sort_values("date")
            )
            daily_long = daily.melt(
                id_vars=["date"],
                value_vars=["realized_units", "predicted_units"],
                var_name="series",
                value_name="units",
            )
            daily_long["series"] = daily_long["series"].map(
                {
                    "realized_units": "Realized units",
                    "predicted_units": "Predicted units",
                }
            )
            daily_chart = px.line(
                daily_long,
                x="date",
                y="units",
                color="series",
                title="Portfolio-level daily demand: realized vs predicted",
                labels={"date": "Date", "units": "Units", "series": ""},
            )
            st.plotly_chart(daily_chart, width="stretch")

            total_realized = daily["realized_units"].sum()
            total_predicted = daily["predicted_units"].sum()
            bias = (total_predicted - total_realized) / total_realized
            test_cols = st.columns(3)
            test_cols[0].metric("Test Realized Units", _format_units(total_realized))
            test_cols[1].metric("Test Predicted Units", _format_units(total_predicted))
            test_cols[2].metric("Portfolio Bias", f"{bias:.1%}")

        st.success(
            f"Selected model: {best['model_name']} "
            f"(R^2={best['r2']:.4f}, MAE={best['mae']:.4f}, "
            f"positive-sales MAE={best['positive_sales_mae']:.4f})."
        )
    else:
        st.info("Model metrics will appear here after running `python scripts/main.py`.")

    st.header("Forecast Quality Diagnostics")
    if predictions is not None:
        diagnostics_cols = st.columns(2)

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
        with diagnostics_cols[0]:
            st.plotly_chart(scatter, width="stretch")

        error_sample = predictions[predictions["actual_units"] > 0].sample(
            n=min(15_000, int((predictions["actual_units"] > 0).sum())),
            random_state=42,
        )
        error_hist = px.histogram(
            error_sample,
            x="absolute_error",
            nbins=60,
            title="Absolute error distribution on positive-sales rows",
            labels={"absolute_error": "Absolute error in units"},
        )
        error_hist.update_xaxes(range=[0, error_sample["absolute_error"].quantile(0.98)])
        with diagnostics_cols[1]:
            st.plotly_chart(error_hist, width="stretch")

    st.header("Management Drilldown: Highest-volume Product")
    if predictions is not None:

        top_items_path = Path(RESULTS_DIR) / "top_items.csv"
        if top_items_path.exists():
            top_item = int(pd.read_csv(top_items_path).iloc[0]["item_nbr"])
        else:
            top_item = int(
                predictions.groupby("item_nbr")["actual_units"].sum().idxmax()
            )

        st.caption(
            f"Item {top_item} is the highest-selling product in the historical data. "
            "Use the controls below to inspect realized versus predicted demand for one store and week."
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
        st.plotly_chart(line, width="stretch")
        st.caption(
            f"This selected calendar week has {week_predictions['date'].nunique()} "
            "available sales dates in the test data."
        )

        week_realized = week_predictions["actual_units"].sum()
        week_predicted = week_predictions["predicted_units"].sum()
        week_error = abs(week_realized - week_predicted)
        week_cols = st.columns(3)
        week_cols[0].metric("Week Realized Units", _format_units(week_realized))
        week_cols[1].metric("Week Predicted Units", _format_units(week_predicted))
        week_cols[2].metric("Week Absolute Error", _format_units(week_error))

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
            width="stretch",
        )
    else:
        st.info(
            "Dated realized-vs-predicted plots will appear after running "
            "`python scripts/main.py` or `python scripts/train_models.py`."
        )

    st.header("Operational Notes")
    st.write(
        "The model is most credible as a short-term planning aid, not as an automatic "
        "ordering system. It should be monitored for forecast bias, retrained as new "
        "sales become available, and paired with business context such as promotions, "
        "holidays, and supply constraints."
    )


if __name__ == "__main__":
    build_app()
