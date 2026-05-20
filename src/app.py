from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config import RESULTS_DIR, WALMART_RAW_DIR


CHART_COLORS = [
    "#7CC4FF",
    "#4FB286",
    "#F2B84B",
    "#D96C75",
    "#9D7FEA",
    "#6ED3CF",
]

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


@st.cache_data
def _read_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=parse_dates)


def _format_units(value: float) -> str:
    return f"{value:,.0f}"


def _format_percent(value: float) -> str:
    return f"{value:.1%}"


def _feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #18202b;
            --panel: #202938;
            --panel-soft: #263244;
            --border: #3b4657;
            --text: #f8fafc;
            --muted: #c5cfdd;
            --accent: #8bc8ff;
            --accent-2: #4fb286;
            --accent-3: #f2b84b;
            --danger: #d96c75;
        }

        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"] {
            display: none;
        }

        #MainMenu,
        footer {
            visibility: hidden;
        }

        .stApp {
            background:
                linear-gradient(180deg, #202938 0%, #18202b 40%, #171f2a 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1360px;
            padding-top: 1.25rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        div[data-testid="stTabs"] button {
            border-radius: 7px;
            padding: 0.55rem 0.9rem;
            color: var(--muted);
            font-weight: 650;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: #2a3648;
            color: var(--text);
            border: 1px solid var(--border);
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #1a202b 0%, #151a22 100%);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            min-height: 124px;
            box-shadow: 0 14px 28px rgba(0, 0, 0, 0.18);
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] label p {
            color: var(--muted);
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.2;
        }

        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] div {
            color: var(--text);
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere;
            line-height: 1.05;
            font-size: clamp(1.35rem, 1.7vw, 2.15rem);
        }

        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] div,
        div[data-testid="stMetricDelta"] p {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }

        .hero {
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.25rem 1.45rem;
            margin-bottom: 1.25rem;
            background:
                linear-gradient(135deg, rgba(139, 200, 255, 0.18), transparent 44%),
                linear-gradient(180deg, #253247 0%, #202938 100%);
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.16);
        }

        .hero-kicker {
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }

        .hero-title {
            color: var(--text);
            font-size: clamp(2rem, 3.6vw, 3rem);
            line-height: 1.05;
            font-weight: 800;
            margin-bottom: 0.6rem;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 1.04rem;
            line-height: 1.55;
            max-width: 920px;
        }

        .insight-card {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            background: var(--panel);
            min-height: 132px;
            height: auto;
            box-shadow: 0 10px 22px rgba(0, 0, 0, 0.12);
        }

        .insight-label {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.45rem;
        }

        .insight-value {
            color: var(--text);
            font-size: 1.55rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.35rem;
        }

        .insight-text {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.45;
        }

        .metric-card {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            background: linear-gradient(180deg, #243044 0%, #202938 100%);
            min-height: 132px;
            height: 132px;
            margin-bottom: 1rem;
            box-shadow: 0 10px 22px rgba(0, 0, 0, 0.12);
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.25;
            margin-bottom: 0.55rem;
            overflow-wrap: anywhere;
        }

        .metric-value {
            color: var(--text);
            font-size: clamp(1.45rem, 1.65vw, 2rem);
            font-weight: 800;
            line-height: 1.08;
            overflow-wrap: anywhere;
            word-break: normal;
        }

        .metric-value-long {
            font-size: clamp(1.05rem, 1.15vw, 1.38rem);
            line-height: 1.12;
        }

        .metric-delta {
            color: var(--accent-2);
            font-size: 0.85rem;
            line-height: 1.25;
            margin-top: 0.65rem;
            overflow-wrap: anywhere;
        }

        .section-note {
            border-left: 4px solid var(--accent);
            background: rgba(124, 196, 255, 0.08);
            color: var(--text);
            padding: 0.9rem 1rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
        }

        .risk-note {
            border-left: 4px solid var(--accent-3);
            background: rgba(242, 184, 75, 0.1);
            color: var(--text);
            padding: 0.9rem 1rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
        }

        .success-note {
            border-left: 4px solid var(--accent-2);
            background: rgba(79, 178, 134, 0.1);
            color: var(--text);
            padding: 0.9rem 1rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Machine learning proof of concept</div>
            <div class="hero-title">Walmart Demand Forecasting</div>
            <div class="hero-copy">
                A business and data view of a supervised forecasting model for
                daily store-item demand. The dashboard explains the business
                use case, the modeling choices, the evidence, and how a manager
                could use the forecast.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _insight_card(label: str, value: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-label">{label}</div>
            <div class="insight-value">{value}</div>
            <div class="insight-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, delta: str | None = None) -> None:
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    value_class = "metric-value metric-value-long" if len(str(value)) > 18 else "metric-value"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="{value_class}">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _note(text: str, variant: str = "section") -> None:
    class_name = {
        "section": "section-note",
        "risk": "risk-note",
        "success": "success-note",
    }[variant]
    st.markdown(f'<div class="{class_name}">{text}</div>', unsafe_allow_html=True)


def _style_chart(fig, height: int | None = None):
    fig.update_layout(
        template="plotly_dark",
        colorway=CHART_COLORS,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F8FAFC", "size": 13},
        title={"font": {"size": 20, "color": "#F8FAFC"}},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": "#F8FAFC", "size": 13},
            "bgcolor": "rgba(32, 41, 56, 0.86)",
            "bordercolor": "#3B4657",
            "borderwidth": 1,
        },
        margin={"l": 20, "r": 20, "t": 70, "b": 40},
        height=height,
    )
    fig.update_xaxes(gridcolor="#3B4657", zerolinecolor="#3B4657")
    fig.update_yaxes(gridcolor="#3B4657", zerolinecolor="#3B4657")
    return fig


def _metric_delta(best_value: float, baseline_value: float, higher_is_better: bool) -> str:
    if baseline_value == 0:
        return ""

    relative_change = (best_value - baseline_value) / abs(baseline_value)
    if not higher_is_better:
        relative_change *= -1
    return f"{relative_change:+.1%} vs lag baseline"


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overview_path = Path(RESULTS_DIR) / "data_overview.csv"
    metrics_path = Path(RESULTS_DIR) / "model_metrics.csv"
    predictions_path = Path(RESULTS_DIR) / "best_model_test_predictions.csv"
    feature_importance_path = Path(RESULTS_DIR) / "feature_importance.csv"
    monthly_seasonality_path = Path(RESULTS_DIR) / "monthly_seasonality.csv"

    missing = [
        path
        for path in [
            overview_path,
            metrics_path,
            predictions_path,
            feature_importance_path,
            monthly_seasonality_path,
        ]
        if not path.exists()
    ]
    if missing:
        missing_names = ", ".join(str(path.relative_to(RESULTS_DIR.parent)) for path in missing)
        st.error(
            "Missing dashboard input files. Run `python scripts/prepare_data.py`, "
            "`python scripts/train_models.py`, and `python scripts/generate_plots.py`."
        )
        st.caption(f"Missing: {missing_names}")
        st.stop()

    overview = _read_csv(overview_path)
    metrics = _read_csv(metrics_path)
    predictions = _read_csv(predictions_path, parse_dates=["date"])
    importance = _read_csv(feature_importance_path)
    monthly_seasonality = _read_csv(monthly_seasonality_path)
    return overview, metrics, predictions, importance, monthly_seasonality


def _overview_values(overview: pd.DataFrame) -> dict[str, str]:
    return dict(zip(overview["metric"], overview["value"], strict=False))


def _best_and_baseline(metrics: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    best = metrics.sort_values("rmsle").iloc[0]
    baseline = metrics.loc[metrics["model_id"] == "lag_blend_baseline"]
    if baseline.empty:
        baseline = metrics.sort_values("rmsle").tail(1)
    return best, baseline.iloc[0]


def _daily_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby("date", as_index=False)
        .agg(
            realized_units=("actual_units", "sum"),
            predicted_units=("predicted_units", "sum"),
            absolute_error=("absolute_error", "sum"),
        )
        .sort_values("date")
    )


def _portfolio_bias(daily: pd.DataFrame) -> float:
    total_realized = daily["realized_units"].sum()
    total_predicted = daily["predicted_units"].sum()
    if total_realized == 0:
        return 0.0
    return (total_predicted - total_realized) / total_realized


def _render_header() -> None:
    st.set_page_config(page_title="Walmart Demand Forecasting", layout="wide")
    _inject_theme()
    _render_hero()


def _render_executive_summary(
    overview: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    values = _overview_values(overview)
    best, baseline = _best_and_baseline(metrics)
    daily = _daily_predictions(predictions)
    bias = _portfolio_bias(daily)
    positive_share = float((predictions["actual_units"] > 0).mean())

    st.subheader("Executive Summary")
    st.write(
        "The project tests whether Walmart can forecast daily unit demand at the "
        "store-product level using historical sales, calendar patterns, and local "
        "weather context. The strongest model is useful for planning at portfolio "
        "and product/store levels, while exact demand spikes remain the hardest "
        "part of the problem. Because most planning inputs, such as calendar "
        "features, recent sales history, and weather forecasts, can be available "
        "several days before the sales day, this type of forecast could give "
        "Walmart enough lead time to move additional stock from warehouses to "
        "stores before demand materializes."
    )

    kpis = st.columns(5)
    with kpis[0]:
        _metric_card("Model", best["model_name"])
    with kpis[1]:
        _metric_card("R2", f"{best['r2']:.3f}")
    with kpis[2]:
        _metric_card(
            "RMSLE",
            f"{best['rmsle']:.3f}",
            _metric_delta(best["rmsle"], baseline["rmsle"], higher_is_better=False),
        )
    with kpis[3]:
        _metric_card("Demand-day MAE", f"{best['positive_sales_mae']:.2f}")
    with kpis[4]:
        _metric_card("Bias", _format_percent(bias))

    insight_cols = st.columns(3)
    with insight_cols[0]:
        _insight_card(
            "Historical rows",
            f"{int(values.get('rows', 0)):,}",
            "Store-product-day observations available for supervised learning.",
        )
    with insight_cols[1]:
        _insight_card(
            "Zero-sales rows",
            _format_percent(float(values["zero_sales_share"])),
            "Most rows contain no demand, so average error alone is misleading.",
        )
    with insight_cols[2]:
        _insight_card(
            "Positive-sales test rows",
            _format_percent(positive_share),
            "The hardest and most operationally important rows to forecast.",
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
    st.plotly_chart(_style_chart(daily_chart, height=460), width="stretch")
    st.caption(
        "This chart shows the model's usefulness for aggregate planning: it should "
        "track timing and scale of demand, not only minimize row-level error."
    )


def _render_data_context(
    overview: pd.DataFrame,
    predictions: pd.DataFrame,
    monthly_seasonality: pd.DataFrame,
) -> None:
    values = _overview_values(overview)

    st.subheader("Business Context And Data Reality")
    st.write(
        "The operational question is simple: how many units should a store expect "
        "to sell for each item? Better forecasts can "
        "reduce stockouts, reduce excess inventory, and help managers focus on the "
        "products and stores that actually drive volume."
    )

    coverage = st.columns(5)
    with coverage[0]:
        _metric_card("Stores", values.get("stores", ""))
    with coverage[1]:
        _metric_card("Items", values.get("items", ""))
    with coverage[2]:
        _metric_card("Stations", values.get("weather_stations", ""))
    with coverage[3]:
        _metric_card("Dates", f"{values.get('date_min')} to {values.get('date_max')}")
    with coverage[4]:
        _metric_card("Zero rows", _format_percent(float(values["zero_sales_share"])))

    _note(
        "The target is extremely sparse: most store-item-day rows have zero sales. "
        "That means average error alone can be misleading, because a model can "
        "look good by predicting values close to zero. This is why the project "
        "reports positive-sales MAE separately.",
        variant="risk",
    )

    item_volume = (
        predictions.groupby("item_nbr", as_index=False)["actual_units"]
        .sum()
        .sort_values("actual_units", ascending=False)
    )
    store_volume = (
        predictions.groupby("store_nbr", as_index=False)["actual_units"]
        .sum()
        .sort_values("actual_units", ascending=False)
    )
    top_item_share = item_volume.head(10)["actual_units"].sum() / item_volume["actual_units"].sum()
    top_store_share = (
        store_volume.head(10)["actual_units"].sum() / store_volume["actual_units"].sum()
    )

    concentration = st.columns(2)
    with concentration[0]:
        _metric_card("Top products", _format_percent(top_item_share))
    with concentration[1]:
        _metric_card("Top stores", _format_percent(top_store_share))

    st.caption(
        "The Kaggle dataset anonymizes stores and products. It does not include "
        "store city, state, latitude, or longitude, so the dashboard focuses on "
        "demand concentration and calendar patterns instead of maps."
    )

    product_rank = item_volume.reset_index(drop=True).copy()
    product_rank["rank"] = product_rank.index + 1
    product_rank["cumulative_demand_share"] = (
        product_rank["actual_units"].cumsum() / product_rank["actual_units"].sum()
    )
    product_rank["entity"] = "Products"

    store_rank = store_volume.reset_index(drop=True).copy()
    store_rank["rank"] = store_rank.index + 1
    store_rank["cumulative_demand_share"] = (
        store_rank["actual_units"].cumsum() / store_rank["actual_units"].sum()
    )
    store_rank["entity"] = "Stores"

    concentration_curve = pd.concat(
        [
            product_rank[["rank", "cumulative_demand_share", "entity"]],
            store_rank[["rank", "cumulative_demand_share", "entity"]],
        ],
        ignore_index=True,
    )
    concentration_chart = px.line(
        concentration_curve,
        x="rank",
        y="cumulative_demand_share",
        color="entity",
        markers=True,
        title="Demand concentration by ranked products and stores",
        labels={
            "rank": "Ranked product/store count",
            "cumulative_demand_share": "Cumulative share of units",
            "entity": "",
        },
    )
    concentration_chart.update_yaxes(tickformat=".0%")

    weekday = predictions.copy()
    weekday["weekday"] = weekday["date"].dt.day_name()
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekday_summary = (
        weekday.groupby("weekday", as_index=False)["actual_units"]
        .sum()
        .assign(
            weekday=lambda frame: pd.Categorical(
                frame["weekday"],
                categories=weekday_order,
                ordered=True,
            )
        )
        .sort_values("weekday")
    )
    weekday_chart = px.bar(
        weekday_summary,
        x="weekday",
        y="actual_units",
        title="Test-period demand by day of week",
        labels={"weekday": "Day of week", "actual_units": "Units"},
    )

    monthly_summary = monthly_seasonality.sort_values("month_number").copy()
    monthly_chart = px.bar(
        monthly_summary,
        x="month_name",
        y="average_daily_units",
        title="Seasonality: average daily demand by calendar month",
        labels={"month_name": "Month", "average_daily_units": "Average daily units"},
        hover_data={
            "month_number": False,
            "total_units": ":,.0f",
            "observed_days": True,
            "average_daily_units": ":,.0f",
        },
    )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(_style_chart(concentration_chart, height=440), width="stretch")
    with chart_cols[1]:
        st.plotly_chart(_style_chart(weekday_chart, height=440), width="stretch")

    st.plotly_chart(_style_chart(monthly_chart, height=430), width="stretch")

    st.write(
        "Business interpretation: because demand is concentrated, managers should "
        "not treat every product and store equally. The highest-volume segments "
        "deserve the most forecast monitoring, while day-of-week and monthly "
        "patterns from the full history help translate forecasts into "
        "replenishment routines."
    )


def _render_methodology(metrics: pd.DataFrame, importance: pd.DataFrame) -> None:
    best, _ = _best_and_baseline(metrics)

    st.subheader("Methodology And Why It Is Defensible")
    st.write(
        "The modeling design follows a forecasting setup: train on past dates, "
        "evaluate on later dates, and prevent sales features from seeing the same "
        "day's target."
    )

    design_cols = st.columns(2)
    with design_cols[0]:
        st.markdown(
            """
            **Validation design**

            - Train: `2012-01-01` to `2014-06-30`
            - Test: `2014-07-01` to `2014-10-31`
            - Split is chronological, not random
            - Final metrics are calculated on the full test period
            """
        )
    with design_cols[1]:
        st.markdown(
            """
            **Leakage controls**

            - Sales lags use only previous store-item sales
            - Rolling means are shifted before calculation
            - Same-day weather is treated as forecastable context
            - Models are compared against a transparent lag baseline
            """
        )

    st.markdown(
        """
        **Why these models?**

        - The lag baseline checks whether machine learning beats simple historical demand.
        - Poisson regression is appropriate for nonnegative count targets.
        - Histogram Gradient Boosting can capture nonlinear interactions between product,
          store, calendar, lag, and weather features.
        """
    )

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
        title=f"What {best['model_name']} learned: permutation feature importance",
        labels={
            "importance_mae": "Increase in MAE when shuffled",
            "feature_label": "",
            "importance_mae_std": "Repeated-shuffle variation",
        },
    )
    st.plotly_chart(_style_chart(importance_chart, height=560), width="stretch")

    sample_rows = int(importance["sample_rows"].iloc[0])
    positive_share = float(importance["positive_sales_share"].iloc[0])
    st.caption(
        "Feature importance is measured by shuffling one input at a time on a "
        f"test sample of {sample_rows:,} rows. The sample is balanced to include "
        f"{positive_share:.1%} positive-sales rows, making demand-driving features "
        "more visible than they would be in the zero-heavy full dataset."
    )


def _render_performance(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    best, baseline = _best_and_baseline(metrics)

    st.subheader("Model Performance")
    st.write(
        "The model comparison answers two questions: does ML beat a transparent "
        "historical-demand baseline, and is the winning model reliable enough for "
        "business planning?"
    )

    comparison = metrics[
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
    ].copy()
    st.dataframe(comparison, width="stretch", hide_index=True)

    perf_cols = st.columns(4)
    with perf_cols[0]:
        _metric_card(
            "RMSLE lift",
            _metric_delta(best["rmsle"], baseline["rmsle"], higher_is_better=False),
        )
    with perf_cols[1]:
        _metric_card(
            "RMSE lift",
            _metric_delta(best["rmse"], baseline["rmse"], higher_is_better=False),
        )
    with perf_cols[2]:
        _metric_card(
            "Demand-day lift",
            _metric_delta(
                best["positive_sales_mae"],
                baseline["positive_sales_mae"],
                higher_is_better=False,
            ),
        )
    with perf_cols[3]:
        _metric_card("Model R2", f"{best['r2']:.3f}")

    metric_long = metrics.melt(
        id_vars=["model_name"],
        value_vars=["mae", "rmse", "rmsle", "positive_sales_mae"],
        var_name="metric",
        value_name="value",
    )
    metric_chart = px.bar(
        metric_long,
        x="metric",
        y="value",
        color="model_name",
        barmode="group",
        title="Model comparison on the chronological test period",
        labels={"metric": "Metric", "value": "Value", "model_name": ""},
    )
    st.plotly_chart(_style_chart(metric_chart, height=450), width="stretch")

    st.markdown(
        """
        **Interpretation**

        Histogram Gradient Boosting is the best model because it reduces both
        relative error and positive-sales error versus the lag baseline. The lag
        baseline remains strong, which confirms that recent store-item demand is
        the most important signal. Poisson regression is theoretically sensible
        for count data, but it underfits the nonlinear demand patterns in this
        dataset.
        """
    )

    diagnostics_cols = st.columns(2)
    plot_sample = predictions.sample(n=min(15_000, len(predictions)), random_state=42)
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
        st.plotly_chart(_style_chart(scatter, height=470), width="stretch")

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
        st.plotly_chart(_style_chart(error_hist, height=470), width="stretch")


def _render_operational_demo(predictions: pd.DataFrame) -> None:
    st.subheader("Operational Demo")
    st.write(
        "This view shows how a manager could inspect a high-volume product, choose "
        "a store, and compare realized versus predicted demand for one week."
    )

    default_item = 37
    default_store = 17
    default_week_start = pd.Timestamp("2014-07-28")

    item_volume = predictions.groupby("item_nbr")["actual_units"].sum().sort_values(ascending=False)
    item_options = item_volume.index.astype(int).tolist()
    selected_item = st.selectbox(
        "Product",
        item_options,
        index=item_options.index(default_item) if default_item in item_options else 0,
        help="Products are sorted by realized test-period unit volume.",
    )

    item_predictions = predictions[predictions["item_nbr"] == selected_item].copy()
    store_ranking = (
        item_predictions.groupby("store_nbr")["actual_units"]
        .sum()
        .sort_values(ascending=False)
    )
    store_options = store_ranking.index.astype(int).tolist()
    selected_store = st.selectbox(
        "Store",
        store_options,
        index=store_options.index(default_store) if selected_item == default_item else 0,
        help="Stores are sorted by realized test-period volume for the selected product.",
    )

    item_store_predictions = item_predictions[
        item_predictions["store_nbr"] == selected_store
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
    if (
        selected_item == default_item
        and selected_store == default_store
        and default_week_start in week_options
    ):
        default_week = default_week_start
    selected_week = st.selectbox(
        "Week",
        week_options,
        index=week_options.index(default_week),
        format_func=lambda value: (
            f"{value.date()} to {(value + pd.Timedelta(days=6)).date()}"
        ),
        help="Default is the highest-volume week for the selected store and product.",
    )

    week_predictions = (
        item_store_predictions[item_store_predictions["week_start"] == selected_week]
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

    week_realized = week_predictions["actual_units"].sum()
    week_predicted = week_predictions["predicted_units"].sum()
    week_error = abs(week_realized - week_predicted)
    week_bias = (week_predicted - week_realized) / week_realized if week_realized else 0

    week_cols = st.columns(4)
    with week_cols[0]:
        _metric_card("Realized", _format_units(week_realized))
    with week_cols[1]:
        _metric_card("Predicted", _format_units(week_predicted))
    with week_cols[2]:
        _metric_card("Abs. error", _format_units(week_error))
    with week_cols[3]:
        _metric_card("Bias", _format_percent(week_bias))

    line = px.line(
        weekly_long,
        x="date",
        y="units",
        color="series",
        markers=True,
        title=f"Product {selected_item}, store {selected_store}: one-week forecast",
        labels={"date": "Date", "units": "Units", "series": ""},
    )
    st.plotly_chart(_style_chart(line, height=460), width="stretch")

    if week_bias < -0.15:
        _note(
            "Operational reading: the model underpredicted this week. In a real "
            "inventory process, this store-product pair would need safety stock or "
            "manager review.",
            variant="risk",
        )
    elif week_bias > 0.15:
        _note(
            "Operational reading: the model overpredicted this week. This could "
            "increase overstock risk if used without business review.",
            variant="risk",
        )
    else:
        _note(
            "Operational reading: weekly forecast bias is controlled for this "
            "store-product-week example.",
            variant="success",
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
        width="stretch",
        hide_index=True,
    )


def _render_limitations() -> None:
    st.subheader("Limitations And Next Steps")
    st.markdown(
        """
        **What the result supports**

        - The model is useful as a short-term planning aid.
        - It performs best when aggregating across many store-item rows.
        - It gives managers a ranked view of where demand is likely to occur.

        **What the result does not support**

        - It should not automatically place replenishment orders without business rules.
        - It should not be assumed to generalize to modern Walmart demand without retraining.
        - It does not model promotions, holidays, stockouts, price changes, or supply limits.

        **Most valuable next improvements**

        - Add promotion and holiday features.
        - Calibrate the model against real weather forecasts rather than historical observations.
        - Add stockout-aware evaluation if inventory availability data becomes available.
        - Monitor forecast bias by store, product, and week after deployment.
        """
    )


def build_app() -> None:
    """Build the Streamlit management dashboard for the ML proof of concept."""
    _render_header()
    overview, metrics, predictions, importance, monthly_seasonality = _load_inputs()

    tabs = st.tabs(
        [
            "Executive Summary",
            "Data Context",
            "Methodology",
            "Performance",
            "Operational Demo",
            "Limitations",
        ]
    )

    with tabs[0]:
        _render_executive_summary(overview, metrics, predictions)
    with tabs[1]:
        _render_data_context(overview, predictions, monthly_seasonality)
    with tabs[2]:
        _render_methodology(metrics, importance)
    with tabs[3]:
        _render_performance(metrics, predictions)
    with tabs[4]:
        _render_operational_demo(predictions)
    with tabs[5]:
        _render_limitations()

    st.caption(
        f"Raw data folder expected locally at `{WALMART_RAW_DIR}`. Data files are "
        "ignored by Git; see README.md for download instructions."
    )


if __name__ == "__main__":
    build_app()
