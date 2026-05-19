from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
WALMART_RAW_DIR = RAW_DATA_DIR / "walmart-recruiting-sales-in-stormy-weather"

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = ROOT_DIR / "plots"
LOGS_DIR = ROOT_DIR / "logs"

TARGET_COLUMN = "units"
RANDOM_STATE = 42
TEST_START_DATE = "2014-07-01"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "modeling_dataset.parquet"
MAX_REASONABLE_UNITS = 6000

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

FEATURE_COLUMNS = [
    "store_nbr",
    "item_nbr",
    "station_nbr",
    "year",
    "month",
    "dayofweek",
    "dayofmonth",
    "weekofyear",
    "quarter",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "days_since_start",
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_mean_90",
    "tmax",
    "tmin",
    "tavg",
    "dewpoint",
    "wetbulb",
    "heat",
    "cool",
    "preciptotal",
    "snowfall",
    "stnpressure",
    "sealevel",
    "resultspeed",
    "resultdir",
    "avgspeed",
    "has_rain",
    "has_snow",
    "has_fog",
    "has_thunder",
    "has_freezing",
    "weather_event_count",
]

CATEGORICAL_COLUMNS = [
    "store_nbr",
    "item_nbr",
    "station_nbr",
    "month",
    "dayofweek",
    "weekofyear",
    "quarter",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "has_rain",
    "has_snow",
    "has_fog",
    "has_thunder",
    "has_freezing",
]

NUMERIC_COLUMNS = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS]

CATEGORY_LEVELS = {
    "store_nbr": list(range(1, 46)),
    "item_nbr": list(range(1, 112)),
    "station_nbr": list(range(1, 21)),
    "month": list(range(1, 13)),
    "dayofweek": list(range(0, 7)),
    "weekofyear": list(range(1, 54)),
    "quarter": list(range(1, 5)),
    "is_weekend": [0, 1],
    "is_month_start": [0, 1],
    "is_month_end": [0, 1],
    "has_rain": [0, 1],
    "has_snow": [0, 1],
    "has_fog": [0, 1],
    "has_thunder": [0, 1],
    "has_freezing": [0, 1],
}

MODELS = {
    "lag_blend_baseline": {
        "name": "Lag Blend Baseline",
        "description": "Transparent baseline using recent store-item sales lags and rolling means.",
        "path": MODELS_DIR / "lag_blend_baseline.joblib",
    },
    "poisson_regression": {
        "name": "Weighted Poisson Regression",
        "description": "Count model with one-hot categorical features and sampling weights.",
        "path": MODELS_DIR / "poisson_regression.joblib",
    },
    "hist_gradient_boosting": {
        "name": "Histogram Gradient Boosting",
        "description": "Non-linear tree model trained on engineered date, lag, and weather features.",
        "path": MODELS_DIR / "hist_gradient_boosting.joblib",
    },
}
