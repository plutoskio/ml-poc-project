from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DATASET_PATH, WALMART_RAW_DIR


WEATHER_NUMERIC_COLUMNS = [
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
]


def clean_weather(raw_weather: pd.DataFrame) -> pd.DataFrame:
    """Clean NOAA-style weather values and create compact weather-event flags."""
    weather = raw_weather.copy()
    weather.columns = [col.strip().replace('"', "") for col in weather.columns]
    weather["date"] = pd.to_datetime(weather["date"])
    weather["station_nbr"] = weather["station_nbr"].astype("int16")

    codesum = weather["codesum"].fillna("").astype(str).str.strip()
    weather["has_rain"] = codesum.str.contains("RA", regex=False).astype("int8")
    weather["has_snow"] = codesum.str.contains("SN", regex=False).astype("int8")
    weather["has_fog"] = (
        codesum.str.contains("FG", regex=False) | codesum.str.contains("BR", regex=False)
    ).astype("int8")
    weather["has_thunder"] = codesum.str.contains("TS", regex=False).astype("int8")
    weather["has_freezing"] = codesum.str.contains("FZ", regex=False).astype("int8")
    weather["weather_event_count"] = (
        codesum.replace("", np.nan).str.split().str.len().fillna(0).astype("int8")
    )

    for col in WEATHER_NUMERIC_COLUMNS:
        values = weather[col].astype(str).str.strip()
        if col in {"preciptotal", "snowfall"}:
            values = values.replace({"T": "0.005"})
        values = values.replace({"M": np.nan, "-": np.nan, "": np.nan, "nan": np.nan})
        weather[col] = pd.to_numeric(values, errors="coerce")

    for col in WEATHER_NUMERIC_COLUMNS:
        station_median = weather.groupby("station_nbr")[col].transform("median")
        weather[col] = weather[col].fillna(station_median).fillna(weather[col].median())

    keep_columns = [
        "station_nbr",
        "date",
        *WEATHER_NUMERIC_COLUMNS,
        "has_rain",
        "has_snow",
        "has_fog",
        "has_thunder",
        "has_freezing",
        "weather_event_count",
    ]
    return weather[keep_columns]


def load_raw_tables(raw_dir: Path = WALMART_RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw Walmart sales, store-station key, and weather tables."""
    train = pd.read_csv(
        raw_dir / "train.csv",
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "item_nbr": "int16", "units": "int16"},
    )
    key = pd.read_csv(
        raw_dir / "key.csv",
        dtype={"store_nbr": "int16", "station_nbr": "int16"},
    )
    weather = pd.read_csv(raw_dir / "weather.csv")
    return train, key, weather


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar variables used by all models."""
    output = df.copy()
    first_date = output["date"].min()

    output["year"] = output["date"].dt.year.astype("int16")
    output["month"] = output["date"].dt.month.astype("int8")
    output["dayofweek"] = output["date"].dt.dayofweek.astype("int8")
    output["dayofmonth"] = output["date"].dt.day.astype("int8")
    output["weekofyear"] = output["date"].dt.isocalendar().week.astype("int8")
    output["quarter"] = output["date"].dt.quarter.astype("int8")
    output["is_weekend"] = output["dayofweek"].isin([5, 6]).astype("int8")
    output["is_month_start"] = output["date"].dt.is_month_start.astype("int8")
    output["is_month_end"] = output["date"].dt.is_month_end.astype("int8")
    output["days_since_start"] = (output["date"] - first_date).dt.days.astype("int16")

    return output


def add_sales_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe one-step-ahead lag and rolling features."""
    output = df.sort_values(["store_nbr", "item_nbr", "date"]).copy()
    group_keys = [output["store_nbr"], output["item_nbr"]]
    unit_group = output.groupby(["store_nbr", "item_nbr"], sort=False)["units"]

    output["lag_1"] = unit_group.shift(1)
    output["lag_7"] = unit_group.shift(7)
    output["lag_28"] = unit_group.shift(28)

    shifted_units = unit_group.shift(1)
    for window in [7, 28, 90]:
        output[f"rolling_mean_{window}"] = (
            shifted_units.groupby(group_keys, sort=False)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

    lag_columns = [
        "lag_1",
        "lag_7",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_28",
        "rolling_mean_90",
    ]
    output[lag_columns] = output[lag_columns].fillna(0).astype("float32")

    return output


def build_modeling_dataset(raw_dir: Path = WALMART_RAW_DIR) -> pd.DataFrame:
    """Build the full modeling table from raw sales, key, and weather data."""
    train, key, raw_weather = load_raw_tables(raw_dir)
    weather = clean_weather(raw_weather)

    df = train.merge(key, on="store_nbr", how="left")
    df = df.merge(weather, on=["station_nbr", "date"], how="left")
    df = add_date_features(df)
    df = add_sales_lag_features(df)

    numeric_columns = df.select_dtypes(include=["float64"]).columns
    df[numeric_columns] = df[numeric_columns].astype("float32")
    df = df.sort_values(["date", "store_nbr", "item_nbr"]).reset_index(drop=True)

    return df


def save_modeling_dataset(output_path: Path = PROCESSED_DATASET_PATH) -> pd.DataFrame:
    """Build and save the processed modeling dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_modeling_dataset()
    df.to_parquet(output_path, index=False)
    return df

