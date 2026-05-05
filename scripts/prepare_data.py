from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import PROCESSED_DATASET_PATH, RESULTS_DIR, TARGET_COLUMN, TEST_START_DATE  # noqa: E402
from features import clean_weather, load_raw_tables, save_modeling_dataset  # noqa: E402


def write_eda_outputs(df: pd.DataFrame) -> None:
    """Write compact EDA tables used by the notebook and Streamlit app."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    overview = pd.DataFrame(
        [
            {"metric": "rows", "value": len(df)},
            {"metric": "stores", "value": df["store_nbr"].nunique()},
            {"metric": "items", "value": df["item_nbr"].nunique()},
            {"metric": "weather_stations", "value": df["station_nbr"].nunique()},
            {"metric": "date_min", "value": df["date"].min().date().isoformat()},
            {"metric": "date_max", "value": df["date"].max().date().isoformat()},
            {"metric": "zero_sales_share", "value": float((df[TARGET_COLUMN] == 0).mean())},
            {"metric": "positive_sales_rows", "value": int((df[TARGET_COLUMN] > 0).sum())},
            {"metric": "chronological_test_start", "value": TEST_START_DATE},
        ]
    )
    overview.to_csv(RESULTS_DIR / "data_overview.csv", index=False)

    monthly_sales = (
        df.assign(month_period=df["date"].dt.to_period("M").astype(str))
        .groupby("month_period", as_index=False)[TARGET_COLUMN]
        .sum()
        .rename(columns={TARGET_COLUMN: "total_units"})
    )
    monthly_sales.to_csv(RESULTS_DIR / "monthly_sales.csv", index=False)

    top_items = (
        df.groupby("item_nbr", as_index=False)[TARGET_COLUMN]
        .sum()
        .sort_values(TARGET_COLUMN, ascending=False)
        .head(20)
        .rename(columns={TARGET_COLUMN: "total_units"})
    )
    top_items.to_csv(RESULTS_DIR / "top_items.csv", index=False)

    top_stores = (
        df.groupby("store_nbr", as_index=False)[TARGET_COLUMN]
        .sum()
        .sort_values(TARGET_COLUMN, ascending=False)
        .head(20)
        .rename(columns={TARGET_COLUMN: "total_units"})
    )
    top_stores.to_csv(RESULTS_DIR / "top_stores.csv", index=False)


def write_weather_missingness() -> None:
    """Document missing/coded weather values before cleaning."""
    _, _, raw_weather = load_raw_tables()
    raw_weather.columns = [col.strip().replace('"', "") for col in raw_weather.columns]
    missing_tokens = {"", "M", "-", "T"}

    rows = []
    for col in raw_weather.columns:
        values = raw_weather[col].astype(str).str.strip()
        token_missing = values.isin(missing_tokens)
        rows.append(
            {
                "column": col,
                "missing_or_coded_rows": int(token_missing.sum()),
                "missing_or_coded_share": float(token_missing.mean()),
            }
        )

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "weather_missingness.csv", index=False)

    cleaned_weather = clean_weather(raw_weather)
    cleaned_weather.describe(include="all").transpose().to_csv(
        RESULTS_DIR / "cleaned_weather_summary.csv"
    )


def main() -> None:
    df = save_modeling_dataset(PROCESSED_DATASET_PATH)
    write_eda_outputs(df)
    write_weather_missingness()
    print(f"Processed dataset saved to {PROCESSED_DATASET_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Zero-sales share: {(df[TARGET_COLUMN] == 0).mean():.2%}")


if __name__ == "__main__":
    main()

