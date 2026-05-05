from typing import Any

import pandas as pd

from config import FEATURE_COLUMNS, PROCESSED_DATASET_PATH, TARGET_COLUMN, TEST_START_DATE


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Load the processed dataset and return X_train, X_test, y_train, y_test."""
    if not PROCESSED_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {PROCESSED_DATASET_PATH}. "
            "Run `python scripts/prepare_data.py` first."
        )

    df = pd.read_parquet(PROCESSED_DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])

    train_mask = df["date"] < pd.Timestamp(TEST_START_DATE)
    test_mask = ~train_mask

    X_train = df.loc[train_mask, FEATURE_COLUMNS].copy()
    X_test = df.loc[test_mask, FEATURE_COLUMNS].copy()
    y_train = df.loc[train_mask, TARGET_COLUMN].copy()
    y_test = df.loc[test_mask, TARGET_COLUMN].copy()

    return X_train, X_test, y_train, y_test
