"""Dataset loading and feature engineering for the quant-finance POC.

The project predicts next-day index direction for DJI, IXIC, and NYSE.  The
important modeling constraint is chronological integrity: features at date ``t``
can only use information available at or before ``t``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import DATA_DIR

DATA_FILES = {
    "DJI": DATA_DIR / "combined_dataframe_DJI.csv",
    "IXIC": DATA_DIR / "combined_dataframe_IXIC.csv",
    "NYSE": DATA_DIR / "combined_dataframe_NYSE.csv",
}

PROCESSED_MODELING_DATASET_FILE = DATA_DIR / "processed_modeling_dataset.csv"
PROCESSED_TRAIN_FILE = DATA_DIR / "processed_train.csv"
PROCESSED_VALIDATION_FILE = DATA_DIR / "processed_validation.csv"
PROCESSED_TEST_FILE = DATA_DIR / "processed_test.csv"

LEAKAGE_COLUMNS = {
    "EMA_10",
    "EMA_20",
    "EMA_50",
    "EMA_200",
    "mom",
    "mom1",
    "mom2",
    "mom3",
    "ROC_5",
    "ROC_10",
    "ROC_15",
    "ROC_20",
}

CORE_ENGINEERED_COLUMNS = {
    "return_20d",
    "volatility_20d",
    "drawdown_60d",
}

RAW_NON_FEATURE_COLUMNS = {
    "Date",
    "Name",
    "Price",
    "index_name",
    "next_return",
    "target",
    "split",
}

TRAIN_END_DATE = pd.Timestamp("2019-12-31")
VALIDATION_END_DATE = pd.Timestamp("2021-12-31")
TEST_START_DATE = pd.Timestamp("2022-01-01")


def _load_single_index(index_name: str, path: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file for {index_name}: {path}")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["index_name"] = index_name

    feature_frame = pd.DataFrame(index=df.index)
    feature_frame["return_1d"] = df["Price"].pct_change(1)
    feature_frame["return_2d"] = df["Price"].pct_change(2)
    feature_frame["return_5d"] = df["Price"].pct_change(5)
    feature_frame["return_10d"] = df["Price"].pct_change(10)
    feature_frame["return_20d"] = df["Price"].pct_change(20)

    feature_frame["volatility_5d"] = feature_frame["return_1d"].rolling(5).std()
    feature_frame["volatility_10d"] = feature_frame["return_1d"].rolling(10).std()
    feature_frame["volatility_20d"] = feature_frame["return_1d"].rolling(20).std()

    feature_frame["volatility_ratio_5d_20d"] = (
        feature_frame["volatility_5d"] / feature_frame["volatility_20d"]
    )
    feature_frame["return_5d_to_volatility_20d"] = (
        feature_frame["return_5d"] / feature_frame["volatility_20d"]
    )
    feature_frame["return_20d_to_volatility_20d"] = (
        feature_frame["return_20d"] / feature_frame["volatility_20d"]
    )

    rolling_high_20d = df["Price"].rolling(20).max()
    rolling_high_60d = df["Price"].rolling(60).max()
    rolling_low_20d = df["Price"].rolling(20).min()
    feature_frame["drawdown_20d"] = df["Price"] / rolling_high_20d - 1
    feature_frame["drawdown_60d"] = df["Price"] / rolling_high_60d - 1
    feature_frame["price_vs_20d_low"] = df["Price"] / rolling_low_20d - 1

    trailing_emas: dict[int, pd.Series] = {}
    for window in [10, 20, 50, 200]:
        trailing_ema = df["Price"].ewm(span=window, adjust=False).mean()
        trailing_emas[window] = trailing_ema
        feature_frame[f"price_vs_trailing_ema_{window}"] = (
            df["Price"] / trailing_ema - 1
        )

    feature_frame["trailing_ema_10_vs_50"] = trailing_emas[10] / trailing_emas[50] - 1
    feature_frame["trailing_ema_20_vs_200"] = trailing_emas[20] / trailing_emas[200] - 1

    target_frame = pd.DataFrame(
        {
            "next_return": df["Price"].shift(-1) / df["Price"] - 1,
        },
        index=df.index,
    )
    target_frame["target"] = (target_frame["next_return"] > 0).astype(int)
    return pd.concat([df, feature_frame, target_frame], axis=1)


def _assign_split(date_series: pd.Series) -> pd.Series:
    split = pd.Series("test", index=date_series.index)
    split[date_series <= TRAIN_END_DATE] = "train"
    split[(date_series > TRAIN_END_DATE) & (date_series <= VALIDATION_END_DATE)] = (
        "validation"
    )
    split[date_series >= TEST_START_DATE] = "test"
    return split


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])
    df["split"] = _assign_split(df["Date"])

    numeric_columns = [
        column
        for column in df.columns
        if column not in RAW_NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    # Fill missing market data within each index using only current/past values,
    # lag all feature columns by one trading day to avoid using same-close data
    # for a close-to-close next-day return, then use training medians for gaps.
    df[numeric_columns] = df.groupby("index_name", group_keys=False)[
        numeric_columns
    ].ffill()
    df[numeric_columns] = df.groupby("index_name", group_keys=False)[
        numeric_columns
    ].shift(1)

    core_columns = [c for c in CORE_ENGINEERED_COLUMNS if c in df.columns]
    if core_columns:
        df = df.dropna(subset=core_columns)

    train_medians = df.loc[df["split"] == "train", numeric_columns].median()
    df[numeric_columns] = df[numeric_columns].fillna(train_medians)
    df[numeric_columns] = df[numeric_columns].fillna(0)

    index_dummies = pd.get_dummies(df["index_name"], prefix="index", dtype=int)
    return pd.concat([df, index_dummies], axis=1)


def load_modeling_dataset() -> pd.DataFrame:
    """Return the complete clean supervised modeling dataset.

    The returned DataFrame includes feature columns plus metadata columns useful
    for training, validation, testing, and backtesting:
    ``Date``, ``index_name``, ``next_return``, ``target``, and ``split``.
    """

    raw_frames = [
        _load_single_index(index_name, path) for index_name, path in DATA_FILES.items()
    ]
    dataset = pd.concat(raw_frames, ignore_index=True)
    dataset = _prepare_features(dataset)
    dataset = dataset.dropna(subset=["next_return", "target"])
    return dataset.sort_values(["Date", "index_name"]).reset_index(drop=True)


def save_processed_datasets() -> dict[str, Any]:
    """Save the processed full/train/validation/test datasets to ``data/``."""

    dataset = load_modeling_dataset()
    outputs = {
        "modeling": (PROCESSED_MODELING_DATASET_FILE, dataset),
        "train": (PROCESSED_TRAIN_FILE, dataset.loc[dataset["split"] == "train"]),
        "validation": (
            PROCESSED_VALIDATION_FILE,
            dataset.loc[dataset["split"] == "validation"],
        ),
        "test": (PROCESSED_TEST_FILE, dataset.loc[dataset["split"] == "test"]),
    }

    for _, (path, frame) in outputs.items():
        frame.to_csv(path, index=False)

    return {name: path for name, (path, _) in outputs.items()}


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Return model input columns from a prepared modeling dataset."""

    return [
        column
        for column in dataset.columns
        if column not in RAW_NON_FEATURE_COLUMNS
        and column not in LEAKAGE_COLUMNS
        and pd.api.types.is_numeric_dtype(dataset[column])
    ]


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return the dataset split used for model evaluation.

    Expected return value:
        A tuple ``(X_train, X_test, y_train, y_test)``.

    Constraints:
    - ``X_train`` and ``X_test`` must contain feature data in a format accepted
      by the trained models stored in ``config.MODELS``.
    - ``y_train`` and ``y_test`` must contain the corresponding targets.
    - ``y_test`` must align with the predictions produced by each loaded model.

    Typical choices for the return types are ``pandas.DataFrame`` /
    ``pandas.Series`` or ``numpy.ndarray``.
    """

    dataset = load_modeling_dataset()
    columns = feature_columns(dataset)

    train_mask = dataset["split"].isin(["train", "validation"])
    test_mask = dataset["split"] == "test"

    X_train = dataset.loc[train_mask, columns].copy()
    X_test = dataset.loc[test_mask, columns].copy()
    y_train = dataset.loc[train_mask, "target"].copy()
    y_test = dataset.loc[test_mask, "target"].copy()

    if X_train.empty or X_test.empty:
        raise ValueError("Chronological split produced an empty train or test set.")

    return X_train, X_test, y_train, y_test
