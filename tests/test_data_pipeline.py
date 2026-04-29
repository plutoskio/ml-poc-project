import pandas as pd

from src.config import DATA_DIR
from src.data import LEAKAGE_COLUMNS, feature_columns, load_modeling_dataset


def test_feature_columns_exclude_known_leakage_columns() -> None:
    dataset = load_modeling_dataset()
    features = feature_columns(dataset)

    assert not set(features) & LEAKAGE_COLUMNS
    assert not {"EMA_10", "EMA_20", "EMA_50", "EMA_200"} & set(features)
    assert "price_vs_trailing_ema_20" in features
    assert dataset[features].isna().sum().sum() == 0


def test_target_is_next_day_direction_per_index() -> None:
    dataset = load_modeling_dataset()

    for _, index_frame in dataset.groupby("index_name"):
        recomputed_next_return = (
            index_frame["Price"].shift(-1) / index_frame["Price"] - 1
        )
        comparable = recomputed_next_return.notna()

        pd.testing.assert_series_equal(
            index_frame.loc[comparable, "next_return"].reset_index(drop=True),
            recomputed_next_return.loc[comparable].reset_index(drop=True),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            index_frame.loc[comparable, "target"].reset_index(drop=True),
            (recomputed_next_return.loc[comparable] > 0)
            .astype(int)
            .reset_index(drop=True),
            check_names=False,
        )


def test_price_features_are_lagged_one_trading_day() -> None:
    raw = pd.read_csv(DATA_DIR / "combined_dataframe_DJI.csv")
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values("Date").reset_index(drop=True)
    raw["raw_return_1d"] = raw["Price"].pct_change(1)

    dataset = load_modeling_dataset()
    dji = dataset.loc[dataset["index_name"] == "DJI"].copy()
    sample = dji.iloc[0]
    raw_position = raw.index[raw["Date"] == sample["Date"]][0]

    assert sample["Date"] == pd.Timestamp("2010-03-31")
    assert sample["return_1d"] == raw.loc[raw_position - 1, "raw_return_1d"]
