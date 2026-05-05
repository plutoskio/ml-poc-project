import pandas as pd

from features import clean_weather
from features import add_sales_lag_features


def test_clean_weather_converts_trace_and_missing_values():
    raw = pd.DataFrame(
        {
            "station_nbr": [1, 1],
            "date": ["2014-01-01", "2014-01-02"],
            "tmax": ["M", "40"],
            "tmin": ["20", "30"],
            "tavg": ["30", "35"],
            "dewpoint": ["10", "20"],
            "wetbulb": ["20", "25"],
            "heat": ["35", "30"],
            "cool": ["0", "0"],
            "codesum": ["RA FZFG", ""],
            "snowfall": ["T", "0.0"],
            "preciptotal": ["T", "M"],
            "stnpressure": ["29.1", "29.2"],
            "sealevel": ["30.1", "30.2"],
            "resultspeed": ["3.0", "4.0"],
            "resultdir": ["20", "21"],
            "avgspeed": ["5.0", "6.0"],
        }
    )

    cleaned = clean_weather(raw)

    assert cleaned["preciptotal"].iloc[0] == 0.005
    assert cleaned["snowfall"].iloc[0] == 0.005
    assert cleaned["has_rain"].iloc[0] == 1
    assert cleaned["has_freezing"].iloc[0] == 1
    assert cleaned["tmax"].isna().sum() == 0


def test_sales_lag_features_do_not_use_current_day_target():
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "store_nbr": [1] * 8,
            "item_nbr": [1] * 8,
            "units": [0, 1, 2, 3, 4, 5, 6, 99],
        }
    )

    transformed = add_sales_lag_features(raw)
    last_row = transformed.sort_values("date").iloc[-1]

    assert last_row["lag_1"] == 6
    assert last_row["lag_7"] == 0
    assert last_row["rolling_mean_7"] == 3
