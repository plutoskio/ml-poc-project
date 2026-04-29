import pandas as pd
import pytest

from src.backtest import (
    choose_threshold,
    make_long_cash_strategy_frame,
    portfolio_daily_returns,
    strategy_metrics,
)


def test_long_cash_strategy_uses_threshold_and_transaction_costs() -> None:
    predictions = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]),
            "index_name": ["DJI", "DJI", "DJI"],
            "next_return": [0.01, -0.02, 0.03],
            "p_up": [0.70, 0.40, 0.80],
        }
    )

    frame = make_long_cash_strategy_frame(
        predictions,
        threshold=0.60,
        cost_per_trade=0.001,
    )

    assert frame["signal"].tolist() == [1, 0, 1]
    assert frame["trade"].tolist() == [1, 1, 1]
    assert frame["strategy_return"].tolist() == pytest.approx([0.009, -0.001, 0.029])


def test_portfolio_daily_returns_equal_weight_multiple_indices() -> None:
    strategy_frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-01-02", "2023-01-02"]),
            "index_name": ["DJI", "IXIC"],
            "signal": [1, 0],
            "trade": [1, 0],
            "gross_strategy_return": [0.02, 0.0],
            "strategy_return": [0.019, 0.0],
        }
    )

    daily = portfolio_daily_returns(strategy_frame)

    assert daily.loc[0, "strategy_return"] == pytest.approx(0.0095)
    assert daily.loc[0, "exposure"] == pytest.approx(0.5)
    assert daily.loc[0, "turnover"] == pytest.approx(0.5)


def test_choose_threshold_respects_min_exposure() -> None:
    predictions = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]),
            "index_name": ["DJI", "DJI", "DJI"],
            "next_return": [0.01, 0.01, -0.01],
            "p_up": [0.51, 0.52, 0.53],
        }
    )

    threshold, metrics = choose_threshold(
        predictions,
        thresholds=[0.50, 0.90],
        cost_per_trade=0.0,
        min_exposure=0.50,
    )

    assert threshold == 0.50
    assert metrics == strategy_metrics(
        make_long_cash_strategy_frame(predictions, 0.50, 0.0)
    )
