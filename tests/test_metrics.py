import numpy as np

from metrics import compute_metrics


def test_compute_metrics_returns_numeric_values():
    metrics = compute_metrics([0, 2, 5], [0.1, 1.5, 6])

    assert set(metrics) >= {"mae", "rmse", "rmsle", "r2", "positive_sales_mae"}
    assert all(np.isfinite(value) for value in metrics.values())

