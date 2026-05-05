from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import MAX_REASONABLE_UNITS


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute regression metrics for Walmart unit sales forecasting."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred, dtype=float)
    finite_pred = np.nan_to_num(
        y_pred_array,
        nan=0.0,
        posinf=MAX_REASONABLE_UNITS,
        neginf=0.0,
    )
    clipped_pred = np.clip(finite_pred, 0, MAX_REASONABLE_UNITS)
    positive_mask = y_true_array > 0
    zero_mask = y_true_array == 0

    metrics = {
        "mae": float(mean_absolute_error(y_true_array, clipped_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, clipped_pred))),
        "rmsle": float(
            np.sqrt(mean_squared_error(np.log1p(y_true_array), np.log1p(clipped_pred)))
        ),
        "r2": float(r2_score(y_true_array, clipped_pred)),
        "mean_actual_units": float(np.mean(y_true_array)),
        "mean_predicted_units": float(np.mean(clipped_pred)),
    }

    if positive_mask.any():
        metrics["positive_sales_mae"] = float(
            mean_absolute_error(y_true_array[positive_mask], clipped_pred[positive_mask])
        )
    else:
        metrics["positive_sales_mae"] = 0.0

    if zero_mask.any():
        metrics["zero_sales_mae"] = float(
            mean_absolute_error(y_true_array[zero_mask], clipped_pred[zero_mask])
        )
    else:
        metrics["zero_sales_mae"] = 0.0

    return metrics
