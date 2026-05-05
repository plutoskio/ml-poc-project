from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import MODELS, PROCESSED_DATASET_PATH, RESULTS_DIR, TEST_START_DATE


BEST_MODEL_PREDICTIONS_PATH = RESULTS_DIR / "best_model_test_predictions.csv"
PREDICTION_SAMPLE_PATH = RESULTS_DIR / "model_predictions_sample.csv"


def save_best_model_predictions(
    metrics: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_size: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Save dated predictions for the best model on the chronological test set."""
    best_row = metrics.sort_values("rmsle").iloc[0]
    best_model_id = str(best_row["model_id"])
    best_model_name = str(best_row["model_name"])
    best_model_path = Path(MODELS[best_model_id]["path"])
    best_model = joblib.load(best_model_path)

    predicted_units = np.maximum(best_model.predict(X_test), 0)

    modeling_df = pd.read_parquet(PROCESSED_DATASET_PATH)
    modeling_df["date"] = pd.to_datetime(modeling_df["date"])
    test_metadata = (
        modeling_df.loc[
            modeling_df["date"] >= pd.Timestamp(TEST_START_DATE),
            ["date", "store_nbr", "item_nbr"],
        ]
        .reset_index(drop=True)
        .copy()
    )

    if len(test_metadata) != len(X_test):
        raise ValueError(
            "Prediction metadata and X_test lengths do not match. "
            f"metadata={len(test_metadata)}, X_test={len(X_test)}"
        )

    predictions = test_metadata
    predictions["actual_units"] = y_test.reset_index(drop=True).to_numpy()
    predictions["predicted_units"] = predicted_units
    predictions["residual"] = predictions["actual_units"] - predictions["predicted_units"]
    predictions["absolute_error"] = predictions["residual"].abs()
    predictions["best_model_id"] = best_model_id
    predictions["best_model_name"] = best_model_name

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(BEST_MODEL_PREDICTIONS_PATH, index=False)

    sample = predictions.sample(
        n=min(sample_size, len(predictions)),
        random_state=random_state,
    )
    sample.to_csv(PREDICTION_SAMPLE_PATH, index=False)

    return predictions

