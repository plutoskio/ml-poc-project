from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MODELS, MODELS_DIR, PROCESSED_DATASET_PATH, RESULTS_DIR, RANDOM_STATE  # noqa: E402
from data import load_dataset_split  # noqa: E402
from features import save_modeling_dataset  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from modeling import (  # noqa: E402
    LagBlendRegressor,
    make_zero_sampling_weights,
    make_hist_gradient_boosting,
    make_poisson_regression,
    select_training_sample,
)
from results import save_best_model_predictions  # noqa: E402


def evaluate_and_save_predictions(
    model_id: str,
    model_name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float | str]:
    predictions = np.maximum(model.predict(X_test), 0)
    metrics = compute_metrics(y_test, predictions)
    return {"model_id": model_id, "model_name": model_name, **metrics}


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not PROCESSED_DATASET_PATH.exists():
        save_modeling_dataset(PROCESSED_DATASET_PATH)

    X_train, X_test, y_train, y_test = load_dataset_split()
    X_train_sample, y_train_sample = select_training_sample(
        X_train,
        y_train,
        zero_to_positive_ratio=8,
    )
    sampling_weights = make_zero_sampling_weights(y_train, y_train_sample)

    model_builders = {
        "lag_blend_baseline": (LagBlendRegressor(), X_train, y_train, {}),
        "poisson_regression": (
            make_poisson_regression(),
            X_train_sample,
            y_train_sample,
            {"regressor__sample_weight": sampling_weights},
        ),
        "hist_gradient_boosting": (make_hist_gradient_boosting(), X_train_sample, y_train_sample, {}),
    }

    rows = []
    for model_id, (model, X_fit, y_fit, fit_params) in model_builders.items():
        print(f"Training {model_id} on {len(X_fit):,} rows...")
        model.fit(X_fit, y_fit, **fit_params)
        joblib.dump(model, MODELS[model_id]["path"])
        rows.append(
            evaluate_and_save_predictions(
                model_id=model_id,
                model_name=MODELS[model_id]["name"],
                model=model,
                X_test=X_test,
                y_test=y_test,
            )
        )

    metrics = pd.DataFrame(rows).sort_values("rmsle")
    metrics.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

    save_best_model_predictions(
        metrics=metrics,
        X_test=X_test,
        y_test=y_test,
        random_state=RANDOM_STATE,
    )

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
