from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from config import MODELS, RANDOM_STATE, RESULTS_DIR


FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.csv"


def _build_diagnostic_sample(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    max_positive_rows: int,
    max_zero_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a stable test sample that includes enough positive-sales rows."""
    positive_index = y_test[y_test > 0].index
    zero_index = y_test[y_test == 0].index

    rng = np.random.default_rng(random_state)
    sampled_positive = rng.choice(
        positive_index.to_numpy(),
        size=min(len(positive_index), max_positive_rows),
        replace=False,
    )
    sampled_zero = rng.choice(
        zero_index.to_numpy(),
        size=min(len(zero_index), max_zero_rows),
        replace=False,
    )
    sampled_index = np.concatenate([sampled_positive, sampled_zero])
    sampled_index.sort()

    return X_test.loc[sampled_index].copy(), y_test.loc[sampled_index].copy()


def save_feature_importance(
    metrics: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path = FEATURE_IMPORTANCE_PATH,
    max_positive_rows: int = 8_000,
    max_zero_rows: int = 8_000,
    n_repeats: int = 3,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Save permutation importance for the selected model on a diagnostic test sample."""
    best_row = metrics.sort_values("rmsle").iloc[0]
    best_model_id = str(best_row["model_id"])
    best_model_name = str(best_row["model_name"])
    model = joblib.load(MODELS[best_model_id]["path"])

    X_sample, y_sample = _build_diagnostic_sample(
        X_test=X_test,
        y_test=y_test,
        max_positive_rows=max_positive_rows,
        max_zero_rows=max_zero_rows,
        random_state=random_state,
    )

    importance = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )

    result = (
        pd.DataFrame(
            {
                "feature": X_sample.columns,
                "importance_mae": importance.importances_mean,
                "importance_mae_std": importance.importances_std,
                "model_id": best_model_id,
                "model_name": best_model_name,
                "sample_rows": len(X_sample),
                "positive_sales_share": float((y_sample > 0).mean()),
            }
        )
        .sort_values("importance_mae", ascending=False)
        .reset_index(drop=True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
