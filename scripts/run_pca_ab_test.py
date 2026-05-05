from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import MAX_REASONABLE_UNITS, NUMERIC_COLUMNS, RANDOM_STATE, RESULTS_DIR  # noqa: E402
from data import load_dataset_split  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from modeling import select_training_sample  # noqa: E402


def build_numeric_ridge(use_pca: bool) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if use_pca:
        steps.append(("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)))
    steps.append(("ridge", Ridge(alpha=3.0)))
    return Pipeline(steps)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_dataset_split()
    X_train_sample, y_train_sample = select_training_sample(
        X_train,
        y_train,
        zero_to_positive_ratio=3,
        random_state=RANDOM_STATE,
    )

    test_index = X_test.sample(n=min(150_000, len(X_test)), random_state=RANDOM_STATE).index
    X_test_sample = X_test.loc[test_index, NUMERIC_COLUMNS]
    y_test_sample = y_test.loc[test_index]

    rows = []
    for use_pca in [False, True]:
        model_id = "numeric_ridge_with_pca" if use_pca else "numeric_ridge_no_pca"
        model = build_numeric_ridge(use_pca=use_pca)
        model.fit(X_train_sample[NUMERIC_COLUMNS], np.log1p(y_train_sample))
        predictions = np.expm1(
            np.clip(model.predict(X_test_sample), 0, np.log1p(MAX_REASONABLE_UNITS))
        )
        metrics = compute_metrics(y_test_sample, predictions)

        row = {
            "experiment": model_id,
            "feature_set": "numeric_engineered_features",
            "train_rows": len(X_train_sample),
            "test_rows": len(X_test_sample),
            **metrics,
        }
        if use_pca:
            pca = model.named_steps["pca"]
            row["pca_components"] = int(pca.n_components_)
            row["pca_explained_variance"] = float(pca.explained_variance_ratio_.sum())
        else:
            row["pca_components"] = 0
            row["pca_explained_variance"] = 0.0
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("rmsle")
    results.to_csv(RESULTS_DIR / "pca_ab_test.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
