from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import MODELS_DIR, RESULTS_DIR
from data import feature_columns, load_modeling_dataset, save_processed_datasets


MODEL_OUTPUTS = {
    "log_reg": MODELS_DIR / "log_reg.joblib",
    "random_forest": MODELS_DIR / "random_forest.joblib",
    "hist_gradient_boosting": MODELS_DIR / "hist_gradient_boosting.joblib",
}


def _classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def _predict_positive_probability(model: Pipeline, X: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(X)[:, 1], index=X.index)

    if hasattr(model, "decision_function"):
        scores = pd.Series(model.decision_function(X), index=X.index)
        return (scores - scores.min()) / (scores.max() - scores.min())

    raise TypeError(f"Model {model} does not expose probabilities or scores.")


def _baseline_rows(
    dataset: pd.DataFrame,
    split_name: str,
) -> list[dict[str, float | str]]:
    split_df = dataset.loc[dataset["split"] == split_name].copy()
    y_true = split_df["target"]

    majority_pred = pd.Series(1, index=split_df.index)
    rows = [
        {
            "model_key": "baseline_majority_up",
            "split": split_name,
            **_classification_metrics(y_true, majority_pred),
        }
    ]

    previous_day_pred = (split_df["return_1d"] > 0).astype(int)
    rows.append(
        {
            "model_key": "baseline_previous_day_direction",
            "split": split_name,
            **_classification_metrics(y_true, previous_day_pred),
        }
    )
    return rows


def _models() -> dict[str, Pipeline]:
    return {
        "log_reg": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=20,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=1,
                        random_state=42,
                    ),
                )
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.04,
                        max_iter=250,
                        max_leaf_nodes=15,
                        l2_regularization=0.2,
                        random_state=42,
                    ),
                )
            ]
        ),
    }


def main() -> None:
    save_processed_datasets()
    dataset = load_modeling_dataset()
    columns = feature_columns(dataset)

    train_df = dataset.loc[dataset["split"] == "train"]
    validation_df = dataset.loc[dataset["split"] == "validation"]
    test_df = dataset.loc[dataset["split"] == "test"]
    fit_df = dataset.loc[dataset["split"].isin(["train", "validation"])]

    X_train = train_df[columns]
    y_train = train_df["target"]
    X_validation = validation_df[columns]
    y_validation = validation_df["target"]
    X_test = test_df[columns]
    y_test = test_df["target"]
    X_fit = fit_df[columns]
    y_fit = fit_df["target"]

    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    rows: list[dict[str, float | str]] = []
    rows.extend(_baseline_rows(dataset, "validation"))
    rows.extend(_baseline_rows(dataset, "test"))

    trained_models = _models()
    for model_key, model in trained_models.items():
        model.fit(X_train, y_train)

        for split_name, X_split, y_split in [
            ("validation", X_validation, y_validation),
            ("test", X_test, y_test),
        ]:
            y_pred = pd.Series(model.predict(X_split), index=X_split.index)
            y_proba = _predict_positive_probability(model, X_split)
            rows.append(
                {
                    "model_key": model_key,
                    "split": split_name,
                    **_classification_metrics(y_split, y_pred, y_proba),
                }
            )

        # Save a final model fit on train + validation. The test set is never fit.
        final_model = _models()[model_key]
        final_model.fit(X_fit, y_fit)
        joblib.dump(final_model, MODEL_OUTPUTS[model_key])

    report = pd.DataFrame(rows)
    report_path = RESULTS_DIR / "training_classification_report.csv"
    report.to_csv(report_path, index=False)

    print("Training complete.")
    print(f"Saved models to: {MODELS_DIR}")
    print(f"Saved report to: {report_path}")
    print("\nValidation/Test classification report:")
    print(report.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
