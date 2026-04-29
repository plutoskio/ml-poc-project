"""Model factory and prediction helpers for supervised classifiers."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR

MODEL_OUTPUTS = {
    "log_reg": MODELS_DIR / "log_reg.joblib",
    "random_forest": MODELS_DIR / "random_forest.joblib",
    "hist_gradient_boosting": MODELS_DIR / "hist_gradient_boosting.joblib",
}


def build_models() -> dict[str, Pipeline]:
    """Return the supervised model pipelines used by the project."""

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


def predict_positive_probability(model: object, features: pd.DataFrame) -> pd.Series:
    """Return positive-class probabilities or normalized decision scores."""

    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(features)[:, 1], index=features.index)

    if hasattr(model, "decision_function"):
        scores = pd.Series(model.decision_function(features), index=features.index)
        denominator = scores.max() - scores.min()
        if denominator == 0:
            return pd.Series(0.5, index=features.index)
        return (scores - scores.min()) / denominator

    raise TypeError(f"Model {model} does not expose probabilities or scores.")
