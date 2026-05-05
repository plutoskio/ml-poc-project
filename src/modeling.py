from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_COLUMNS,
    CATEGORY_LEVELS,
    MAX_REASONABLE_UNITS,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
)


class LagBlendRegressor:
    """Simple transparent baseline using recent item-store sales history."""

    def __init__(self, lag_7_weight: float = 0.45, rolling_28_weight: float = 0.35):
        self.lag_7_weight = lag_7_weight
        self.rolling_28_weight = rolling_28_weight
        self.global_mean_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LagBlendRegressor":
        self.global_mean_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.global_mean_ is None:
            raise RuntimeError("LagBlendRegressor must be fitted before prediction.")

        lag_7 = X["lag_7"].to_numpy(dtype=float)
        rolling_28 = X["rolling_mean_28"].to_numpy(dtype=float)
        rolling_90 = X["rolling_mean_90"].to_numpy(dtype=float)

        remainder_weight = 1.0 - self.lag_7_weight - self.rolling_28_weight
        prediction = (
            self.lag_7_weight * lag_7
            + self.rolling_28_weight * rolling_28
            + remainder_weight * rolling_90
        )
        cold_start_mask = (lag_7 == 0) & (rolling_28 == 0) & (rolling_90 == 0)
        prediction[cold_start_mask] = self.global_mean_
        return np.maximum(prediction, 0)


def cast_hist_gradient_features(X: pd.DataFrame) -> pd.DataFrame:
    """Preserve categorical dtypes so HistGradientBoosting treats them correctly."""
    output = X.copy()
    for column in CATEGORICAL_COLUMNS:
        output[column] = output[column].astype(
            pd.CategoricalDtype(categories=CATEGORY_LEVELS[column])
        )
    for column in NUMERIC_COLUMNS:
        output[column] = output[column].astype(np.float32)
    return output


def bounded_expm1(log_predictions: np.ndarray) -> np.ndarray:
    """Convert log predictions back to units without allowing impossible overflow."""
    clipped = np.clip(log_predictions, 0, np.log1p(MAX_REASONABLE_UNITS))
    return np.expm1(clipped)


def make_ridge_log_regression() -> TransformedTargetRegressor:
    """Build a regularized linear model for sparse, high-cardinality tabular data."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    categories=[CATEGORY_LEVELS[column] for column in CATEGORICAL_COLUMNS],
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
                CATEGORICAL_COLUMNS,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                NUMERIC_COLUMNS,
            ),
        ],
        sparse_threshold=0.3,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=3.0, random_state=RANDOM_STATE)),
        ]
    )

    return TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=bounded_expm1,
        check_inverse=False,
    )


def make_poisson_regression() -> Pipeline:
    """Build a nonnegative count model with weighted zero-sampling support."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    categories=[CATEGORY_LEVELS[column] for column in CATEGORICAL_COLUMNS],
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
                CATEGORICAL_COLUMNS,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                NUMERIC_COLUMNS,
            ),
        ],
        sparse_threshold=0.3,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                PoissonRegressor(
                    alpha=10.0,
                    solver="lbfgs",
                    max_iter=1000,
                    tol=1e-3,
                ),
            ),
        ]
    )


def make_hist_gradient_boosting() -> TransformedTargetRegressor:
    """Build a compact non-linear model for engineered sales and weather features."""
    model = Pipeline(
        steps=[
            ("cast_dtypes", FunctionTransformer(cast_hist_gradient_features, validate=False)),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.08,
                    max_iter=180,
                    max_leaf_nodes=31,
                    min_samples_leaf=40,
                    l2_regularization=0.15,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=bounded_expm1,
        check_inverse=False,
    )


def select_training_sample(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    zero_to_positive_ratio: int = 4,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Keep all positive sales rows and a controlled sample of zero-sales rows."""
    positive_index = y_train[y_train > 0].index
    zero_index = y_train[y_train == 0].index

    max_zero_rows = min(len(zero_index), max(len(positive_index) * zero_to_positive_ratio, 1))
    sampled_zero_index = (
        pd.Series(zero_index)
        .sample(n=max_zero_rows, random_state=random_state, replace=False)
        .to_numpy()
    )

    sampled_index = np.concatenate([positive_index.to_numpy(), sampled_zero_index])
    sampled_index.sort()

    return X_train.loc[sampled_index].copy(), y_train.loc[sampled_index].copy()


def make_zero_sampling_weights(y_full: pd.Series, y_sample: pd.Series) -> np.ndarray:
    """Weight sampled rows so zero/positive prevalence matches the full training set."""
    full_zero_count = int((y_full == 0).sum())
    full_positive_count = int((y_full > 0).sum())
    sample_zero_count = max(int((y_sample == 0).sum()), 1)
    sample_positive_count = max(int((y_sample > 0).sum()), 1)

    zero_weight = full_zero_count / sample_zero_count
    positive_weight = full_positive_count / sample_positive_count

    return np.where(y_sample.to_numpy() == 0, zero_weight, positive_weight)
