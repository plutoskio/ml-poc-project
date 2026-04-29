"""Train supervised classifiers and save model artifacts."""

from __future__ import annotations

import joblib
import pandas as pd

from src.config import MODELS_DIR, RESULTS_DIR
from src.data import feature_columns, load_modeling_dataset, save_processed_datasets
from src.metrics import compute_classification_metrics
from src.modeling import MODEL_OUTPUTS, build_models, predict_positive_probability


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
            **compute_classification_metrics(y_true, majority_pred),
        }
    ]

    previous_day_pred = (split_df["return_1d"] > 0).astype(int)
    rows.append(
        {
            "model_key": "baseline_previous_day_direction",
            "split": split_name,
            **compute_classification_metrics(y_true, previous_day_pred),
        }
    )
    return rows


def main() -> None:
    """Train supervised models and write classification reports."""

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

    trained_models = build_models()
    for model_key, model in trained_models.items():
        model.fit(X_train, y_train)

        for split_name, X_split, y_split in [
            ("validation", X_validation, y_validation),
            ("test", X_test, y_test),
        ]:
            y_pred = pd.Series(model.predict(X_split), index=X_split.index)
            y_proba = predict_positive_probability(model, X_split)
            rows.append(
                {
                    "model_key": model_key,
                    "split": split_name,
                    **compute_classification_metrics(y_split, y_pred, y_proba),
                }
            )

        # Save a final model fit on train + validation. The test set is never fit.
        final_model = build_models()[model_key]
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
