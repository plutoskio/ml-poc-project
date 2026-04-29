import pytest

from src.metrics import compute_classification_metrics, compute_metrics


def test_compute_metrics_returns_template_metric_set() -> None:
    metrics = compute_metrics([0, 1, 1, 0], [0, 1, 0, 0])

    assert set(metrics) == {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
    }
    assert metrics["accuracy"] == pytest.approx(0.75)


def test_compute_classification_metrics_can_include_roc_auc() -> None:
    metrics = compute_classification_metrics(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 0, 0],
        y_proba=[0.1, 0.9, 0.4, 0.2],
    )

    assert "roc_auc" in metrics
    assert metrics["roc_auc"] == pytest.approx(1.0)
