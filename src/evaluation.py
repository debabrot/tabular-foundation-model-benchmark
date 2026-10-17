"""Task-specific metrics and comparison tables from measured predictions."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score, root_mean_squared_error,
)

from .data import Task

METRIC_COLUMNS = {
    "binary": ["roc_auc", "average_precision", "f1", "precision", "recall", "accuracy"],
    "multiclass": ["macro_f1", "weighted_f1", "accuracy"],
    "regression": ["rmse", "mae", "r2"],
}
PRIMARY_METRICS = {task: columns[0] for task, columns in METRIC_COLUMNS.items()}


def evaluate_predictions(
    task: Task, y_true: np.ndarray, predictions: np.ndarray, *,
    probabilities: np.ndarray | None = None, classes: np.ndarray | None = None,
    positive_label=None,
) -> dict[str, float]:
    """Evaluate labels and probabilities, using the explicit probability-column order."""
    if task not in PRIMARY_METRICS:
        raise ValueError(f"Unknown task: {task}")
    y_true, predictions = np.asarray(y_true), np.asarray(predictions)
    if y_true.ndim != 1 or predictions.shape != y_true.shape or len(y_true) < 2:
        raise ValueError("Expected matching one-dimensional targets and predictions with at least two rows")
    if task == "regression":
        if not np.isfinite(y_true).all() or not np.isfinite(predictions).all():
            raise ValueError("Regression targets and predictions must be finite")
        return {"rmse": float(root_mean_squared_error(y_true, predictions)),
                "mae": float(mean_absolute_error(y_true, predictions)),
                "r2": float(r2_score(y_true, predictions))}
    classes = np.asarray(classes)
    if classes.ndim != 1 or len(np.unique(classes)) != len(classes):
        raise ValueError("Provide unique classes in probability-column order")
    if ((task == "binary" and len(classes) != 2)
            or (task == "multiclass" and len(classes) < 3)):
        raise ValueError("Class count does not match task")
    if not np.isin(y_true, classes).all() or not np.isin(predictions, classes).all():
        raise ValueError("Targets or predictions contain an unknown class")
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        if (probabilities.shape != (len(y_true), len(classes))
                or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)):
            raise ValueError("Invalid probabilities or probability-column count")
    if task == "binary":
        if probabilities is None or positive_label not in classes:
            raise ValueError("Binary metrics need probabilities and an explicit positive_label")
        truth = y_true == positive_label
        if len(np.unique(truth)) != 2:
            raise ValueError("ROC-AUC requires both classes in the evaluation partition")
        scores = probabilities[:, np.flatnonzero(classes == positive_label)[0]]
        predicted_positive = predictions == positive_label
        return {"roc_auc": float(roc_auc_score(truth, scores)),
                "average_precision": float(average_precision_score(truth, scores)),
                "f1": float(f1_score(truth, predicted_positive, zero_division=0)),
                "precision": float(precision_score(truth, predicted_positive, zero_division=0)),
                "recall": float(recall_score(truth, predicted_positive, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, predictions))}
    return {"macro_f1": float(f1_score(y_true, predictions, labels=classes, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, predictions, labels=classes, average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, predictions))}


def comparison_table(results: dict, partition: str = "test") -> pd.DataFrame:
    """Create one row per model from the saved, unrounded metric values."""
    if partition not in ("validation", "test"):
        raise ValueError("Choose validation or test metrics")
    table = pd.DataFrame([
        {"model": row["name"], **row["metrics"].get(partition, {})} for row in results["models"]
    ]).set_index("model")
    return table.reindex(columns=METRIC_COLUMNS[results["task"]])
