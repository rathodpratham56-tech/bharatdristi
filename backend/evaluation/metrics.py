from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(labels: list[int], predictions: list[int], probabilities: np.ndarray | None = None, class_names: list[str] | None = None) -> dict[str, Any]:
    labels_array = np.asarray(labels)
    predictions_array = np.asarray(predictions)
    matrix = confusion_matrix(labels_array, predictions_array)
    per_class = {}
    for index in range(matrix.shape[0]):
        true_positive = matrix[index, index]
        false_negative = matrix[index, :].sum() - true_positive
        false_positive = matrix[:, index].sum() - true_positive
        true_negative = matrix.sum() - true_positive - false_negative - false_positive
        per_class[str(class_names[index] if class_names and index < len(class_names) else index)] = {
            "precision": float(true_positive / max(true_positive + false_positive, 1)),
            "recall_sensitivity": float(true_positive / max(true_positive + false_negative, 1)),
            "specificity": float(true_negative / max(true_negative + false_positive, 1)),
            "support": int(matrix[index, :].sum()),
        }
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels_array, predictions_array)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_array, predictions_array)),
        "macro_precision": float(precision_score(labels_array, predictions_array, average="macro", zero_division=0)),
        "macro_recall_sensitivity": float(recall_score(labels_array, predictions_array, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels_array, predictions_array, average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "roc_auc": "not available" if probabilities is None else "computed only when all classes are represented",
    }
    if probabilities is not None:
        try:
            result["roc_auc"] = float(roc_auc_score(labels_array, probabilities, multi_class="ovr", average="macro"))
        except ValueError:
            pass
    return result
