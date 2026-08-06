"""
Evaluation metric utilities.

This module provides reusable functions for computing classification
metrics for the dementia detection model.

The module is framework-independent and operates only on
ground-truth labels and predicted labels.

Run:
    python -m src.evaluation.metrics
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute the confusion matrix.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.

    Returns:
        Confusion matrix as a NumPy array.
    """
    return confusion_matrix(y_true, y_pred)


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """
    Compute the classification report.

    The report includes:

    - Precision
    - Recall
    - F1-score
    - Support
    - Macro Average
    - Weighted Average
    - Overall Accuracy

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Ordered list of class names.

    Returns:
        Classification report as a dictionary.
    """
    return classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Ordered list of class names.

    Returns:
        Dictionary containing:

        {
            "confusion_matrix": ndarray,
            "classification_report": dict,
            "accuracy": float
        }
    """
    cm = compute_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
    )

    report = compute_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    metrics = {
        "confusion_matrix": cm,
        "classification_report": report,
        "accuracy": report["accuracy"],
    }

    return metrics


def main() -> None:
    """
    Demonstrate metric computation using sample data.

    This function enables standalone execution of the module.
    """

    class_names = [
        "NonDemented",
        "VeryMildDemented",
        "MildDemented",
        "ModerateDemented",
    ]

    y_true = np.array(
        [
            0, 0, 0,
            1, 1, 1,
            2, 2, 2,
            3, 3, 3,
        ]
    )

    y_pred = np.array(
        [
            0, 1, 0,
            1, 1, 2,
            2, 2, 1,
            3, 2, 3,
        ]
    )

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    print("\nConfusion Matrix")
    print("-" * 50)
    print(metrics["confusion_matrix"])

    print("\nAccuracy")
    print("-" * 50)
    print(f"{metrics['accuracy']:.4f}")

    print("\nClassification Report")
    print("-" * 50)

    report = metrics["classification_report"]

    for class_name in class_names:
        class_metrics = report[class_name]

        print(f"\n{class_name}")
        print(
            f"Precision : {class_metrics['precision']:.4f}"
        )
        print(
            f"Recall    : {class_metrics['recall']:.4f}"
        )
        print(
            f"F1-Score  : {class_metrics['f1-score']:.4f}"
        )
        print(
            f"Support   : {class_metrics['support']}"
        )

    print("\nMacro Average")
    print("-" * 50)
    print(report["macro avg"])

    print("\nWeighted Average")
    print("-" * 50)
    print(report["weighted avg"])


if __name__ == "__main__":
    main()