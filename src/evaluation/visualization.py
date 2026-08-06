"""
Visualization utilities for model evaluation.

This module provides reusable plotting functions for
visualizing evaluation results.

Run:
    python -m src.evaluation.visualization
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.dataset.loader import INDEX_TO_LABEL


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a confusion matrix heatmap.

    Args:
        confusion_matrix:
            Confusion matrix.

        class_names:
            List of class names.

        save_path:
            Optional output image path.
    """
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_misclassified_images(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    probabilities: np.ndarray | None = None,
    max_images: int = 9,
) -> None:
    """
    Display misclassified images.

    Args:
        images:
            Image array.

        y_true:
            Ground-truth labels.

        y_pred:
            Predicted labels.

        class_names:
            List of class names.

        probabilities:
            Prediction probabilities.

        max_images:
            Maximum number of images to display.
    """
    wrong_indices = np.where(y_true != y_pred)[0]

    if len(wrong_indices) == 0:
        print("No misclassified images found.")
        return

    num_images = min(max_images, len(wrong_indices))

    plt.figure(figsize=(12, 12))

    for index in range(num_images):
        image_index = wrong_indices[index]

        plt.subplot(3, 3, index + 1)

        plt.imshow(images[image_index].astype(np.uint8))
        plt.axis("off")

        title = (
            f"True: {class_names[y_true[image_index]]}\n"
            f"Pred: {class_names[y_pred[image_index]]}"
        )

        if probabilities is not None:
            confidence = np.max(probabilities[image_index])
            title += f"\nConf: {confidence:.2f}"

        plt.title(title, fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_prediction_distribution(
    y_pred: np.ndarray,
    class_names: list[str],
) -> None:
    """
    Plot prediction distribution.

    Args:
        y_pred:
            Predicted labels.

        class_names:
            List of class names.
    """
    counts = np.bincount(
        y_pred,
        minlength=len(class_names),
    )

    plt.figure(figsize=(8, 5))

    plt.bar(class_names, counts)

    plt.title("Prediction Distribution")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")

    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Demonstrate visualization utilities.
    """

    class_names = [
        INDEX_TO_LABEL[i]
        for i in sorted(INDEX_TO_LABEL)
    ]

    confusion = np.array(
        [
            [45, 2, 0, 0],
            [2, 40, 3, 0],
            [0, 2, 35, 1],
            [0, 0, 2, 18],
        ]
    )

    plot_confusion_matrix(
        confusion_matrix=confusion,
        class_names=class_names,
    )

    y_pred = np.array(
        [
            0, 0, 0,
            1, 1, 2,
            2, 2, 2,
            3, 3, 3,
        ]
    )

    plot_prediction_distribution(
        y_pred=y_pred,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()