"""
Production evaluation pipeline.

Run:
    python -m src.evaluation.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config.model_config import (
    BATCH_SIZE,
    BASELINE_MODEL_SAVE_PATH,
)
from src.dataset.loader import (
    INDEX_TO_LABEL,
    create_dataloaders,
)
from src.models.cbam import (
    CBAM,
    ChannelAttention,
    SpatialAttention,
)
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_misclassified_images,
    plot_prediction_distribution,
)


CLASS_NAMES = [
    INDEX_TO_LABEL[i]
    for i in sorted(INDEX_TO_LABEL)
]


def load_trained_model(
    model_path: str | Path,
    custom_objects: dict[str, Any] | None = None,
) -> tf.keras.Model:
    """
    Load a trained Keras model.

    Args:
        model_path: Path to the saved model.
        custom_objects: Dictionary containing custom layers if required.

    Returns:
        Loaded Keras model.
    """
    print(f"\nLoading model: {model_path}")

    return tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
    )


def evaluate_model(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
) -> tuple[float, float]:
    """
    Evaluate the model on the test dataset.

    Args:
        model: Trained Keras model.
        test_dataset: Test tf.data.Dataset.

    Returns:
        Tuple containing test loss and accuracy.
    """
    loss, accuracy = model.evaluate(
        test_dataset,
        verbose=1,
    )

    return loss, accuracy


def generate_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Generate predictions.

    Args:
        model: Trained model.
        dataset: Test dataset.

    Returns:
        images,
        true labels,
        predicted labels,
        prediction probabilities.
    """

    images = []
    labels = []

    for batch_images, batch_labels in dataset:
        images.append(batch_images.numpy())
        labels.append(batch_labels.numpy())

    images = np.concatenate(images)
    y_true = np.concatenate(labels)

    probabilities = model.predict(
        dataset,
        verbose=1,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    return (
        images,
        y_true,
        y_pred,
        probabilities,
    )


def save_metrics(
    metrics: dict,
    test_loss: float,
    test_accuracy: float,
    output_dir: Path,
) -> None:
    """
    Save evaluation metrics.

    Args:
        metrics: Dictionary returned by compute_metrics().
        test_loss: Test loss.
        test_accuracy: Test accuracy.
        output_dir: Output directory.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = metrics["classification_report"]

    report["test_loss"] = float(test_loss)
    report["test_accuracy"] = float(test_accuracy)

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    pd.DataFrame(report).transpose().to_csv(
        output_dir / "classification_report.csv"
    )

    pd.DataFrame(
        metrics["confusion_matrix"],
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(
        output_dir / "confusion_matrix.csv"
    )


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Save predictions.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        probabilities: Prediction probabilities.
        output_dir: Output directory.
    """

    prediction_df = pd.DataFrame(
        {
            "true_label": y_true,
            "predicted_label": y_pred,
            "confidence": probabilities.max(axis=1),
        }
    )

    prediction_df.to_csv(
        output_dir / "predictions.csv",
        index=False,
    )


def run_evaluation(
    model_path: str | Path,
    model_name: str,
    custom_objects: dict[str, Any] | None = None,
) -> None:
    """
    Run complete evaluation pipeline.

    Args:
        model_path: Saved model path.
        model_name: Name of model.
        custom_objects: Optional custom layers.
    """

    output_dir = (
        Path("outputs")
        / "evaluation"
        / model_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_trained_model(
        model_path=model_path,
        custom_objects=custom_objects,
    )

    _, _, test_dataset = create_dataloaders()

    test_loss, test_accuracy = evaluate_model(
        model=model,
        test_dataset=test_dataset,
    )

    (
        images,
        y_true,
        y_pred,
        probabilities,
    ) = generate_predictions(
        model=model,
        dataset=test_dataset,
    )

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=CLASS_NAMES,
    )

    print("\n" + "=" * 60)
    print(f"{model_name.upper()} Evaluation Summary")
    print("=" * 60)

    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")

    save_metrics(
        metrics=metrics,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        output_dir=output_dir,
    )

    save_predictions(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        output_dir=output_dir,
    )

    plot_confusion_matrix(
        confusion_matrix=metrics["confusion_matrix"],
        class_names=CLASS_NAMES,
        save_path=output_dir / "confusion_matrix.png",
    )

    plot_prediction_distribution(
        y_pred=y_pred,
        class_names=CLASS_NAMES,
    )

    plot_misclassified_images(
        images=images,
        y_true=y_true,
        y_pred=y_pred,
        class_names=CLASS_NAMES,
        probabilities=probabilities,
    )

    print("\nEvaluation completed successfully.")
    print(f"Results saved to: {output_dir}")


def main() -> None:
    """
    Standalone execution.
    """

    run_evaluation(
        model_path=BASELINE_MODEL_SAVE_PATH,
        model_name="baseline",
        custom_objects={
            "CBAM": CBAM,
            "ChannelAttention": ChannelAttention,
            "SpatialAttention": SpatialAttention,
        },
    )


if __name__ == "__main__":
    main()