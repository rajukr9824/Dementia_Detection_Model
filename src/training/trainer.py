from __future__ import annotations

from pathlib import Path

import pandas as pd
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Model

from src.config import model_config
from src.dataset.loader import create_dataloaders
from src.models.efficientnet import build_efficientnet_model
from src.models.efficientnet_cbam import build_efficientnet_cbam
from src.training.callbacks import get_callbacks
from src.training.compile import compile_model


def save_training_history(
    history: History,
    history_path: str,
) -> None:
    """Save training history as a CSV file.

    Args:
        history: History object returned by model.fit().
        history_path: Path where the history CSV will be saved.
    """

    history_file = Path(history_path)

    history_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_df = pd.DataFrame(history.history)

    history_df.to_csv(
        history_file,
        index=False,
    )

    print(
        f"Training history saved to {history_file}"
    )


def train(
    model: Model,
    model_save_path: str,
    csv_log_path: str,
    tensorboard_log_dir: str,
    history_path: str,
) -> History:
    """Generic training pipeline.

    Args:
        model: TensorFlow model.
        model_save_path: Path to save the best model.
        csv_log_path: Path for CSV training logs.
        tensorboard_log_dir: TensorBoard log directory.
        history_path: Path to save training history.

    Returns:
        TensorFlow History object.
    """

    train_dataset, validation_dataset, _ = (
        create_dataloaders()
    )

    model = compile_model(model)

    callbacks = get_callbacks(
        model_save_path=model_save_path,
        csv_log_path=csv_log_path,
        tensorboard_log_dir=tensorboard_log_dir,
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=model_config.EPOCHS,
        callbacks=callbacks,
    )

    save_training_history(
        history=history,
        history_path=history_path,
    )

    return history


def train_efficientnet() -> History:
    """Train the EfficientNetV2 baseline."""

    model = build_efficientnet_model()

    return train(
        model=model,
        model_save_path="saved_models/efficientnet.keras",
        csv_log_path="outputs/efficientnet_log.csv",
        tensorboard_log_dir="logs/efficientnet",
        history_path="outputs/efficientnet_history.csv",
    )


def train_cbam() -> History:
    """Train the EfficientNetV2 + CBAM model."""

    model = build_efficientnet_cbam()

    return train(
        model=model,
        model_save_path="saved_models/efficientnet_cbam.keras",
        csv_log_path="outputs/efficientnet_cbam_log.csv",
        tensorboard_log_dir="logs/efficientnet_cbam",
        history_path="outputs/efficientnet_cbam_history.csv",
    )


def main() -> None:
    """Run the training pipeline."""

    # ---------------------------------------------
    # Select which model to train
    # ---------------------------------------------

    history = train_cbam()

    # history = train_efficientnet()

    print("\nTraining Complete")

    print(
        f"Best Validation Accuracy: "
        f"{max(history.history['val_accuracy']):.4f}"
    )


if __name__ == "__main__":
    main()