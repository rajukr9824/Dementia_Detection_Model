"""
Creates TensorFlow callbacks used during model training.
"""

from __future__ import annotations

from pathlib import Path

from tensorflow.keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from src.config import model_config


def get_callbacks(
    model_save_path: str,
    csv_log_path: str,
    tensorboard_log_dir: str,
) -> list[Callback]:
    """Create TensorFlow callbacks.

    Args:
        model_save_path:
            Path where the best model will be saved.
        csv_log_path:
            Path for saving the CSV training log.
        tensorboard_log_dir:
            Directory for TensorBoard logs.

    Returns:
        List of TensorFlow callback objects.
    """

    # --------------------------------------------------
    # Create required directories
    # --------------------------------------------------

    Path(model_save_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(csv_log_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(tensorboard_log_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Early Stopping
    # --------------------------------------------------

    early_stopping = EarlyStopping(
        monitor=model_config.MONITOR_METRIC,
        patience=model_config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    # --------------------------------------------------
    # Save Best Model
    # --------------------------------------------------

    model_checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor=model_config.MONITOR_METRIC,
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    # --------------------------------------------------
    # TensorBoard
    # --------------------------------------------------

    tensorboard = TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
    )

    # --------------------------------------------------
    # CSV Logger
    # --------------------------------------------------

    csv_logger = CSVLogger(
        filename=csv_log_path,
        append=False,
    )

    # --------------------------------------------------
    # Reduce Learning Rate
    # --------------------------------------------------

    reduce_lr = ReduceLROnPlateau(
        monitor=model_config.MONITOR_METRIC,
        factor=model_config.LR_FACTOR,
        patience=model_config.LR_PATIENCE,
        verbose=1,
    )

    return [
        early_stopping,
        model_checkpoint,
        tensorboard,
        csv_logger,
        reduce_lr,
    ]


def main() -> None:
    """Test callback creation."""

    callbacks = get_callbacks(
        model_save_path="saved_models/test_model.keras",
        csv_log_path="outputs/test_training_log.csv",
        tensorboard_log_dir="logs/test",
    )

    print("Callbacks Created Successfully\n")

    for callback in callbacks:
        print(callback.__class__.__name__)


if __name__ == "__main__":
    main()