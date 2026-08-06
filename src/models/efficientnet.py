from __future__ import annotations

import tensorflow as tf

from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.models import Model

from src.config.model_config import (
    DROPOUT_RATE,
    INPUT_SHAPE,
    MODEL_NAME,
    NUM_CLASSES,
    PRETRAINED_WEIGHTS,
)

def freeze_base_model(base_model: Model) -> Model:
    """
    Freeze all layers of the pretrained backbone.

    Args:
        base_model:
            Pretrained EfficientNetV2 backbone.

    Returns:
        Frozen backbone model.
    """
    base_model.trainable = False
    return base_model

def build_efficientnet_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = DROPOUT_RATE,
) -> Model:
    """
    Build the EfficientNetV2B0 baseline model for multi-class dementia
    classification using transfer learning.

    The architecture consists of:
        Input
            ↓
        EfficientNetV2B0 (ImageNet pretrained, frozen)
            ↓
        GlobalAveragePooling2D
            ↓
        Dropout
            ↓
        Dense (Softmax)

    Args:
        input_shape:
            Shape of the input image as (height, width, channels).

        num_classes:
            Number of output classes.

        dropout_rate:
            Dropout probability applied before the output layer.

    Returns:
        A TensorFlow Keras model ready for compilation.
    """

    # ------------------------------------------------------------------
    # Input Layer
    # ------------------------------------------------------------------
    inputs = Input(
        shape=input_shape,
        name="input_image",
    )

    # ------------------------------------------------------------------
    # Load EfficientNetV2B0 Backbone
    # ------------------------------------------------------------------
    base_model = EfficientNetV2B0(
        include_top=False,
        weights=PRETRAINED_WEIGHTS,
        input_shape=input_shape,
    )

    # ------------------------------------------------------------------
    # Freeze pretrained backbone
    # ------------------------------------------------------------------
    base_model = freeze_base_model(base_model)

    # ------------------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------------------
    features = base_model(inputs, training=False)

    # ------------------------------------------------------------------
    # Classification Head
    # ------------------------------------------------------------------
    x = GlobalAveragePooling2D(
        name="global_average_pooling",
    )(features)

    x = Dropout(
        rate=dropout_rate,
        name="dropout",
    )(x)

    outputs = Dense(
        units=num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    # ------------------------------------------------------------------
    # Create Model
    # ------------------------------------------------------------------
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name=MODEL_NAME,
    )

    return model

def unfreeze_top_layers(
    base_model: Model,
    fine_tune_at: int,
) -> Model:
    """
    Unfreeze the top layers of the pretrained backbone for fine-tuning.

    Layers before ``fine_tune_at`` remain frozen, while layers from
    ``fine_tune_at`` onward become trainable.

    Args:
        base_model:
            Pretrained EfficientNetV2 backbone.

        fine_tune_at:
            Index of the first layer to unfreeze.

    Returns:
        Updated backbone model with selected layers trainable.
    """

    # Freeze lower layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Unfreeze upper layers
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    return base_model

def print_model_summary(model: Model) -> None:
    """
    Print the summary of a Keras model.

    Args:
        model:
            TensorFlow Keras model.

    Returns:
        None
    """
    model.summary()

def main() -> None:
    """
    Build the EfficientNetV2 baseline model and display its summary.
    """

    model = build_efficientnet_model()

    print_model_summary(model)

if __name__ == "__main__":
    main()