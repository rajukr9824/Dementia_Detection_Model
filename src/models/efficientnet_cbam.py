from __future__ import annotations

from typing import Optional

from tensorflow import keras
from tensorflow.keras import layers

from src.models.cbam import CBAM
from src.training.compile import compile_model


def freeze_base_model(
    base_model: keras.Model,
) -> None:
    """Freeze all layers of the backbone model.

    Args:
        base_model: Pretrained backbone model.
    """
    base_model.trainable = False


def unfreeze_top_layers(
    base_model: keras.Model,
    num_layers: int = 20,
) -> None:
    """Unfreeze the last N layers of the backbone model.

    Args:
        base_model: Pretrained backbone model.
        num_layers: Number of layers to unfreeze from the end.
    """
    base_model.trainable = True

    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False


def build_efficientnet_cbam(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    dropout_rate: float = 0.30,
    weights: Optional[str] = "imagenet",
    reduction_ratio: int = 16,
    kernel_size: int = 7,
) -> keras.Model:
    """Build EfficientNetV2B0 with a CBAM attention block.

    Architecture:
        Input
            ↓
        EfficientNetV2B0
            ↓
        CBAM
            ↓
        GlobalAveragePooling2D
            ↓
        Dropout
            ↓
        Dense(num_classes)
            ↓
        Softmax

    Args:
        input_shape: Input image shape.
        num_classes: Number of output classes.
        dropout_rate: Dropout probability.
        weights: Pretrained weights.
        reduction_ratio: CBAM channel reduction ratio.
        kernel_size: Spatial attention kernel size.

    Returns:
        Compiled Keras model.
    """

    # ------------------------------------------------------------------
    # Input Layer
    # ------------------------------------------------------------------
    inputs = keras.Input(
        shape=input_shape,
        name="input_image",
    )

    # ------------------------------------------------------------------
    # EfficientNetV2 Backbone
    # ------------------------------------------------------------------
    base_model = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=weights,
        input_tensor=inputs,
        pooling=None,
    )

    # Freeze pretrained backbone
    freeze_base_model(base_model)

    # ------------------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------------------
    x = base_model(inputs)

    # ------------------------------------------------------------------
    # CBAM Block
    # ------------------------------------------------------------------
    x = CBAM(
        reduction_ratio=reduction_ratio,
        kernel_size=kernel_size,
        name="cbam",
    )(x)

    # ------------------------------------------------------------------
    # Classification Head
    # ------------------------------------------------------------------
    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling",
    )(x)

    x = layers.Dropout(
        rate=dropout_rate,
        name="dropout",
    )(x)

    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        name="classifier",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="EfficientNetV2B0_CBAM",
    )

    return model

def main() -> None:
    """Build and compile the EfficientNetV2 + CBAM model."""

    model = build_efficientnet_cbam()

    model = compile_model(model)

    print("=" * 70)
    print("EfficientNetV2 + CBAM")
    print("=" * 70)

    model.summary()

    print("\nModel compiled successfully!")

    print(f"Optimizer : {model.optimizer.__class__.__name__}")
    print(f"Loss      : {model.loss.__class__.__name__}")
    print(f"Metrics   : {model.metrics_names}")


if __name__ == "__main__":
    main()