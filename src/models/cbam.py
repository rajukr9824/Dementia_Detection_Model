from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Register custom layers for Keras serialization so saved models
# referencing these classes can be loaded without providing a
# `custom_objects` mapping.


@keras.utils.register_keras_serializable()
class ChannelAttention(layers.Layer):
    """Channel Attention module from the CBAM paper."""

    def __init__(
        self,
        reduction_ratio: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(
        self,
        input_shape: tf.TensorShape,
    ) -> None:
        """Build the Channel Attention module."""

        channels = int(input_shape[-1])
        self.channels = channels

        hidden_units = max(
            channels // self.reduction_ratio,
            1,
        )

        self.avg_pool = layers.GlobalAveragePooling2D(
            name="channel_gap",
        )

        self.max_pool = layers.GlobalMaxPooling2D(
            name="channel_gmp",
        )

        self.shared_mlp = keras.Sequential(
            [
                layers.Dense(
                    hidden_units,
                    activation="relu",
                    name="channel_fc1",
                ),
                layers.Dense(
                    channels,
                    name="channel_fc2",
                ),
            ],
            name="shared_mlp",
        )

        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
    ) -> tf.Tensor:
        """Forward pass."""

        avg_features = self.avg_pool(inputs)
        avg_features = self.shared_mlp(avg_features)

        max_features = self.max_pool(inputs)
        max_features = self.shared_mlp(max_features)

        attention = avg_features + max_features
        attention = tf.nn.sigmoid(attention)

        attention = tf.expand_dims(attention, axis=1)
        attention = tf.expand_dims(attention, axis=1)

        return inputs * attention

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "reduction_ratio": self.reduction_ratio,
            }
        )
        return config


@keras.utils.register_keras_serializable()
class SpatialAttention(layers.Layer):
    """Spatial Attention module from the CBAM paper."""

    def __init__(
        self,
        kernel_size: int = 7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(
        self,
        input_shape: tf.TensorShape,
    ) -> None:
        """Build Spatial Attention."""

        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
            activation="sigmoid",
            use_bias=False,
            name="spatial_conv",
        )

        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
    ) -> tf.Tensor:
        """Forward pass."""

        avg_pool = tf.reduce_mean(
            inputs,
            axis=-1,
            keepdims=True,
        )

        max_pool = tf.reduce_max(
            inputs,
            axis=-1,
            keepdims=True,
        )

        attention = tf.concat(
            [avg_pool, max_pool],
            axis=-1,
        )

        attention = self.conv(attention)

        return inputs * attention

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()

        config.update(
            {
                "kernel_size": self.kernel_size,
            }
        )

        return config


@keras.utils.register_keras_serializable()
class CBAM(layers.Layer):
    """Convolutional Block Attention Module."""

    def __init__(
        self,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(
            reduction_ratio=reduction_ratio,
            name="channel_attention",
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=kernel_size,
            name="spatial_attention",
        )

    def call(
        self,
        inputs: tf.Tensor,
    ) -> tf.Tensor:
        """Forward pass."""

        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)

        return x

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()

        config.update(
            {
                "reduction_ratio": self.reduction_ratio,
                "kernel_size": self.kernel_size,
            }
        )

        return config


if __name__ == "__main__":

    print("=" * 60)
    print("Testing Channel Attention")
    print("=" * 60)

    dummy_input = tf.random.normal(
        (2, 7, 7, 1280)
    )

    channel = ChannelAttention()

    output = channel(dummy_input)

    print("Input Shape :", dummy_input.shape)
    print("Output Shape:", output.shape)

    print("\n")

    print("=" * 60)
    print("Testing Spatial Attention")
    print("=" * 60)

    spatial = SpatialAttention()

    output = spatial(dummy_input)

    print("Input Shape :", dummy_input.shape)
    print("Output Shape:", output.shape)

    print("\n")

    print("=" * 60)
    print("Testing CBAM")
    print("=" * 60)

    cbam = CBAM()

    output = cbam(dummy_input)

    print("Input Shape :", dummy_input.shape)
    print("Output Shape:", output.shape)

    print("\n")

    inputs = keras.Input(
        shape=(7, 7, 1280),
    )

    outputs = CBAM()(inputs)

    model = keras.Model(
        inputs,
        outputs,
        name="CBAM_Model",
    )

    model.summary()