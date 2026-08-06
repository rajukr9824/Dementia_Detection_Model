from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.models import Model


def compile_model(model: Model) -> Model:
    """
    Compile a TensorFlow model for multi-class dementia classification.

    Args:
        model:
            TensorFlow Keras model.

    Returns:
        Compiled TensorFlow Keras model.
    """

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
    )

    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy",
        ),
    ]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    return model