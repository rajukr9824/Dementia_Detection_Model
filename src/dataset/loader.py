from pathlib import Path
from typing import Dict

import tensorflow as tf
import pandas as pd
import numpy as np

from src.preprocessing.preprocessing import (
    IMAGE_SIZE,
    read_image,
    resize_image,
    apply_clahe,
    normalize_image,
)

from src.augmentation.augmentation import augment_image


DATASET_PATH = Path("data") / "processed"

TRAIN_FILE = "train.csv"
VALIDATION_FILE = "validation.csv"
TEST_FILE = "test.csv"

# Dataset split names
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"

BATCH_SIZE = 32
SHUFFLE_BUFFER = 1000

AUTOTUNE = tf.data.AUTOTUNE


LABEL_MAPPING = {
    "MildDemented": 0,
    "ModerateDemented": 1,
    "NonDemented": 2,
    "VeryMildDemented": 3,
}

INDEX_TO_LABEL = {
    value: key
    for key, value in LABEL_MAPPING.items()
}
SPLIT_FILES = {
    TRAIN_SPLIT: TRAIN_FILE,
    VALIDATION_SPLIT: VALIDATION_FILE,
    TEST_SPLIT: TEST_FILE,
}

def load_split_dataframe(
    split: str,
) -> pd.DataFrame:
    """
    Load a dataset split from disk.

    Parameters
    ----------
    split : str
        Dataset split to load.
        One of: "train", "validation", "test".

    Returns
    -------
    pd.DataFrame
        DataFrame containing image paths and labels.

    Raises
    ------
    ValueError
        If the split name is invalid.

    FileNotFoundError
        If the CSV file does not exist.
    """

    if split not in SPLIT_FILES:
        raise ValueError(
            f"Invalid split '{split}'. "
            f"Expected one of: {list(SPLIT_FILES.keys())}"
        )

    file_path = DATASET_PATH / SPLIT_FILES[split]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {file_path}"
        )

    return pd.read_csv(file_path)

def preprocess_sample(
    image_path: str,
    label: str,
    is_training: bool,
) -> tuple[np.ndarray, np.int32]:
    """
    Load and preprocess a single image sample.

    Parameters
    ----------
    image_path : str
        Path to the MRI image.

    label : str
        Class label.

    is_training : bool
        Whether the sample belongs to the training split.
        If True, data augmentation is applied.

    Returns
    -------
    tuple[np.ndarray, np.int32]
        Preprocessed image and encoded label.
    """

    # Read image
    image = read_image(Path(image_path))

    # Resize image
    image = resize_image(image)

    # Apply CLAHE
    image = apply_clahe(image)

    # Apply augmentation only during training
    if is_training:
        image = augment_image(image)

    # Normalize image
    image = normalize_image(image)

    # Encode label
    encoded_label = LABEL_MAPPING[label]

    return image, np.int32(encoded_label)

def tf_preprocess_sample(
    image_path,
    label,
    is_training,
):
    """
    TensorFlow wrapper around preprocess_sample().
    """

    image_path = image_path.numpy().decode("utf-8")
    label = label.numpy().decode("utf-8")

    image, label = preprocess_sample(
        image_path=image_path,
        label=label,
        is_training=is_training,
    )

    return image, label

def create_tf_dataset(
    df: pd.DataFrame,
    is_training: bool,
) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset from a DataFrame.
    """

    image_paths = df["image_path"].astype(str).values
    labels = df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    def map_function(image_path, label):
        image, encoded_label = tf.py_function(
            func=lambda p, l: tf_preprocess_sample(
                p,
                l,
                is_training,
            ),
            inp=[image_path, label],
            Tout=(tf.float32, tf.int32),
        )

        image.set_shape((*IMAGE_SIZE, 3))
        encoded_label.set_shape(())

        return image, encoded_label

    dataset = dataset.map(
        map_function,
        num_parallel_calls=AUTOTUNE,
    )

    if is_training:
        dataset = dataset.shuffle(
            SHUFFLE_BUFFER
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset

def create_dataloaders() -> tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    tf.data.Dataset,
]:
    """
    Create TensorFlow datasets for training,
    validation, and testing.

    Returns
    -------
    tuple
        Training, validation, and testing datasets.
    """

    train_df = load_split_dataframe(TRAIN_SPLIT)
    validation_df = load_split_dataframe(VALIDATION_SPLIT)
    test_df = load_split_dataframe(TEST_SPLIT)

    train_dataset = create_tf_dataset(
        train_df,
        is_training=True,
    )

    validation_dataset = create_tf_dataset(
        validation_df,
        is_training=False,
    )

    test_dataset = create_tf_dataset(
        test_df,
        is_training=False,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )

def main() -> None:
    """
    Test the TensorFlow data pipeline.
    """

    train_dataset, validation_dataset, test_dataset = (
        create_dataloaders()
    )

    images, labels = next(iter(train_dataset))

    print("=" * 50)
    print("TensorFlow Data Loader")
    print("=" * 50)

    print(f"Images Shape : {images.shape}")
    print(f"Labels Shape : {labels.shape}")

    print(f"Image dtype : {images.dtype}")
    print(f"Label dtype : {labels.dtype}")

    print(f"Minimum Pixel : {tf.reduce_min(images).numpy():.4f}")
    print(f"Maximum Pixel : {tf.reduce_max(images).numpy():.4f}")

if __name__ == "__main__":
    main()