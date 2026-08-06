from pathlib import Path

import cv2
import pandas as pd
import numpy as np

from src.preprocessing.visualization import visualize_preprocessing

# Supported image formats
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Dataset location
DATASET_PATH = Path("data/raw")


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Load all image paths and labels from the dataset.

    Args:
        dataset_path (Path): Path to the dataset directory.

    Returns:
        pd.DataFrame: DataFrame containing image paths and labels.
    """

    image_paths = []
    labels = []

    for class_dir in sorted(dataset_path.iterdir()):

        if not class_dir.is_dir():
            continue

        label = class_dir.name

        for image_path in sorted(class_dir.iterdir()):

            if image_path.suffix.lower() in VALID_EXTENSIONS:
                image_paths.append(image_path)
                labels.append(label)

    return pd.DataFrame({
        "image_path": image_paths,
        "label": labels
    })

def read_image(image_path: Path) -> np.ndarray:
    """
    Read an image from disk and convert it to RGB.

    Args:
        image_path (Path): Path to the image.

    Returns:
        np.ndarray: RGB image.

    Raises:
        ValueError: If the image cannot be read.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

IMAGE_SIZE = (224, 224)

def resize_image(image: np.ndarray,
                 image_size: tuple = IMAGE_SIZE) -> np.ndarray:
    """
    Resize an image to the specified dimensions.

    Args:
        image (np.ndarray): Input image.
        image_size (tuple): Target size (width, height).

    Returns:
        np.ndarray: Resized image.
    """

    resized_image = cv2.resize(
        image,
        image_size,
        interpolation=cv2.INTER_AREA
    )

    return resized_image

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to an RGB image.

    Args:
        image (np.ndarray): RGB image.

    Returns:
        np.ndarray: RGB image after CLAHE.
    """

    # Convert RGB → LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Split into channels
    l, a, b = cv2.split(lab)

    # Create CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    # Apply CLAHE to Lightness channel only
    l = clahe.apply(l)

    # Merge channels
    lab = cv2.merge((l, a, b))

    # Convert back to RGB
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return image

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Convert image to float32 without scaling.

    EfficientNetV2 performs input rescaling internally.
    """
    return image.astype(np.float32)


if __name__ == "__main__":

    # Load dataset
    df = load_dataset(DATASET_PATH)

    # Select first image
    sample_path = df.iloc[0]["image_path"]

    # Read image
    image = read_image(sample_path)

    # Resize image
    resized = resize_image(image)

    # Apply CLAHE
    clahe_image = apply_clahe(resized)

    # Normalize image
    normalized = normalize_image(clahe_image)

    # Print results
    print("Original Shape :", image.shape)
    print("Resized Shape :", resized.shape)
    print("Normalized dtype :", normalized.dtype)
    print("Minimum Pixel :", normalized.min())
    print("Maximum Pixel :", normalized.max())
    visualize_preprocessing(
    original=image,
    resized=resized,
    clahe_image=clahe_image
)