import albumentations as A
import numpy as np

train_transform = A.Compose([
    A.Rotate(
        limit=10,
        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.1,
        contrast_limit=0.1,
        p=0.5
    ),
    A.GaussNoise(
        std_range=(0.02, 0.05),
        p=0.3
    ),
    A.Affine(
        translate_percent=(-0.05, 0.05),
        scale=(0.95, 1.05),
        p=0.5
    )
])

def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Apply augmentation pipeline to an image.

    Args:
        image (np.ndarray): RGB image.

    Returns:
        np.ndarray: Augmented image.
    """

    transformed = train_transform(image=image)

    return transformed["image"]

from src.preprocessing.preprocessing import (
    DATASET_PATH,
    load_dataset,
    read_image
)

from src.augmentation.visualization import (
    visualize_augmentation
)

if __name__ == "__main__":

    # Load dataset
    df = load_dataset(DATASET_PATH)

    # Select first image
    sample_path = df.iloc[0]["image_path"]

    # Read image
    image = read_image(sample_path)

    # Apply augmentation
    augmented = augment_image(image)

    # Visualize
    visualize_augmentation(
        original=image,
        augmented=augmented
    )