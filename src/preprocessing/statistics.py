from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from src.preprocessing.preprocessing import (
    DATASET_PATH,
    load_dataset
)

def get_dataset_statistics(df: pd.DataFrame) -> None:
    """
    Display basic dataset statistics.
    """

    print("\n========== DATASET SUMMARY ==========\n")

    print(f"Total Images : {len(df)}")
    print(f"Number of Classes : {df['label'].nunique()}")

    print("\nImages per Class:\n")

    print(df["label"].value_counts())

def plot_class_distribution(df: pd.DataFrame):

    counts = df["label"].value_counts()

    plt.figure(figsize=(8,5))

    counts.plot(kind="bar")

    plt.title("Class Distribution")

    plt.xlabel("Class")

    plt.ylabel("Images")

    plt.tight_layout()

    plt.show()

def image_size_statistics(
    df: pd.DataFrame,
    sample_size=200
):

    sample_df = df.sample(
        n=min(sample_size, len(df)),
        random_state=42
    )

    heights = []
    widths = []

    for image_path in tqdm(
        sample_df["image_path"],
        desc="Reading Image Sizes"
    ):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        h, w = image.shape[:2]

        heights.append(h)
        widths.append(w)

    print("\n========== IMAGE SIZE ==========\n")

    print(f"Average Height : {sum(heights)/len(heights):.2f}")
    print(f"Average Width  : {sum(widths)/len(widths):.2f}")

    print(f"Minimum Height : {min(heights)}")
    print(f"Maximum Height : {max(heights)}")

    print(f"Minimum Width  : {min(widths)}")
    print(f"Maximum Width  : {max(widths)}")

def dataset_summary(df: pd.DataFrame):

    print("\n========== DATASET SUMMARY ==========\n")

    print(f"Total Images      : {len(df)}")
    print(f"Number of Classes : {df['label'].nunique()}")

    print("\nImages per Class:\n")

    print(df["label"].value_counts())

def check_corrupted_images(df: pd.DataFrame):

    corrupted = []

    for image_path in tqdm(
        df["image_path"],
        desc="Checking Images"
    ):

        image = cv2.imread(str(image_path))

        if image is None:
            corrupted.append(image_path)

    print("\nCorrupted Images :", len(corrupted))

    return corrupted

if __name__ == "__main__":

    df = load_dataset(DATASET_PATH)

    dataset_summary(df)

    corrupted = check_corrupted_images(df)

    image_size_statistics(df)

    plot_class_distribution(df)