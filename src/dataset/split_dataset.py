from pathlib import Path
from typing import Tuple
import math

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.preprocessing import load_dataset

DATASET_PATH: Path = Path("data") / "raw"

TRAIN_SIZE: float = 0.80
VALIDATION_SIZE: float = 0.10
TEST_SIZE: float = 0.10

RANDOM_STATE: int = 42

OUTPUT_DIR: Path = Path("data") / "processed"

TRAIN_FILE: str = "train.csv"
VALIDATION_FILE: str = "validation.csv"
TEST_FILE: str = "test.csv"

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate the dataset before splitting.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing image paths and labels.

    Raises
    ------
    ValueError
        If the dataset is empty, required columns are missing,
        missing values are found, or the split ratios are invalid.
    """

    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    required_columns = {"image_path", "label"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df[["image_path", "label"]].isnull().any().any():
        raise ValueError(
            "The dataset contains missing values in required columns."
        )

    total_ratio = TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE

    if not math.isclose(total_ratio, 1.0):
        raise ValueError(
            "TRAIN_SIZE, VALIDATION_SIZE and TEST_SIZE must sum to 1.0."
        )

def split_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, validation, and test sets
    using stratified sampling.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing image paths and labels.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, and test DataFrames.
    """

    validate_dataset(df)

    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        stratify=df["label"],
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return train_df, validation_df, test_df

def save_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Save the dataset splits as CSV files.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.

    validation_df : pd.DataFrame
        Validation dataset.

    test_df : pd.DataFrame
        Test dataset.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        OUTPUT_DIR / TRAIN_FILE,
        index=False,
    )

    validation_df.to_csv(
        OUTPUT_DIR / VALIDATION_FILE,
        index=False,
    )

    test_df.to_csv(
        OUTPUT_DIR / TEST_FILE,
        index=False,
    )

def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the saved dataset splits.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Training, validation, and test DataFrames.

    Raises
    ------
    FileNotFoundError
        If one or more split files are missing.
    """

    required_files = [
        OUTPUT_DIR / TRAIN_FILE,
        OUTPUT_DIR / VALIDATION_FILE,
        OUTPUT_DIR / TEST_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {file_path}"
            )

    train_df = pd.read_csv(OUTPUT_DIR / TRAIN_FILE)
    validation_df = pd.read_csv(OUTPUT_DIR / VALIDATION_FILE)
    test_df = pd.read_csv(OUTPUT_DIR / TEST_FILE)

    return train_df, validation_df, test_df

def print_split_summary(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Print a summary of the dataset splits.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.

    validation_df : pd.DataFrame
        Validation dataset.

    test_df : pd.DataFrame
        Test dataset.
    """

    total_images = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    print("=" * 50)
    print("Dataset Split Summary")
    print("=" * 50)

    print(f"Total Images      : {total_images}")
    print(f"Training Images   : {len(train_df)}")
    print(f"Validation Images : {len(validation_df)}")
    print(f"Testing Images    : {len(test_df)}")

    print("\nTraining Class Distribution")
    print(train_df["label"].value_counts())

    print("\nValidation Class Distribution")
    print(validation_df["label"].value_counts())

    print("\nTesting Class Distribution")
    print(test_df["label"].value_counts())

def main() -> None:
    """Execute the complete dataset splitting pipeline."""

    # Load dataset
    df = load_dataset(DATASET_PATH)

    # Split dataset
    train_df, validation_df, test_df = split_dataset(df)

    # Save splits
    save_splits(
        train_df,
        validation_df,
        test_df,
    )

    # Load saved splits
    train_df, validation_df, test_df = load_splits()

    # Display summary
    print_split_summary(
        train_df,
        validation_df,
        test_df,
    )

if __name__ == "__main__":
    main()