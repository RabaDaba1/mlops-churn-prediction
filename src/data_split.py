from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CUSTOMER_ID_COLUMN_PROCESSED,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    SPLIT_DATA_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.logs import get_logger

logger = get_logger("data_split")


def split_data(
    input_path: Path = PROCESSED_DATA_DIR,
    output_path: Path = SPLIT_DATA_DIR,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    logger.info("Starting data splitting...")

    try:
        csv_file = next(input_path.glob("*.csv"))
    except StopIteration:
        logger.error(f"No CSV file found in {input_path}")
        raise

    df = pd.read_csv(csv_file)

    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in the dataset.")
        raise ValueError(f"Target column '{target_column}' not found.")

    if CUSTOMER_ID_COLUMN_PROCESSED in df.columns:
        df = df.drop(columns=[CUSTOMER_ID_COLUMN_PROCESSED])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_output_file = output_path / "train.csv"
    test_output_file = output_path / "test.csv"

    train_df.to_csv(train_output_file, index=False)
    test_df.to_csv(test_output_file, index=False)

    logger.info(f"Data splitting complete. Train set saved to {train_output_file}")
    logger.info(f"Data splitting complete. Test set saved to {test_output_file}")


if __name__ == "__main__":
    split_data()
