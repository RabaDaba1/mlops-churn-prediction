from pathlib import Path

import pandas as pd

from src.config import (
    CLEANED_DATA_FILE,
    CUSTOMER_ID_COLUMN_RAW,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RAW_DATA_FILE,
)
from src.logs import get_logger

logger = get_logger("data_cleaning")


def clean_data(
    raw_data_path: Path = RAW_DATA_DIR / RAW_DATA_FILE,
    processed_data_path: Path = PROCESSED_DATA_DIR / CLEANED_DATA_FILE,
    column_to_drop: str = CUSTOMER_ID_COLUMN_RAW,
):
    logger.info(f"Reading raw data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)

    logger.info("Cleaning data...")

    df.columns = [col.replace(" ", "_").lower() for col in df.columns]

    if column_to_drop in df.columns:
        df = df.drop(columns=[column_to_drop])
        logger.info(f"Dropped '{column_to_drop}' column.")

    logger.info(f"Saving cleaned data to {processed_data_path}")
    df.to_csv(processed_data_path, index=False)

    logger.info("Data cleaning complete.")


if __name__ == "__main__":
    clean_data()
