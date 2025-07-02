from pathlib import Path

import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter

from src.config import KAGGLE_DATASET_NAME, RAW_DATA_DIR, RAW_DATA_FILE
from src.logs import get_logger

logger = get_logger("data_collection")


def download_dataset(
    output_path: Path = RAW_DATA_DIR,
    dataset_name: str = KAGGLE_DATASET_NAME,
    file_path: str = RAW_DATA_FILE,
):
    logger.info(f"Downloading dataset: {dataset_name}")

    try:
        df: pd.DataFrame = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS, dataset_name, file_path
        )
    except Exception as e:
        logger.error(f"Failed to download dataset. Error: {e}")
        raise

    logger.info(f"Dataset downloaded successfully. Saving to {output_path}")

    (output_path / file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / file_path, index=False)

    logger.info("Dataset download and copy complete.")


if __name__ == "__main__":
    download_dataset()
