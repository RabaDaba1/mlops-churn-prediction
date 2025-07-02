from pathlib import Path

import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter

from src.config import RAW_DATA_DIR
from src.logs import get_logger

logger = get_logger("data_collection")


def download_dataset(output_path: Path):
    dataset_name = "muhammadshahidazeem/customer-churn-dataset"
    file_path = "customer_churn_dataset-testing-master.csv"
    logger.info(f"Downloading dataset: {dataset_name}")

    try:
        df: pd.DataFrame = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS, dataset_name, file_path
        )

        logger.info(f"Dataset downloaded successfully. Saving to {output_path}")

        df.to_csv(output_path / file_path, index=False)

        logger.info("Dataset download and copy complete.")
    except Exception as e:
        logger.error(f"Failed to download or copy dataset. Error: {e}")
        logger.error(
            "Please ensure your Kaggle API credentials are set up correctly. "
            "Place your kaggle.json file in ~/.kaggle/kaggle.json"
        )
        raise


if __name__ == "__main__":
    download_dataset(RAW_DATA_DIR)
