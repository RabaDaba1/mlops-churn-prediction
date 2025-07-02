import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.logs import get_logger

logger = get_logger("data_preparation")


def prepare_data():
    raw_data_path = RAW_DATA_DIR / "customer_churn_dataset-testing-master.csv"
    processed_data_path = PROCESSED_DATA_DIR / "prepared_data.csv"

    logger.info(f"Reading raw data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)

    logger.info("Preparing data...")

    df.columns = [col.replace(" ", "_").lower() for col in df.columns]

    if "customerid" in df.columns:
        df = df.drop(columns=["customerid"])
        logger.info("Dropped 'customerid' column.")

    logger.info(f"Saving prepared data to {processed_data_path}")
    df.to_csv(processed_data_path, index=False)

    logger.info("Data preparation complete.")


if __name__ == "__main__":
    prepare_data()
