from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_DIR, SPLIT_DATA_DIR, config
from src.logs import get_logger

logger = get_logger("data_split")


def split_data(
    input_path: Path = PROCESSED_DATA_DIR,
    output_path: Path = SPLIT_DATA_DIR,
    target_column: str = config.target_column,
    test_size: float = config.data_split.test_size,
    random_state: int = config.random_state,
):
    logger.info("Starting data splitting...")

    csv_file = next(input_path.glob("*.csv"))
    df = pd.read_csv(csv_file)

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
