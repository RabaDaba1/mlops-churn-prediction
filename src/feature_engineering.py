from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.config import FEATURES_DIR, SPLIT_DATA_DIR, TARGET_COLUMN
from src.logs import get_logger

logger = get_logger("feature_engineering")


def feature_engineering(
    input_path: Path = SPLIT_DATA_DIR,
    output_path: Path = FEATURES_DIR,
    target_column: str = TARGET_COLUMN,
):
    logger.info("Starting feature engineering...")

    train_df = pd.read_csv(input_path / "train.csv")
    test_df = pd.read_csv(input_path / "test.csv")

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
    encoder.fit(X_train[categorical_cols])

    X_train_encoded = pd.DataFrame(
        encoder.transform(X_train[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X_train.index,
    )
    X_train_final = X_train.drop(columns=categorical_cols).join(X_train_encoded)

    X_test_encoded = pd.DataFrame(
        encoder.transform(X_test[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X_test.index,
    )
    X_test_final = X_test.drop(columns=categorical_cols).join(X_test_encoded)

    train_featured = pd.concat([X_train_final, y_train], axis=1)
    test_featured = pd.concat([X_test_final, y_test], axis=1)

    train_output_file = output_path / "train_featured.csv"
    test_output_file = output_path / "test_featured.csv"
    train_featured.to_csv(train_output_file, index=False)
    test_featured.to_csv(test_output_file, index=False)

    logger.info(
        f"Feature engineering complete. Saved to {train_output_file} and {test_output_file}"
    )


if __name__ == "__main__":
    feature_engineering()
