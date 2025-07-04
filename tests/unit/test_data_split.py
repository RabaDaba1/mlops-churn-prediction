from pathlib import Path

import pandas as pd

from src.pipeline.data_split import split_data


def test_split_data(tmp_path: Path):
    processed_data_dir = tmp_path / "processed"
    processed_data_dir.mkdir()
    split_data_dir = tmp_path / "split"
    split_data_dir.mkdir()

    processed_file = processed_data_dir / "cleaned.csv"
    df = pd.DataFrame({"feature": range(100), "churn": [0] * 50 + [1] * 50})
    df.to_csv(processed_file, index=False)

    split_data(
        input_path=processed_data_dir,
        output_path=split_data_dir,
        target_column="churn",
        test_size=0.2,
        random_state=42,
    )

    train_path = split_data_dir / "train.csv"
    test_path = split_data_dir / "test.csv"

    assert train_path.exists()
    assert test_path.exists()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    assert len(train_df) == 80
    assert len(test_df) == 20
    assert "churn" in train_df.columns
    assert "churn" in test_df.columns

    assert train_df["churn"].value_counts()[0] == 40
    assert train_df["churn"].value_counts()[1] == 40
    assert test_df["churn"].value_counts()[0] == 10
    assert test_df["churn"].value_counts()[1] == 10
