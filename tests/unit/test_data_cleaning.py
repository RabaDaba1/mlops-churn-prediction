from pathlib import Path

import pandas as pd

from src.pipeline.data_cleaning import clean_data


def test_clean_data(tmp_path: Path):
    raw_data_dir = tmp_path / "raw"
    raw_data_dir.mkdir()
    processed_data_dir = tmp_path / "processed"
    processed_data_dir.mkdir()

    raw_data_path = raw_data_dir / "test_raw.csv"
    processed_data_path = processed_data_dir / "test_cleaned.csv"

    raw_df = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3],
            "Some Column": ["A", "B", "C"],
            "Another Value": [10.1, 20.2, 30.3],
        }
    )
    raw_df.to_csv(raw_data_path, index=False)

    clean_data(
        raw_data_path=raw_data_path,
        processed_data_path=processed_data_path,
        column_to_drop="customerid",
    )

    assert processed_data_path.exists()
    cleaned_df = pd.read_csv(processed_data_path)

    expected_columns = ["some_column", "another_value"]
    assert all(col in cleaned_df.columns for col in expected_columns)

    assert "customerid" not in cleaned_df.columns
    assert "CustomerID" not in cleaned_df.columns

    assert len(cleaned_df) == 3
    assert cleaned_df["some_column"].tolist() == ["A", "B", "C"]
