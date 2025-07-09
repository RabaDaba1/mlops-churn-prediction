from pathlib import Path

import joblib
import pandas as pd

from src.pipeline.feature_engineering import feature_engineering


def test_feature_engineering(tmp_path: Path):
    split_data_dir = tmp_path / "split"
    split_data_dir.mkdir()
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    train_df = pd.DataFrame(
        {
            "num_feat": [1, 2, 3],
            "cat_feat": ["A", "B", "A"],
            "churn": [0, 1, 0],
        }
    )
    test_df = pd.DataFrame(
        {
            "num_feat": [4, 5, 6],
            "cat_feat": ["B", "C", "A"],
            "churn": [1, 0, 1],
        }
    )
    train_df.to_csv(split_data_dir / "train.csv", index=False)
    test_df.to_csv(split_data_dir / "test.csv", index=False)

    from src.config import MODEL_DIR

    original_model_dir = MODEL_DIR
    import src.config

    src.config.MODEL_DIR = model_dir

    feature_engineering(
        input_path=split_data_dir,
        output_path=features_dir,
        target_column="churn",
        model_dir=model_dir,
    )

    train_featured_path = features_dir / "train.csv"
    test_featured_path = features_dir / "test.csv"
    preprocessor_path = model_dir / "preprocessor.joblib"

    assert train_featured_path.exists()
    assert test_featured_path.exists()
    assert preprocessor_path.exists()

    train_featured_df = pd.read_csv(train_featured_path)
    assert "cat_feat_B" in train_featured_df.columns
    assert "cat_feat" not in train_featured_df.columns
    assert len(train_featured_df.columns) == 3

    test_featured_df = pd.read_csv(test_featured_path)
    assert "cat_feat_C" not in test_featured_df.columns
    assert len(test_featured_df.columns) == 3

    preprocessor = joblib.load(preprocessor_path)
    assert "encoder" in preprocessor
    assert preprocessor["categorical_cols"] == ["cat_feat"]

    src.config.MODEL_DIR = original_model_dir
