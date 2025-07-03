import os
from pathlib import Path

import pandas as pd
import wandb
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.config import FEATURES_DIR, PARAMS_FILE, TARGET_COLUMN
from src.logs import get_logger

logger = get_logger("model_training")


def train_model(
    input_path: Path = FEATURES_DIR,
    params_file: Path = PARAMS_FILE,
    target_column: str = TARGET_COLUMN,
):
    logger.info("Starting model training...")

    train_df = pd.read_csv(input_path / "train_featured.csv")
    test_df = pd.read_csv(input_path / "test_featured.csv")

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    with open(params_file) as f:
        params = yaml.safe_load(f)

    xgb_params = params["xgboost"]

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        config=xgb_params,
        job_type="training",
    )

    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    logger.info(f"Model metrics: {metrics}")
    wandb.log(metrics)

    model_artifact = wandb.Artifact(
        "churn-model",
        type="model",
        description="XGBoost churn prediction model",
        metadata=xgb_params,
    )
    model_path = "model.json"
    model.save_model(model_path)
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)

    run.finish()
    logger.info("Model training complete and artifact logged to W&B.")


if __name__ == "__main__":
    train_model()
