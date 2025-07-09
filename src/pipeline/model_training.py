import os
from pathlib import Path

import dvc.api
import git
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

import wandb
from src.config import (
    FEATURES_DIR,
    MODEL_DIR,
    MODEL_FILENAME,
    TARGET_COLUMN,
    config,
)
from src.logs import get_logger

logger = get_logger("model_training")


def train_model(
    input_path: Path = FEATURES_DIR,
    target_column: str = TARGET_COLUMN,
):
    logger.info("Starting model training...")

    train_df = pd.read_csv(input_path / "train.csv")
    test_df = pd.read_csv(input_path / "test.csv")

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    repo = git.Repo(search_parent_directories=True)
    git_commit = repo.head.object.hexsha
    data_url = dvc.api.get_url(path=str(input_path), repo=repo.working_tree_dir)

    wandb_config = {
        "git_commit": git_commit,
        "data_url": data_url,
        "target_column": target_column,
        "test_size": config.data_split.test_size,
        "random_state": config.random_state,
        **config.hyperparameters.model_dump(),
    }

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        config=wandb_config,
        job_type="training",
    )

    model = XGBClassifier(
        **config.hyperparameters.model_dump(), random_state=config.random_state
    )
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

    model_path = MODEL_DIR / MODEL_FILENAME
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")

    run.finish()

    logger.info("Model training complete.")


if __name__ == "__main__":
    train_model()
