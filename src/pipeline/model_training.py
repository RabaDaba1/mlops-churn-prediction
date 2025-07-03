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
    PARAMS_FILE,
    TARGET_COLUMN,
    VERSION_FILE,
    WANDB_MODEL_NAME,
    config,
)
from src.logs import get_logger

logger = get_logger("model_training")


def get_and_increment_version(version_file: Path) -> str:
    with open(version_file, "r+") as f:
        version_str = f.read().strip()
        if not version_str.startswith("v"):
            raise ValueError("Version file is corrupted")
        current_version_num = int(version_str[1:])
        current_version = f"v{current_version_num}"
        next_version_num = current_version_num + 1
        f.seek(0)
        f.write(f"v{next_version_num}")
        f.truncate()
    return current_version


def train_model(
    input_path: Path = FEATURES_DIR,
    params_file: Path = PARAMS_FILE,
    target_column: str = TARGET_COLUMN,
    version_file: Path = VERSION_FILE,
):
    logger.info("Starting model training...")

    train_df = pd.read_csv(input_path / "train_featured.csv")
    test_df = pd.read_csv(input_path / "test_featured.csv")

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

    model_path = MODEL_DIR / "model.json"
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")

    model_version = get_and_increment_version(version_file)
    logger.info(f"Using model version: {model_version}")

    model_artifact = wandb.Artifact(
        WANDB_MODEL_NAME,
        type="model",
        description="XGBoost churn prediction model",
        metadata={
            "git_commit": git_commit,
            "data_url": data_url,
            "version": model_version,
        },
    )
    model_artifact.add_file(str(model_path))
    model_artifact.add_file(str(MODEL_DIR / "preprocessor.joblib"))
    wandb.log_artifact(
        model_artifact,
        name=WANDB_MODEL_NAME,
        aliases=["latest", model_version],
    )

    run.finish()
    logger.info("Model training complete and artifact logged to W&B.")


if __name__ == "__main__":
    train_model()
