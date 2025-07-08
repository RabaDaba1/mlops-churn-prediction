import os
from pathlib import Path

import dvc.api
import git
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import wandb
from src.config import (
    FEATURES_DIR,
    MODEL_DIR,
    TARGET_COLUMN,
    WANDB_MODEL_NAME,
)
from src.logs import get_logger

logger = get_logger("model_evaluation")


def evaluate_model(
    model_path: Path = MODEL_DIR / "model.json",
    test_data_path: Path = FEATURES_DIR / "test_featured.csv",
    target_column: str = TARGET_COLUMN,
):
    logger.info("Starting model evaluation...")

    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    new_model = XGBClassifier()
    new_model.load_model(model_path)
    roc_auc = roc_auc_score(y_test, new_model.predict_proba(X_test)[:, 1])
    logger.info(f"New model ROC AUC: {roc_auc}")

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        job_type="evaluation",
    )

    run.log({"roc_auc": roc_auc})

    repo = git.Repo(search_parent_directories=True)
    git_commit = repo.head.object.hexsha
    data_url = dvc.api.get_url(path=str(FEATURES_DIR), repo=repo.working_tree_dir)

    model_artifact = wandb.Artifact(
        WANDB_MODEL_NAME,
        type="model",
        description="XGBoost churn prediction model",
        metadata={
            "git_commit": git_commit,
            "data_url": data_url,
            "roc_auc": roc_auc,
        },
    )
    model_artifact.add_file(str(model_path))
    model_artifact.add_file(str(MODEL_DIR / "preprocessor.joblib"))

    branch_name = repo.active_branch.name
    alias = "production" if branch_name == "main" else f"dev-{branch_name}"

    run.log_artifact(model_artifact, aliases=[alias])
    logger.info("New model artifact logged to W&B and marked as 'latest'.")

    run.finish()

    logger.info("Model evaluation complete.")


if __name__ == "__main__":
    evaluate_model()
