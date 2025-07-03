from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
SPLIT_DATA_DIR = DATA_DIR / "split"
MODEL_DIR = ARTIFACTS_DIR / "models"
LOG_DIR = PROJECT_ROOT / "logs"

for dir in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FEATURES_DIR,
    SPLIT_DATA_DIR,
    MODEL_DIR,
    LOG_DIR,
]:
    dir.mkdir(parents=True, exist_ok=True)

# DVC
PARAMS_FILE = PROJECT_ROOT / "params.yaml"
VERSION_FILE = PROJECT_ROOT / "version.txt"

# Data
RAW_DATA_FILE = "customer_churn_dataset-testing-master.csv"
CLEANED_DATA_FILE = "cleaned_data.csv"
KAGGLE_DATASET_NAME = "muhammadshahidazeem/customer-churn-dataset"

# W&B
WANDB_MODEL_NAME = "customer-churn-model"
WANDB_MODEL_VERSION = "v0"

# Columns
TARGET_COLUMN = "churn"
CUSTOMER_ID_COLUMN_RAW = "customerid"
CUSTOMER_ID_COLUMN_PROCESSED = "CustomerID"


class Hyperparameters(BaseModel):
    colsample_bytree: float
    eval_metric: str
    gamma: float
    learning_rate: float
    max_depth: int
    min_child_weight: int
    n_estimators: int
    objective: str
    subsample: float


class DataSplit(BaseModel):
    test_size: float


class Config(BaseModel):
    random_state: int
    hyperparameters: Hyperparameters
    data_split: DataSplit
    target_column: str


def load_config() -> Config:
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)

    return Config(**params)


config = load_config()
