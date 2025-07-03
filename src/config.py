from pathlib import Path

from dotenv import load_dotenv

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

# Data
RAW_DATA_FILE = "customer_churn_dataset-testing-master.csv"
CLEANED_DATA_FILE = "cleaned_data.csv"
KAGGLE_DATASET_NAME = "muhammadshahidazeem/customer-churn-dataset"

# W&B
WANDB_MODEL_NAME = "churn-model"

# Columns
TARGET_COLUMN = "churn"
CUSTOMER_ID_COLUMN_RAW = "customerid"
CUSTOMER_ID_COLUMN_PROCESSED = "CustomerID"

# Data Splitting
TEST_SIZE = 0.2
RANDOM_STATE = 42
