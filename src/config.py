from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
SPLIT_DATA_DIR = DATA_DIR / "split"
MODEL_DIR = PROJECT_ROOT / "models"

LOG_DIR = PROJECT_ROOT / "logs"

# Data
RAW_DATA_FILE = "customer_churn_dataset-testing-master.csv"
CLEANED_DATA_FILE = "cleaned_data.csv"
KAGGLE_DATASET_NAME = "muhammadshahidazeem/customer-churn-dataset"

# Columns
TARGET_COLUMN = "churn"
CUSTOMER_ID_COLUMN_RAW = "customerid"
CUSTOMER_ID_COLUMN_PROCESSED = "CustomerID"

# Data Splitting
TEST_SIZE = 0.2
RANDOM_STATE = 42

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
