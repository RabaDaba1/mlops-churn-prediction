# MLOps Project: Real-time Customer Churn Prediction

This project demonstrates a full MLOps pipeline for predicting customer churn in real-time.

## Overview

The goal is to build a reproducible and automated machine learning system using modern technologies. The project covers the entire lifecycle from data ingestion to model deployment and monitoring.

### Technology Stack

-   **Data & Pipeline Versioning:** DVC (+ Git)
-   **Experiment Tracking & Model Registry:** Weights & Biases (W&B)
-   **Feature Store:** Feast
-   **CI/CD:** GitHub Actions
-   **Data Streaming:** Apache Kafka
-   **Workflow Orchestration:** Apache Airflow
-   **Containerization & Orchestration:** Docker & Kubernetes
-   **Model Serving:** KServe on Kubernetes
-   **Monitoring:** Prometheus & Grafana


## How to install

1.  **Create a virtual environment and install dependencies:**
    ```bash
    uv python install 3.11
    uv sync --dev
    ```

2.  **Set up pre-commit hooks (optional but recommended):**
    This will ensure code quality and run DVC hooks automatically.
    ```bash
    pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type post-checkout
    ```

3.  **Set up Kaggle API credentials:**
    To download the dataset, you need a Kaggle account and an API token.
    -   Go to your Kaggle account settings, and click on "Create New API Token".
    -   This will download a `kaggle.json` file.
    -   Place this file in `~/.kaggle/kaggle.json`.

4.  **Set up Weights & Biases (W&B):**
    -   Log in to your W&B account:
        ```bash
        wandb login
        ```
    -   Set the following environment variables for your project. You can add them to a `.env` file.
        ```bash
        export WANDB_PROJECT="customer-churn-prediction"
        export WANDB_ENTITY="your-wandb-username"
        ```

## How to run

To run the full data pipeline, use DVC:

```bash
dvc repro
```

This command will execute all stages defined in `dvc.yaml`, including data processing, feature engineering, and model training. Each run will create a new versioned model artifact in W&B (e.g., `v0`, `v1`).

## How to run the prediction service

The API is configured to use a specific model version to ensure stability. This version is defined in `src/config.py`.

1.  **Update Model Version (Optional):**
    After training and validating a new model, update the `WANDB_MODEL_VERSION` in `src/config.py` to point to the new version you want to deploy.
    ```python
    # src/config.py
    WANDB_MODEL_VERSION = "v1" # Or your desired version
    ```
    Commit this change to version control your API and model pairing.

2.  **Set W&B environment variables:**
    The API service needs to connect to W&B to download the model.
    ```bash
    export WANDB_PROJECT="customer-churn-prediction"
    export WANDB_ENTITY="your-wandb-username"
    ```

3.  **Start the FastAPI server:**
    ```bash
    uvicorn src.api.main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

4.  **Send a prediction request:**
    You can use the example client to send a request to the running server.
    ```bash
    dvc repro predict
    ```
    This will run the `src/api/client.py` script, which sends a sample request and prints the prediction.
