# MLOps Project: Real-time Customer Churn Prediction

This project demonstrates a full MLOps pipeline for predicting customer churn in real-time.

## Overview

The goal is to build a reproducible and automated machine learning system using modern technologies. The project covers the entire lifecycle from data ingestion to model deployment and monitoring.

### Technology Stack

-   **Data & Pipeline Versioning:** DVC (+ Git)
-   **Feature Store:** Feast
-   **Data Streaming:** Apache Kafka
-   **Workflow Orchestration:** Apache Airflow
-   **Containerization & Orchestration:** Docker & Kubernetes
-   **CI/CD:** GitHub Actions
-   **Experiment Tracking & Model Registry:** Weights & Biases (W&B)
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

This command will execute all stages defined in `dvc.yaml`, including data processing, feature engineering, and model training.
