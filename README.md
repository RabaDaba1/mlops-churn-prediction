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


## CI/CD Setup (GitHub Actions)

For the continuous training pipeline in `.github/workflows/main.yml` to run, you need to configure the following secrets and variables in your GitHub repository settings (`Settings > Secrets and variables > Actions`):

### Secrets

-   `B2_ACCESS_KEY_ID`: Your Backblaze B2 Application Key ID.
-   `B2_SECRET_ACCESS_KEY`: Your Backblaze B2 Application Key.
-   `KAGGLE_JSON`: The content of your `kaggle.json` API token file.
-   `WANDB_API_KEY`: Your Weights & Biases API key.
-   `WANDB_ENTITY`: Your W&B username or organization name.
-   `CICD_PAT`: A GitHub Personal Access Token with repository write permissions to push dvc.lock changes.

### Variables

-   `WANDB_PROJECT`: The name of your W&B project (e.g., `customer-churn-prediction`).


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
    -   Go to your Kaggle account settings page and click "Create New API Token".
    -   This will download a `kaggle.json` file containing your credentials.
    -   Place this file in `~/.kaggle/kaggle.json` on macOS/Linux or `C:\Users\<Windows-username>\.kaggle\kaggle.json` on Windows. The application will automatically use it for authentication.

4.  **Set up Weights & Biases (W&B):**
    -   Log in to your W&B account. This will store your credentials for the CLI.
        ```bash
        wandb login
        ```
    -   Set the following environment variables for your project. You can add them to a `.env` file for local development.
        ```bash
        WANDB_PROJECT="customer-churn-prediction"
        WANDB_ENTITY="your-wandb-username"
        ```

5.  **Set up DVC remote storage (optional):**
    This project is configured to use a Backblaze B2 bucket as a DVC remote. To push/pull data, you need to configure your credentials.
    -   First, add the remote:
        ```bash
        dvc remote add -d b2remote s3://customer-churn-prediction/dvc
        dvc remote modify b2remote endpointurl https://s3.eu-central-003.backblazeb2.com
        ```
    -   Then, configure your credentials locally. These will not be committed to Git.
        ```bash
        dvc remote modify --local b2remote access_key_id YOUR_KEY_ID
        dvc remote modify --local b2remote secret_access_key YOUR_SECRET
        ```

## How to run

To run the full data pipeline, use DVC:

```bash
uv run dvc repro
```

This command will execute all stages defined in `dvc.yaml`, including data processing, feature engineering, and model training. The final `model_evaluation` step compares the newly trained model's performance (ROC AUC) against the current `latest` model in the W&B registry. If the new model performs better or equally, it is promoted by being logged to W&B with the `latest` alias, creating a new version (e.g., `v0`, `v1`).

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
    The API service needs to connect to W&B to download the model and preprocessor.
    ```bash
    export WANDB_PROJECT="customer-churn-prediction"
    export WANDB_ENTITY="your-wandb-username"
    ```

3.  **Start the FastAPI server:**
    ```bash
    uvicorn src.api.main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

## How to run with Docker

You can run the prediction service inside a Docker container using Docker Compose. This is the recommended method for local development and testing.

1.  **Create an environment file:**
    Create a new `.env` file in the project root. Fill in your W&B credentials:
    ```bash
    # .env
    WANDB_PROJECT="customer-churn-prediction"
    WANDB_ENTITY="your-wandb-entity"
    WANDB_API_KEY="your-wandb-api-key"
    ```

2.  **Build and run the container:**
    From the root of the project, run:
    ```bash
    docker-compose up --build
    ```
    To run in detached mode, add the `-d` flag: `docker-compose up --build -d`.

    The API will be accessible at `http://localhost:8000`. To stop the service, run `docker-compose down`.
