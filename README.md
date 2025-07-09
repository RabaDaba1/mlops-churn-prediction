# MLOps Project: Real-time Customer Churn Prediction

A full MLOps pipeline for real-time customer churn prediction using modern tools and best practices.

---

## Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the model training](#running-the-model-training)
6. [Running the API](#running-the-api)
7. [Docker Usage](#docker-usage)
8. [Kubernetes Deployment with KServe](#kubernetes-deployment-with-kserve)
9. [Kubeflow Pipeline](#kubeflow-pipeline)
10. [CI/CD Pipeline](#cicd-pipeline)

---

## Overview

This project demonstrates a reproducible and automated ML system for customer churn prediction, covering the full lifecycle from data ingestion to deployment and monitoring.

---

## Tech Stack

- **Data & Pipeline Versioning:** DVC (+ Git)
- **Experiment Tracking & Model Registry:** Weights & Biases (W&B)
- **Feature Store:** Feast
- **CI/CD:** GitHub Actions
- **Data Streaming:** Apache Kafka
- **Workflow Orchestration:** Apache Airflow
- **Containerization:** Docker & Kubernetes
- **Model Serving:** KServe
- **Monitoring:** Prometheus & Grafana

---

## Installation

1. **Create a virtual environment and install dependencies:**
    ```bash
    uv python install 3.11
    uv sync --dev
    ```

2. **(Optional) Set up pre-commit hooks:**
    ```bash
    uv run pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type post-checkout
    ```

---

## Configuration

### 1. Kaggle API

- Download your `kaggle.json` from Kaggle account settings.
- Place it in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<User>\.kaggle\kaggle.json` (Windows).

### 2. Weights & Biases (W&B)

- Log in:
    ```bash
    uv run wandb login
    ```
- Set environment variables (add to `.env` for local development):
    ```bash
    WANDB_PROJECT="customer-churn-prediction"
    WANDB_ENTITY="your-wandb-username"
    ```

### 3. DVC Remote (Backblaze B2)

- Add and configure remote:
    ```bash
    uv run dvc remote add -d b2remote s3://customer-churn-prediction/dvc
    uv run dvc remote modify b2remote endpointurl https://s3.eu-central-003.backblazeb2.com
    uv run dvc remote modify --local b2remote access_key_id YOUR_KEY_ID
    uv run dvc remote modify --local b2remote secret_access_key YOUR_SECRET
    ```

---

## Running the model training pipeline

Run the full data pipeline with:
```bash
uv run dvc repro
```
This executes all stages in `dvc.yaml`, including data processing, feature engineering, model training, and evaluation. The evaluation step promotes the new model in W&B if it outperforms the previous one.

---

## Running the API

1. **(Optional) Update Model Version:**
    - Edit `WANDB_MODEL_VERSION` in `src/config.py` to deploy a specific model version.


2. **Start the FastAPI server:**
    ```bash
    uv run uvicorn src.api.main:app --reload
    ```
    The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## Docker Usage

1. **Create a `.env` file** in the project root with your W&B credentials:
    ```env
    WANDB_PROJECT="customer-churn-prediction"
    WANDB_ENTITY="your-wandb-entity"
    WANDB_API_KEY="your-wandb-api-key"
    ```

2. **Build and run the container:**
    ```bash
    docker-compose up --build
    ```
    The API will be accessible at [http://localhost:8000](http://localhost:8000).

---

## Kubernetes Deployment with KServe

This project can be deployed on a Kubernetes cluster with KServe for scalable model serving. It uses a separate transformer for preprocessing and a predictor for inference.

1.  **Prerequisites:**
    - A running Kubernetes cluster (e.g., `kind`).
    - KServe installed on the cluster.
    - An S3-compatible bucket for storing model artifacts.

2.  **Model Storage:**
    - The KServe `InferenceService` expects model artifacts to be in an S3 bucket.
    - After running the training pipeline, you must manually upload `model.bst` and `preprocessor.joblib` from the `artifacts/models` directory to your S3 bucket.
    - Create a Kubernetes secret named `s3-credentials` with your S3 credentials:
      ```bash
      kubectl create secret generic s3-credentials \
        --from-literal=AWS_ACCESS_KEY_ID='YOUR_ACCESS_KEY' \
        --from-literal=AWS_SECRET_ACCESS_KEY='YOUR_SECRET_KEY'
      ```

3.  **Deploy the `InferenceService`:**
    - Update `k8s/inferenceservice.yaml` with your Docker Hub username and S3 bucket details.
    - Apply the manifests:
      ```bash
      kubectl apply -f k8s/
      ```

---

## Kubeflow Pipeline

A Kubeflow pipeline is defined to orchestrate the ML workflow on Kubernetes.

1.  **Prerequisites:**
    - A running Kubernetes cluster with Kubeflow Pipelines installed.

2.  **Compile and Run the Pipeline:**
    - Compile the pipeline definition:
      ```bash
      uv run python src/kubeflow/pipeline.py
      ```
      This will generate `kubeflow_pipeline.yaml`.
    - Upload and run this YAML file through the Kubeflow Pipelines UI.

---

## CI/CD Pipeline

- **Continuous Integration (CI):**
  On every push or pull request, GitHub Actions runs linting and tests inside a reproducible container.

- **Continuous Training (CT):**
  On every push, a GitHub Actions workflow runs the full DVC pipeline in a containerized environment. This includes data fetching, processing, feature engineering, model training, and evaluation. The resulting model and preprocessor are logged to Weights & Biases (W&B) as artifacts.

- **Continuous Delivery (CD):**
  On push to `main`, GitHub Actions builds and pushes the API and KServe transformer Docker images to Docker Hub.

- **Model Versioning:**
  The latest trained model from the `main` branch is always tagged as `production` in W&B. The API and KServe transformer load the model artifact with the `production` alias.

- **Required GitHub Secrets:**
    - `DOCKERHUB_USERNAME`
    - `DOCKERHUB_TOKEN`
    - `B2_ACCESS_KEY_ID`
    - `B2_SECRET_ACCESS_KEY`
    - `KAGGLE_JSON`
    - `WANDB_API_KEY`
    - `WANDB_ENTITY`
    - `CICD_PAT`
- **Required GitHub Variables:**
    - `WANDB_PROJECT`

---
