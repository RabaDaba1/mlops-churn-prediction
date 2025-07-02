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

2.  **Create a virtual environment and install dependencies:**
    ```bash
    uv python install 3.11
    uv sync --extra dev
    ```

3.  **Set up Kaggle API credentials:**
    To download the dataset, you need a Kaggle account and an API token.
    -   Go to your Kaggle account settings, and click on "Create New API Token".
    -   This will download a `kaggle.json` file.
    -   Place this file in `~/.kaggle/kaggle.json`.

