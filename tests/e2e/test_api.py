import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

os.environ["WANDB_PROJECT"] = "test-project"
os.environ["WANDB_ENTITY"] = "test-entity"

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    mock_preprocessor = {
        "categorical_cols": [
            "gender",
            "subscription_type",
            "contract_length",
        ],
        "numerical_cols": [
            "age",
            "tenure",
            "usage_frequency",
            "support_calls",
            "payment_delay",
            "total_spend",
            "last_interaction",
        ],
        "encoder": OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, drop="first"
        ),
        "feature_names_out": [
            "gender_Male",
            "subscription_type_Premium",
            "subscription_type_Standard",
            "contract_length_Monthly",
            "contract_length_Yearly",
        ],
    }
    dummy_cat_data = pd.DataFrame(
        {
            "gender": ["Male", "Female"],
            "subscription_type": ["Basic", "Premium", "Standard"],
            "contract_length": ["Monthly", "Annual", "Yearly"],
        }
    )
    mock_preprocessor["encoder"].fit(dummy_cat_data)

    mock_model = MagicMock(spec=XGBClassifier)
    mock_model.predict.return_value = np.array([1])
    booster = MagicMock()
    booster.feature_names = [
        "age",
        "tenure",
        "usage_frequency",
        "support_calls",
        "payment_delay",
        "total_spend",
        "last_interaction",
        "gender_Male",
        "subscription_type_Premium",
        "subscription_type_Standard",
        "contract_length_Monthly",
        "contract_length_Yearly",
    ]
    mock_model.get_booster.return_value = booster

    with (
        patch("src.api.main.wandb") as mock_wandb,
        patch("src.api.main.joblib.load", return_value=mock_preprocessor),
        patch("src.api.main.XGBClassifier") as mock_xgb_class,
    ):
        mock_xgb_class.return_value = mock_model
        mock_artifact = MagicMock()
        mock_artifact.download.return_value = "/fake/dir"
        mock_wandb.init.return_value.use_artifact.return_value = mock_artifact

        with TestClient(app) as test_client:
            yield test_client


def test_read_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Churn Prediction API"}


def test_predict_success(client: TestClient):
    input_data = {
        "age": 42,
        "gender": "Female",
        "tenure": 24,
        "usage_frequency": 20,
        "support_calls": 1,
        "payment_delay": 5,
        "subscription_type": "Standard",
        "contract_length": "Annual",
        "total_spend": 1200.50,
        "last_interaction": 10,
    }

    response = client.post("/predict", json=input_data)

    assert response.status_code == 200
    response_data = response.json()
    assert "prediction" in response_data
    assert response_data["prediction"] in [0, 1]
    assert response_data["prediction"] == 1


def test_predict_default_values(client: TestClient):
    response = client.post("/predict", json={})

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["prediction"] == 1


def test_predict_validation_error(client: TestClient):
    invalid_data = {"age": "not-an-int"}

    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
    assert "validation_error" in response.json()["detail"][0]["type"]
