import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier

import wandb
from src.config import MODEL_DIR, WANDB_MODEL_NAME

preprocessor: dict | None = None
model: XGBClassifier | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, preprocessor

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY")
    )
    artifact = run.use_artifact(
        f"{WANDB_MODEL_NAME}", type="model", aliases=["production"]
    )
    artifact_dir = artifact.download(root=MODEL_DIR)

    model = XGBClassifier()
    model.load_model(os.path.join(artifact_dir, "model.json"))
    preprocessor = joblib.load(os.path.join(artifact_dir, "preprocessor.joblib"))
    run.finish()
    yield


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)


class PredictionInput(BaseModel):
    age: int = 30
    gender: str = "Male"
    tenure: int = 12
    usage_frequency: int = 15
    support_calls: int = 3
    payment_delay: int = 10
    subscription_type: str = "Premium"
    contract_length: str = "Monthly"
    total_spend: float = 500.0
    last_interaction: int = 15


class PredictionOutput(BaseModel):
    prediction: int


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    input_df = pd.DataFrame([input_data.model_dump()])

    cat_cols = preprocessor["categorical_cols"]
    num_cols = preprocessor["numerical_cols"]
    encoded_cols = preprocessor["encoder"].transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(
        encoded_cols,
        columns=preprocessor["feature_names_out"],
        index=input_df.index,
    )

    processed_df = pd.concat([input_df[num_cols], encoded_df], axis=1)

    ordered_cols = model.get_booster().feature_names
    processed_df = processed_df[ordered_cols]

    prediction = model.predict(processed_df)[0]
    return {"prediction": int(prediction)}


@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API"}
