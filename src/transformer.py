import os
from typing import Dict

import joblib
import pandas as pd

import wandb
from kserve import Model, ModelServer
from src.config import MODEL_DIR, WANDB_MODEL_NAME


class Transformer(Model):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.preprocessor = None
        self.model_input_features = None

    def load(self) -> bool:
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT"), entity=os.getenv("WANDB_ENTITY")
        )
        artifact = run.use_artifact(f"{WANDB_MODEL_NAME}:production", type="model")
        artifact_dir = artifact.download(root=MODEL_DIR)
        self.preprocessor = joblib.load(
            os.path.join(artifact_dir, "preprocessor.joblib")
        )
        run.finish()
        print("Transformer loaded successfully.")
        return True

    def preprocess(self, inputs: Dict) -> Dict:
        input_df = pd.DataFrame(inputs["instances"])

        cat_cols = self.preprocessor["categorical_cols"]
        num_cols = self.preprocessor["numerical_cols"]

        encoded_cols = self.preprocessor["encoder"].transform(input_df[cat_cols])
        encoded_df = pd.DataFrame(
            encoded_cols,
            columns=self.preprocessor["feature_names_out"],
            index=input_df.index,
        )

        processed_df = pd.concat([input_df[num_cols], encoded_df], axis=1)

        return {"instances": processed_df.to_numpy().tolist()}


if __name__ == "__main__":
    model = Transformer(
        name="churn-transformer", predictor_host="churn-predictor-default"
    )
    model.load()
    ModelServer().start([model])
