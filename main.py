"""FastAPI service serving predictions from the exported arrhythmia model artifacts."""
import json

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

ARTIFACTS_DIR = "artifacts"

app = FastAPI(title="Arrhythmia Classifier")

artifacts = {}


@app.on_event("startup")
def load_artifacts():
    artifacts["imputer"] = joblib.load(f"{ARTIFACTS_DIR}/imputer.joblib")
    artifacts["scaler"] = joblib.load(f"{ARTIFACTS_DIR}/scaler.joblib")
    artifacts["pca"] = joblib.load(f"{ARTIFACTS_DIR}/pca.joblib")
    artifacts["model"] = joblib.load(f"{ARTIFACTS_DIR}/model.joblib")
    with open(f"{ARTIFACTS_DIR}/feature_schema.json") as f:
        schema = json.load(f)
    artifacts["feature_columns"] = schema["feature_columns"]
    artifacts["class_names"] = schema["class_names"]


class PredictRequest(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def validate_length(cls, value):
        expected = len(artifacts["feature_columns"])
        if len(value) != expected:
            raise ValueError(
                f"features must have exactly {expected} values, got {len(value)}"
            )
        return value


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    return {
        "feature_columns": artifacts["feature_columns"],
        "class_names": artifacts["class_names"],
    }


@app.post("/predict")
def predict(request: PredictRequest):
    x = np.array(request.features).reshape(1, -1)

    x_imputed = artifacts["imputer"].transform(x)
    x_scaled = artifacts["scaler"].transform(x_imputed)
    x_pca = artifacts["pca"].transform(x_scaled)

    model = artifacts["model"]
    predicted_class = int(model.predict(x_pca)[0])
    probabilities = model.predict_proba(x_pca)[0]

    class_index = list(model.classes_).index(predicted_class)
    confidence = float(probabilities[class_index])

    class_names = artifacts["class_names"]
    if 1 <= predicted_class <= len(class_names):
        class_name = class_names[predicted_class - 1]
    else:
        class_name = "Unknown"

    return {
        "predicted_class": predicted_class,
        "class_name": class_name,
        "confidence": confidence,
    }
