import json
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
model = joblib.load(BASE / "model.joblib")
meta = json.loads((BASE / "model_meta.json").read_text())

app = FastAPI(title="Breast Cancer Model API")


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=30, max_length=30)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": meta["model_name"],
        "model_version": meta["model_version"],
        "run_id": meta["run_id"],
    }


@app.post("/predict")
def predict(req: PredictRequest):
    pred = int(model.predict([req.features])[0])
    result = {
        "prediction": pred,
        "label": "benign" if pred == 1 else "malignant",
        "model_version": meta["model_version"],
    }
    if hasattr(model, "predict_proba"):
        result["probability"] = float(max(model.predict_proba([req.features])[0]))
    return result