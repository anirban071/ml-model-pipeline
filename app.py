from typing import Annotated
import logging
import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import numpy as np

logger = logging.getLogger("uvicorn.error")

# 1. Initialize API
app = FastAPI(title="Classification API")

# 2. Define expected input schema
# Features must be finite and within a sane range: the model casts inputs to float32,
# so values near float64 max (or inf) make scikit-learn raise.
FEATURE_LIMIT = 1e6
Feature = Annotated[float, Field(allow_inf_nan=False, ge=-FEATURE_LIMIT, le=FEATURE_LIMIT)]

class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_1: Feature
    feature_2: Feature
    feature_3: Feature
    feature_4: Feature

# 3. Load model globally (Cold start mitigation)
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(data: InferenceRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found on server.")

    # Convert incoming JSON payload to numpy array
    input_data = np.array([[
        data.feature_1,
        data.feature_2,
        data.feature_3,
        data.feature_4
    ]])

    try:
        prediction = model.predict(input_data)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    return {
        "status": "success",
        "prediction": int(prediction[0])
    }
