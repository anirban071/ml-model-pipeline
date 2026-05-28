from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# 1. Initialize API
app = FastAPI(title="Classification API")

# 2. Define expected input schema
class InferenceRequest(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float

# 3. Load model globally (Cold start mitigation)
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

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
    
    prediction = model.predict(input_data)
    
    return {
        "status": "success",
        "prediction": int(prediction[0])
    }