from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Define the Input Data Model with REAL Default Values
class HousingFeatures(BaseModel):
    MedInc: float = 8.3252        # Default value instead of 0
    HouseAge: float = 41.0
    AveRooms: float = 6.98
    AveBedrms: float = 1.02
    Population: float = 322.0
    AveOccup: float = 2.55
    Latitude: float = 37.88
    Longitude: float = -122.23

@app.get("/")
def home():
    return {"message": "Housing Price Prediction API is Running!"}

@app.post("/predict")
def predict(features: HousingFeatures):
    # Load the model
    model = joblib.load("model.pkl")
    
    # Prepare data
    data = pd.DataFrame([features.dict()])
    
    # Predict
    prediction = model.predict(data)
    return {"prediction": prediction[0]}