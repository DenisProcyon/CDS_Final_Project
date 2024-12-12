from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

model_path = Path(__file__).parent / "best_model.pkl"
pipeline = joblib.load(model_path)

app = FastAPI(
    title="CDS Final Project: Decision Tree Prediction API",
    description="👨‍❤️‍👨👨‍❤️‍👨👨‍❤️‍👨"
)

class CarFeatures(BaseModel):
    Make: str
    Model: str
    Year: int
    Mileage: int
    Condition: str

@app.post("/predict", summary="Make a car price prediction")
async def predict(car: CarFeatures):
    try:
        car_data = pd.DataFrame([car.dict()])
        prediction = pipeline.predict(car_data)
        return {"prediction": float(prediction[0])}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", summary="Health check")
async def root():
    return {"message": "API is running. Use /predict to make predictions"}
