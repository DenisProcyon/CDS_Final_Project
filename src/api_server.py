from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

from pathlib import Path

from preprocessor import Preprocessor

model_path = Path(__file__).parent / "best_model.pkl"

model = joblib.load(model_path)
maetadata_path = Path(__file__).parent.parent / "data/Safercar_data.csv"
metadata = pd.read_csv(maetadata_path)

app = FastAPI(title="CDS Final Project: Decision Tree Prediction API", description="🇺🇦🇺🇦🇺🇦")

class CarFeatures(BaseModel):
    Make: str
    Model: str
    Year: int
    Mileage: int
    Condition: str

def preprocess_input(data: pd.DataFrame):
    data = data.drop(columns="Model", errors="ignore")

    preprocessor = Preprocessor(data=data, metadata=metadata)
    pipeline = [
        (preprocessor.assign_metadata_avg, {"columns": ["OVERALL_STARS", "CURB_WEIGHT", "MIN_GROSS_WEIGHT"]}),
        (preprocessor.assign_countries, {}),
        (preprocessor.assign_age, {}),
        (preprocessor.transform_cat_to_num, {"columns": ["Condition", "Country", "Make"]}),
    ]

    for func, kwargs in pipeline:
        preprocessor.data = func(**kwargs)

    cleaned_data = preprocessor.data
    print("Cleaned data:\n", cleaned_data)

    return cleaned_data

@app.post("/predict", summary="Make a car price prediction")
async def predict(cars: list[CarFeatures]):
    try:
        cars_data = pd.DataFrame([car.dict() for car in cars])
        print("Input data as DataFrame:\n", cars_data)

        features = preprocess_input(cars_data)

        predictions = model.predict(features)
        return {"predictions": predictions.tolist()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", summary="Health check")
async def root():
    return {"message": "API is running. Use /predict to make predictions"}