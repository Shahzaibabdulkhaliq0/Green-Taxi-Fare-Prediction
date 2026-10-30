from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.train_model import PROJECT_ROOT


app = FastAPI(
    title="Green Taxi Fare Prediction API",
    description="API for predicting Green Taxi trip fares.",
    version="1.0.0",
)


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


class TripInput(BaseModel):
    vendorid: int
    Store_and_fwd_flag: str
    rate_code: int
    Pickup_longitude: float
    Pickup_latitude: float
    Dropoff_longitude: float
    Dropoff_latitude: float
    Passenger_count: int = Field(ge=1)
    Trip_distance: float = Field(gt=0)
    Trip_type: int
    pickup_hour: int = Field(ge=0, le=23)
    pickup_weekday: int = Field(ge=0, le=6)
    pickup_month: int = Field(ge=1, le=12)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run tune_model.py first."
        )

    return joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {
        "message": "Green Taxi Fare Prediction API",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


@app.post("/predict")
def predict_fare(trip: TripInput):
    try:
        model = load_model()

        input_data = pd.DataFrame([trip.model_dump()])

        for column in ["vendorid", "rate_code", "Trip_type"]:
            if column in input_data.columns:
                input_data[column] = input_data[column].astype("object")

        prediction = float(model.predict(input_data)[0])

        return {
            "predicted_fare": round(prediction, 2),
            "currency": "USD",
        }

    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )