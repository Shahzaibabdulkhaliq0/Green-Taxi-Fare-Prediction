import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.models.train_model import PROJECT_ROOT


app = FastAPI(
    title="Green Taxi Fare Prediction API",
    description="API for predicting Green Taxi trip fares.",
    version="1.0.0",
)


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


@app.post("/predict")
def predict_fare(trip: TripInput):
    start_time = time.perf_counter()

    try:
        model = load_model()

        input_data = pd.DataFrame([trip.model_dump()])

        for column in ["vendorid", "rate_code", "Trip_type"]:
            if column in input_data.columns:
                input_data[column] = input_data[column].astype("object")

        prediction = float(model.predict(input_data)[0])
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # One JSON line per prediction. ECS sends stdout to CloudWatch Logs.
        print(json.dumps({
            "event": "prediction",
            "time": datetime.now(timezone.utc).isoformat(),
            "latency_ms": latency_ms,
            "predicted_fare": round(prediction, 2),
            "input": trip.model_dump(),
        }), flush=True)

        return {
            "predicted_fare": round(prediction, 2),
            "currency": "USD",
        }

    except FileNotFoundError as error:
        print(json.dumps({
            "event": "prediction_error",
            "status": 503,
            "error": str(error),
        }), flush=True)

        raise HTTPException(status_code=503, detail=str(error))

    except Exception as error:
        print(json.dumps({
            "event": "prediction_error",
            "status": 500,
            "error": str(error),
        }), flush=True)

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )
