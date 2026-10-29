from pathlib import Path

import joblib
import pandas as pd

from src.models.train_model import PROJECT_ROOT, load_params


def load_model():
    model_path = PROJECT_ROOT / "models" / "best_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run tune_model.py first."
        )

    model = joblib.load(model_path)
    print("Trained model loaded successfully.")

    return model


def prepare_input(data):
    df = pd.DataFrame([data])

    for column in ["vendorid", "rate_code", "Trip_type"]:
        if column in df.columns:
            df[column] = df[column].astype("object")

    return df


def predict_fare(model, data):
    input_data = prepare_input(data)
    predictions = model.predict(input_data)

    fare = float(predictions[0])

    print(f"Predicted fare: ${fare:.2f}")

    return fare


if __name__ == "__main__":
    model = load_model()

    sample_trip = {
        "vendorid": 2,
        "Store_and_fwd_flag": "N",
        "rate_code": 1,
        "Pickup_longitude": -73.9857,
        "Pickup_latitude": 40.7484,
        "Dropoff_longitude": -73.9851,
        "Dropoff_latitude": 40.7580,
        "Passenger_count": 1,
        "Trip_distance": 1.2,
        "Trip_type": 1,
        "pickup_hour": 14,
        "pickup_weekday": 2,
        "pickup_month": 6,
    }

    predict_fare(model, sample_trip)