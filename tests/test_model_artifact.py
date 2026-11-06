from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"

SAMPLE_TRIP = {
    "vendorid": 2,
    "Store_and_fwd_flag": "N",
    "rate_code": 1,
    "Pickup_longitude": -73.95,
    "Pickup_latitude": 40.75,
    "Dropoff_longitude": -73.98,
    "Dropoff_latitude": 40.76,
    "Passenger_count": 1,
    "Trip_distance": 3.5,
    "Trip_type": 1,
    "pickup_hour": 14,
    "pickup_weekday": 2,
    "pickup_month": 6,
}


def test_model_file_exists():
    assert MODEL_PATH.exists(), "models/best_model.joblib is missing"


def test_model_gives_a_sensible_fare():
    model = joblib.load(MODEL_PATH)

    trip = pd.DataFrame([SAMPLE_TRIP])
    trip[["vendorid", "rate_code", "Trip_type"]] = trip[
        ["vendorid", "rate_code", "Trip_type"]
    ].astype("object")

    fare = float(model.predict(trip)[0])

    assert 2 < fare < 100


def test_longer_trip_costs_more_than_short_trip():
    model = joblib.load(MODEL_PATH)

    short_trip = dict(SAMPLE_TRIP, Trip_distance=0.5)
    long_trip = dict(SAMPLE_TRIP, Trip_distance=15.0)

    trips = pd.DataFrame([short_trip, long_trip])
    trips[["vendorid", "rate_code", "Trip_type"]] = trips[
        ["vendorid", "rate_code", "Trip_type"]
    ].astype("object")

    short_fare, long_fare = model.predict(trips)

    assert long_fare > short_fare
