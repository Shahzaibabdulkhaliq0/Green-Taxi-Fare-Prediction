from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)

GOOD_TRIP = {
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


def test_real_model_predict_through_api():
    response = client.post("/predict", json=GOOD_TRIP)

    assert response.status_code == 200
    assert response.json()["currency"] == "USD"
    assert response.json()["predicted_fare"] > 0


def test_zero_passengers_rejected():
    response = client.post("/predict", json=dict(GOOD_TRIP, Passenger_count=0))
    assert response.status_code == 422


def test_zero_distance_rejected():
    response = client.post("/predict", json=dict(GOOD_TRIP, Trip_distance=0))
    assert response.status_code == 422


def test_hour_out_of_range_rejected():
    response = client.post("/predict", json=dict(GOOD_TRIP, pickup_hour=24))
    assert response.status_code == 422


def test_weekday_out_of_range_rejected():
    response = client.post("/predict", json=dict(GOOD_TRIP, pickup_weekday=7))
    assert response.status_code == 422


def test_month_out_of_range_rejected():
    response = client.post("/predict", json=dict(GOOD_TRIP, pickup_month=13))
    assert response.status_code == 422


def test_text_instead_of_number_rejected():
    response = client.post("/predict", json=dict(GOOD_TRIP, Trip_distance="far"))
    assert response.status_code == 422


def test_missing_field_rejected():
    bad_trip = dict(GOOD_TRIP)
    del bad_trip["Trip_distance"]

    response = client.post("/predict", json=bad_trip)

    assert response.status_code == 422
