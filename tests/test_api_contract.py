from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from src.api.main import MODEL_PATH, TripInput, app
from src.features import build_features as features

PROJECT_ROOT = Path(__file__).resolve().parents[1]

client = TestClient(app)

VALID_TRIP = {
    "vendorid": 2,
    "Store_and_fwd_flag": "N",
    "rate_code": 1,
    "Pickup_longitude": -73.9857,
    "Pickup_latitude": 40.7484,
    "Dropoff_longitude": -73.9851,
    "Dropoff_latitude": 40.7580,
    "Passenger_count": 1,
    "Trip_distance": 3.5,
    "Trip_type": 1,
    "pickup_hour": 14,
    "pickup_weekday": 2,
    "pickup_month": 6,
}


# =========================
# MODEL FILE AND API CONTRACT
# =========================

def test_model_file_exists():
    assert MODEL_PATH.exists(), (
        "models/best_model.joblib is missing. "
        "The Docker image would start but /predict would return 503."
    )


def test_model_columns_match_api_input():
    model = joblib.load(MODEL_PATH)

    model_columns = set(model.feature_names_in_)
    api_columns = set(TripInput.model_fields)

    assert model_columns == api_columns


def test_real_model_returns_positive_fare():
    response = client.post("/predict", json=VALID_TRIP)

    assert response.status_code == 200

    body = response.json()

    assert body["currency"] == "USD"
    assert isinstance(body["predicted_fare"], float)
    assert body["predicted_fare"] > 0


# =========================
# INPUT VALIDATION
# =========================

@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("Passenger_count", 0),
        ("Trip_distance", 0),
        ("Trip_distance", -2.5),
        ("pickup_hour", 24),
        ("pickup_hour", -1),
        ("pickup_weekday", 7),
        ("pickup_month", 0),
        ("pickup_month", 13),
        ("Pickup_latitude", "not-a-number"),
    ],
)
def test_predict_rejects_bad_values(field, bad_value):
    payload = dict(VALID_TRIP)
    payload[field] = bad_value

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("missing_field", list(VALID_TRIP))
def test_predict_rejects_missing_field(missing_field):
    payload = dict(VALID_TRIP)
    del payload[missing_field]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# =========================
# DOCS PAGE (the URL used to check a deployment)
# =========================

def test_docs_page_is_available():
    response = client.get("/docs")

    assert response.status_code == 200


def test_openapi_lists_predict_and_health():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/predict" in paths
    assert "/health" in paths


# =========================
# DATA VALIDATION
# =========================

def test_bad_pickup_datetime_rows_are_dropped():
    df = pd.DataFrame({
        "Fare_amount": [10.0, 20.0],
        "pickup_datetime": ["01/01/2015 08:30:00 AM", "not a date"],
    })

    params = {
        "target": "Fare_amount",
        "drop_columns": [],
        "pickup_datetime_format": "%m/%d/%Y %I:%M:%S %p",
    }

    result = features.create_features(df, params)

    assert len(result) == 1
    assert result["pickup_hour"].iloc[0] == 8


def test_leakage_columns_are_removed_from_features():
    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as file:
        params = yaml.safe_load(file)["feature_engineering"]

    df = pd.DataFrame({
        "Fare_amount": [10.0],
        "Total_amount": [12.0],
        "Tip_amount": [1.0],
        "pickup_datetime": ["01/01/2015 08:30:00 AM"],
        "dropoff_datetime": ["01/01/2015 08:45:00 AM"],
    })

    result = features.create_features(df, params)

    assert "Total_amount" not in result.columns
    assert "Tip_amount" not in result.columns
    assert "Fare_amount" in result.columns


# =========================
# PIPELINE CONFIG CONSISTENCY
# =========================

def test_dvc_stage_scripts_exist():
    with (PROJECT_ROOT / "dvc.yaml").open(encoding="utf-8") as file:
        stages = yaml.safe_load(file)["stages"]

    for name, stage in stages.items():
        script = stage["cmd"].replace("python ", "")

        assert (PROJECT_ROOT / script).exists(), (
            f"Stage {name} points to a missing script: {script}"
        )


def test_dvc_param_sections_exist_in_params_yaml():
    with (PROJECT_ROOT / "dvc.yaml").open(encoding="utf-8") as file:
        stages = yaml.safe_load(file)["stages"]

    with (PROJECT_ROOT / "params.yaml").open(encoding="utf-8") as file:
        params = yaml.safe_load(file)

    for name, stage in stages.items():
        for section in stage.get("params", []):
            assert section in params, (
                f"Stage {name} needs params section: {section}"
            )
