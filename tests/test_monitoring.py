import json

from fastapi.testclient import TestClient

from src.api.main import app

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


def read_log_lines(captured_text):
    lines = []

    for line in captured_text.splitlines():
        if line.startswith("{"):
            lines.append(json.loads(line))

    return lines


def test_prediction_writes_one_json_log_line(capsys):
    response = client.post("/predict", json=VALID_TRIP)

    assert response.status_code == 200

    logs = read_log_lines(capsys.readouterr().out)

    assert len(logs) == 1

    log = logs[0]

    assert log["event"] == "prediction"
    assert log["predicted_fare"] == response.json()["predicted_fare"]
    assert log["latency_ms"] >= 0
    assert log["input"] == VALID_TRIP
    assert "time" in log


def test_missing_model_writes_error_log(capsys, monkeypatch):
    def mock_load_model():
        raise FileNotFoundError("Model not found")

    monkeypatch.setattr("src.api.main.load_model", mock_load_model)

    response = client.post("/predict", json=VALID_TRIP)

    assert response.status_code == 503

    logs = read_log_lines(capsys.readouterr().out)

    assert logs[0]["event"] == "prediction_error"
    assert logs[0]["status"] == 503


def test_invalid_input_writes_no_prediction_log(capsys):
    client.post("/predict", json={"vendorid": 2})

    logs = read_log_lines(capsys.readouterr().out)

    assert logs == []
