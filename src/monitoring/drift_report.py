"""Data drift report: training data vs. what the live API received.

Step 1 - export prediction logs from CloudWatch (run in your terminal):

    aws logs filter-log-events \
      --log-group-name /ecs/green-taxi-api \
      --filter-pattern '{ $.event = "prediction" }' \
      --region eu-north-1 > reports/prediction_logs.json

Step 2 - pip install -r requirements-monitoring.txt

Step 3 - python src/monitoring/drift_report.py

Output: reports/drift_report.html (open it in the browser).
"""

import json
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
LOGS_PATH = PROJECT_ROOT / "reports" / "prediction_logs.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "drift_report.html"

INPUT_COLUMNS = [
    "vendorid",
    "Store_and_fwd_flag",
    "rate_code",
    "Pickup_longitude",
    "Pickup_latitude",
    "Dropoff_longitude",
    "Dropoff_latitude",
    "Passenger_count",
    "Trip_distance",
    "Trip_type",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
]

# Reference data: what the model was trained on
reference = pd.read_csv(REFERENCE_PATH, low_memory=False)
reference = reference[INPUT_COLUMNS].sample(
    n=min(5000, len(reference)),
    random_state=42,
)

# Current data: inputs that real users sent to /predict
cloudwatch_response = json.loads(LOGS_PATH.read_text(encoding="utf-8"))

rows = []

for event in cloudwatch_response["events"]:
    record = json.loads(event["message"])
    rows.append(record["input"])

current = pd.DataFrame(rows)[INPUT_COLUMNS]

print("Reference rows:", len(reference))
print("Current rows:", len(current))

report = Report([DataDriftPreset()])
snapshot = report.run(current, reference)

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
snapshot.save_html(str(REPORT_PATH))

print("Drift report saved to:", REPORT_PATH)
