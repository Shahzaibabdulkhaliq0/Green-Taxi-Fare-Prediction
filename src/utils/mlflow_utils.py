
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_mlflow(experiment_name):
    tracking_uri = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_model_comparison(model_name, params, metrics):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


def log_artifact(file_path):
    path = Path(file_path)

    if path.exists():
        mlflow.log_artifact(str(path))