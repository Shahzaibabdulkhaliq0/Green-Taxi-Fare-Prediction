from pathlib import Path

import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_mlflow(experiment_name):
    tracking_uri = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_model_comparison(model_name, params, metrics):
    with mlflow.start_run(run_name=model_name):
        safe_params = {
            str(key): str(value)
            for key, value in params.items()
        }

        mlflow.log_params(safe_params)
        mlflow.log_metrics(metrics)


def log_training_results(results):
    with mlflow.start_run(run_name="Model Comparison"):
        for _, row in results.iterrows():
            model_name = str(row["model"])

            mlflow.log_metrics({
                f"{model_name}_R2": float(row["R2"]),
                f"{model_name}_MAE": float(row["MAE"]),
                f"{model_name}_RMSE": float(row["RMSE"]),
            })

        log_artifact(
            PROJECT_ROOT / "reports" / "model_comparison.csv"
        )


def log_tuning_results(
    model_name,
    best_params,
    metrics,
    artifact_paths=None,
):
    with mlflow.start_run(run_name=f"{model_name}-Tuned"):
        safe_params = {
            str(key): str(value)
            for key, value in best_params.items()
        }

        mlflow.log_params(safe_params)
        mlflow.log_metrics(metrics)

        if artifact_paths:
            for artifact_path in artifact_paths:
                log_artifact(artifact_path)


def log_evaluation(metrics, report_path=None):
    with mlflow.start_run(run_name="Final Model Evaluation"):
        mlflow.log_metrics(metrics)

        if report_path is not None:
            log_artifact(report_path)


def log_artifact(file_path):
    path = Path(file_path)

    if path.exists() and path.is_file():
        mlflow.log_artifact(str(path))