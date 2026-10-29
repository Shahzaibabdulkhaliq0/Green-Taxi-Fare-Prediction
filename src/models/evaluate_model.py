from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.models.train_model import PROJECT_ROOT, load_params
from src.utils.mlflow_utils import (
    setup_mlflow,
    log_evaluation,
)

def load_model():
    model_path = PROJECT_ROOT / "models" / "best_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. "
            "Run tune_model.py first."
        )

    model = joblib.load(model_path)
    print("Final model loaded successfully")

    return model


def load_test_data():
    params = load_params()
    target = params["target"]

    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    df = pd.read_csv(test_path)

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in test data."
        )

    print("Test dataset:", df.shape)

    return df


def evaluate_model(model, df):
    target = load_params()["target"]

    X_test = df.drop(columns=[target]).copy()
    y_test = df[target]

    for column in ["vendorid", "rate_code", "Trip_type"]:
        if column in X_test.columns:
            X_test[column] = X_test[column].astype("object")

    predictions = model.predict(X_test)

    results = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(mean_squared_error(y_test, predictions) ** 0.5),
        "R2": float(r2_score(y_test, predictions)),
    }

    print("\nFinal Test Results")
    print("------------------")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    return pd.DataFrame([results])



if __name__ == "__main__":
    params = load_params()
    setup_mlflow(params["experiment_name"])

    model = load_model()
    test_df = load_test_data()

    results = evaluate_model(model, test_df)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "test_evaluation.csv"
    results.to_csv(report_path, index=False)

    metrics = {
        metric: float(value)
        for metric, value in results.iloc[0].items()
    }

    log_evaluation(
        metrics=metrics,
        report_path=report_path,
    )

    print(f"\nReport saved: {report_path}")
    print("Evaluation metrics logged to MLflow.")