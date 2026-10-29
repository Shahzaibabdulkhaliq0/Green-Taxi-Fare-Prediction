from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

from src.utils.mlflow_utils import (
    setup_mlflow,
    log_tuning_results,
)

from src.models.train_model import (
    PROJECT_ROOT,
    load_params,
    load_data,
    prepare_data,
    create_pipeline,
)


def load_selected_model_name():
    report_path = PROJECT_ROOT / "reports" / "model_comparison.csv"

    if not report_path.exists():
        raise FileNotFoundError(
            f"Model comparison report not found: {report_path}. "
            "Run train_model.py first."
        )

    results = pd.read_csv(report_path)

    if results.empty:
        raise ValueError("Model comparison report is empty.")

    results = results.sort_values(
        by=["R2", "RMSE", "MAE"],
        ascending=[False, True, True],
    )

    model_name = results.iloc[0]["model"]
    print("Selected model:", model_name)

    return model_name


def get_selected_model(model_name, params):
    model_configs = params["models"]

    if model_name == "Linear Regression":
        return LinearRegression(
            **model_configs["linear_regression"]
        )

    if model_name == "Random Forest":
        return RandomForestRegressor(
            **model_configs["random_forest"]
        )

    raise ValueError(f"Unsupported model: {model_name}")


def get_parameter_grid(model, params):
    tuning_configs = params["hyperparameter_tuning"]

    if isinstance(model, LinearRegression):
        model_config = tuning_configs["linear_regression"]

    elif isinstance(model, RandomForestRegressor):
        model_config = tuning_configs["random_forest"]

    else:
        raise ValueError(
            f"No tuning configuration for {type(model).__name__}"
        )

    return {
        f"model__{parameter}": values
        for parameter, values in model_config.items()
    }


def tune_selected_model(
    pipeline,
    parameter_grid,
    X,
    y,
    params,
    model_name,
):
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring={
            "R2": "r2",
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
        },
        refit="R2",
        cv=params["cv_folds"],
        n_jobs=1,
        verbose=2,
        return_train_score=False,
    )

    search.fit(X, y)

    print("\nBest parameters:", search.best_params_)
    print("Best mean CV R2:", round(search.best_score_, 4))

    results = pd.DataFrame(search.cv_results_)

    results = results[
        [
            "params",
            "mean_test_R2",
            "mean_test_MAE",
            "mean_test_RMSE",
            "rank_test_R2",
        ]
    ].copy()

    results["mean_test_MAE"] *= -1
    results["mean_test_RMSE"] *= -1
    results = results.sort_values("rank_test_R2")

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    grid_results_path = reports_dir / "grid_search_results.csv"
    grid_results_path and results.to_csv(
        grid_results_path,
        index=False,
    )

    best_parameters_path = reports_dir / "best_parameters.csv"

    best_parameters = pd.DataFrame(
        list(search.best_params_.items()),
        columns=["parameter", "value"],
    )
    best_parameters.to_csv(best_parameters_path, index=False)

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "best_model.joblib"
    joblib.dump(search.best_estimator_, model_path, compress=3)

    best_index = search.best_index_

    best_metrics = {
        "CV_R2": float(search.best_score_),
        "CV_MAE": float(
            -search.cv_results_["mean_test_MAE"][best_index]
        ),
        "CV_RMSE": float(
            -search.cv_results_["mean_test_RMSE"][best_index]
        ),
    }

    log_tuning_results(
        model_name=model_name,
        best_params=search.best_params_,
        metrics=best_metrics,
        artifact_paths=[
            grid_results_path,
            best_parameters_path,
            model_path,
        ],
    )

    print(f"\nSaved final model: {model_path}")
    print(f"Saved tuning results: {grid_results_path}")
    print(f"Saved best parameters: {best_parameters_path}")

    return search.best_estimator_


if __name__ == "__main__":
    params = load_params()
    setup_mlflow(params["experiment_name"])

    model_name = load_selected_model_name()
    df = load_data()

    X, y, numerical_columns, categorical_columns = prepare_data(
        df,
        params,
    )

    model = get_selected_model(model_name, params)

    pipeline = create_pipeline(
        model,
        numerical_columns,
        categorical_columns,
    )

    parameter_grid = get_parameter_grid(model, params)

    tune_selected_model(
        pipeline,
        parameter_grid,
        X,
        y,
        params,
        model_name,
    )