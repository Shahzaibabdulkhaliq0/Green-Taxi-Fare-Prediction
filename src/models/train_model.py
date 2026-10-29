from pathlib import Path

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params():
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)["model_training"]


def load_data():
    data_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    df = pd.read_csv(data_path)

    print("Training data:", df.shape)
    return df


def prepare_data(df, params):
    target = params["target"]

    X = df.drop(columns=[target])
    y = df[target]

    for column in ["vendorid", "rate_code", "Trip_type"]:
        if column in X.columns:
            X[column] = X[column].astype("object")

    categorical_columns = X.select_dtypes(
        include=["object", "category", "str"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    print("Numerical features:", len(numerical_columns))
    print("Categorical features:", len(categorical_columns))

    return X, y, numerical_columns, categorical_columns


def create_pipeline(model, numerical_columns, categorical_columns):
    numerical_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
    ])

    categorical_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numbers", numerical_steps, numerical_columns),
        ("categories", categorical_steps, categorical_columns),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])


def compare_models(
    X,
    y,
    numerical_columns,
    categorical_columns,
    params,
):
    model_configs = params["models"]

    models = {}

    if "linear_regression" in model_configs:
        models["Linear Regression"] = LinearRegression(
            **model_configs["linear_regression"]
        )

    if "random_forest" in model_configs:
        models["Random Forest"] = RandomForestRegressor(
            **model_configs["random_forest"]
        )

    if not models:
        raise ValueError("No models configured in params.yaml")

    results = []

    for name, model in models.items():
        print(f"\nEvaluating: {name}")

        pipeline = create_pipeline(
            model,
            numerical_columns,
            categorical_columns,
        )

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=params["cv_folds"],
            scoring={
                "R2": "r2",
                "MAE": "neg_mean_absolute_error",
                "RMSE": "neg_root_mean_squared_error",
            },
            n_jobs=1,
        )

        mean_r2 = float(scores["test_R2"].mean())
        mean_mae = float(-scores["test_MAE"].mean())
        mean_rmse = float(-scores["test_RMSE"].mean())

        results.append({
            "model": name,
            "R2": mean_r2,
            "MAE": mean_mae,
            "RMSE": mean_rmse,
        })

        print(
            f"R2={mean_r2:.4f}, "
            f"MAE={mean_mae:.4f}, "
            f"RMSE={mean_rmse:.4f}"
        )

    results = pd.DataFrame(results).sort_values(
        by=["R2", "RMSE", "MAE"],
        ascending=[False, True, True],
    )

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        reports_dir / "model_comparison.csv",
        index=False,
    )

    print("\nModel comparison:")
    print(results.to_string(index=False))

    best_name = results.iloc[0]["model"]
    best_model = models[best_name]

    print("\nAutomatically selected model:", best_name)
    print("Selection metric: Highest mean CV R2")

    return best_name, best_model


if __name__ == "__main__":
    params = load_params()

    df = load_data()

    X, y, numerical_columns, categorical_columns = prepare_data(
        df,
        params,
    )

    best_name, best_model = compare_models(
        X,
        y,
        numerical_columns,
        categorical_columns,
        params,
    )