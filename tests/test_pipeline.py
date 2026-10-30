
import pandas as pd
import pytest

from src.data import make_dataset as dataset
from src.features import build_features as features
from src.models import train_model as training
from src.models import tune_model as tuning
from src.models import evaluate_model as evaluation
from src.models import predict_model as prediction

# =========================
# DATA PREPARATION TESTS
# =========================

def test_load_data(tmp_path, monkeypatch):
    file_path = tmp_path / "sample.csv"

    pd.DataFrame({
        "Fare_amount": [10.0, 20.0]
    }).to_csv(file_path, index=False)

    monkeypatch.setattr(dataset, "PROJECT_ROOT", tmp_path)

    params = {
        "raw_data_path": "sample.csv"
    }

    df = dataset.load_data(params)

    assert not df.empty
    assert "Fare_amount" in df.columns
    assert len(df) == 2


def test_clean_data():
    df = pd.DataFrame({
        "Fare_amount": [10, 20, 0, -5, None, 10],
        "VendorID": [1, 2, 3, 4, 5, 1]
    })

    cleaned = dataset.clean_data(df, "Fare_amount")

    assert len(cleaned) == 2
    assert cleaned["Fare_amount"].min() > 0
    assert cleaned.duplicated().sum() == 0


def test_clean_data_invalid_target():
    df = pd.DataFrame({
        "Fare_amount": ["invalid", "abc", None]
    })

    cleaned = dataset.clean_data(df, "Fare_amount")

    assert cleaned.empty


def test_save_data_isolated(tmp_path, monkeypatch):
    train = pd.DataFrame({"Fare_amount": [10.0, 20.0]})
    test = pd.DataFrame({"Fare_amount": [30.0]})

    monkeypatch.setattr(dataset, "PROJECT_ROOT", tmp_path)

    dataset.save_data(
        train,
        test,
        {"interim_data_dir": "data/interim"}
    )

    train_path = tmp_path / "data/interim/train.csv"
    test_path = tmp_path / "data/interim/test.csv"

    assert train_path.exists()
    assert test_path.exists()


def test_load_params():
    params = dataset.load_params()

    assert "target" in params
    assert "test_size" in params
    assert "random_state" in params


def test_data_main(monkeypatch):
    monkeypatch.setattr(dataset, "load_data", lambda params: pd.DataFrame({
        "Fare_amount": [10, 20, 30, 40, 50],
        "VendorID": [1, 2, 1, 2, 1]
    }))

    monkeypatch.setattr(dataset, "load_params", lambda: {
        "target": "Fare_amount",
        "test_size": 0.2,
        "random_state": 42
    })

    saved = {}

    def mock_save_data(train, test, params):
        saved["train"] = train
        saved["test"] = test

    monkeypatch.setattr(dataset, "save_data", mock_save_data)

    dataset.main()

    assert "train" in saved
    assert "test" in saved
    assert len(saved["train"]) + len(saved["test"]) == 5


# =========================
# FEATURE ENGINEERING TESTS
# =========================

def test_feature_load_params():
    params = features.load_params()

    assert "input_dir" in params
    assert "output_dir" in params
    assert "target" in params


def test_feature_load_data(tmp_path):
    file_path = tmp_path / "sample.csv"

    pd.DataFrame({
        "Fare_amount": [10.0, 20.0]
    }).to_csv(file_path, index=False)

    df = features.load_data(file_path)

    assert not df.empty
    assert len(df) == 2


def test_create_features():
    df = pd.DataFrame({
        "Fare_amount": [10.0, 20.0],
        "pickup_datetime": [
            "01/01/2015 08:30:00 AM",
            "01/02/2015 09:45:00 AM"
        ],
        "dropoff_datetime": [
            "01/01/2015 08:45:00 AM",
            "01/02/2015 10:00:00 AM"
        ],
        "Total_amount": [12.0, 22.0]
    })

    params = {
        "target": "Fare_amount",
        "drop_columns": ["Total_amount", "dropoff_datetime"],
        "pickup_datetime_format": "%m/%d/%Y %I:%M:%S %p"
    }

    result = features.create_features(df, params)

    assert "pickup_hour" in result.columns
    assert "pickup_weekday" in result.columns
    assert "pickup_month" in result.columns
    assert "Fare_amount" in result.columns
    assert "pickup_datetime" not in result.columns
    assert "dropoff_datetime" not in result.columns


def test_create_features_preserves_original_data():
    df = pd.DataFrame({
        "Fare_amount": [10.0],
        "pickup_datetime": ["01/01/2015 08:30:00 AM"],
        "dropoff_datetime": ["01/01/2015 08:45:00 AM"],
        "Total_amount": [12.0]
    })

    original_columns = df.columns.tolist()

    params = {
        "target": "Fare_amount",
        "drop_columns": ["Total_amount", "dropoff_datetime"],
        "pickup_datetime_format": "%m/%d/%Y %I:%M:%S %p"
    }

    features.create_features(df, params)

    assert df.columns.tolist() == original_columns


def test_feature_save_data(tmp_path):
    train = pd.DataFrame({"Fare_amount": [10.0]})
    test = pd.DataFrame({"Fare_amount": [20.0]})

    features.save_data(train, test, tmp_path)

    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "test.csv").exists()


def test_feature_main(monkeypatch):
    sample_df = pd.DataFrame({
        "Fare_amount": [10.0],
        "pickup_datetime": ["01/01/2015 08:30:00 AM"],
        "dropoff_datetime": ["01/01/2015 08:45:00 AM"],
        "Total_amount": [12.0]
    })

    monkeypatch.setattr(features, "load_params", lambda: {
        "input_dir": "data/interim",
        "output_dir": "data/processed",
        "target": "Fare_amount",
        "drop_columns": ["Total_amount", "dropoff_datetime"],
        "pickup_datetime_format": "%m/%d/%Y %I:%M:%S %p"
    })

    monkeypatch.setattr(features, "load_data", lambda path: sample_df.copy())

    saved = {}

    def mock_save_data(train, test, output_path):
        saved["train"] = train
        saved["test"] = test

    monkeypatch.setattr(features, "save_data", mock_save_data)

    features.main()

    assert "train" in saved
    assert "test" in saved
    assert "pickup_hour" in saved["train"].columns


# =========================
# MODEL TRAINING TESTS
# =========================

def test_training_load_params():
    params = training.load_params()

    assert "target" in params
    assert "models" in params
    assert "cv_folds" in params


def test_training_load_data(tmp_path, monkeypatch):
    df = pd.DataFrame({
        "Fare_amount": [10.0, 20.0]
    })

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)

    df.to_csv(data_dir / "train.csv", index=False)

    monkeypatch.setattr(training, "PROJECT_ROOT", tmp_path)

    result = training.load_data()

    assert len(result) == 2
    assert "Fare_amount" in result.columns


def test_prepare_data():
    df = pd.DataFrame({
        "Fare_amount": [10.0, 20.0, 30.0],
        "vendorid": [1, 2, 1],
        "distance": [2.5, 3.0, 4.0],
        "payment": ["Cash", "Card", "Cash"]
    })

    params = {"target": "Fare_amount"}

    X, y, numerical, categorical = training.prepare_data(
        df, params
    )

    assert "Fare_amount" not in X.columns
    assert len(y) == 3
    assert "distance" in numerical
    assert "vendorid" in categorical
    assert "payment" in categorical


def test_create_pipeline():
    from sklearn.ensemble import RandomForestRegressor

    X = pd.DataFrame({
        "distance": [2.5, 3.0, None, 4.0],
        "payment": ["Cash", "Card", "Cash", "Card"]
    })

    y = [10.0, 20.0, 15.0, 25.0]

    model = RandomForestRegressor(
        n_estimators=5,
        random_state=42
    )

    pipeline = training.create_pipeline(
        model,
        ["distance"],
        ["payment"]
    )

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert len(predictions) == 4
    assert pipeline.named_steps["model"] is model


def test_compare_models(tmp_path, monkeypatch):
    from sklearn.linear_model import LinearRegression

    df = pd.DataFrame({
        "distance": [1, 2, 3, 4, 5, 6],
        "payment": ["Cash", "Card", "Cash", "Card", "Cash", "Card"],
    })

    X = df
    y = pd.Series([10, 20, 30, 40, 50, 60])

    params = {
        "cv_folds": 3,
        "models": {
            "linear_regression": {
                "fit_intercept": True
            }
        }
    }

    monkeypatch.setattr(training, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        training,
        "log_training_results",
        lambda results: None
    )

    best_name, best_model = training.compare_models(
        X,
        y,
        ["distance"],
        ["payment"],
        params
    )

    report = tmp_path / "reports" / "model_comparison.csv"

    assert best_name == "Linear Regression"
    assert isinstance(best_model, LinearRegression)
    assert report.exists()

    results = pd.read_csv(report)

    assert "R2" in results.columns
    assert "MAE" in results.columns
    assert "RMSE" in results.columns


# =========================
# HYPERPARAMETER TUNING TESTS
# =========================

def test_load_selected_model_name(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    pd.DataFrame({
        "model": ["Linear Regression", "Random Forest"],
        "R2": [0.50, 0.70],
        "RMSE": [5.0, 4.0],
        "MAE": [3.0, 2.0]
    }).to_csv(reports_dir / "model_comparison.csv", index=False)

    monkeypatch.setattr(tuning, "PROJECT_ROOT", tmp_path)

    result = tuning.load_selected_model_name()

    assert result == "Random Forest"


def test_load_selected_model_name_missing_report(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "PROJECT_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError):
        tuning.load_selected_model_name()


def test_load_selected_model_name_empty_report(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    pd.DataFrame(
        columns=["model", "R2", "RMSE", "MAE"]
    ).to_csv(reports_dir / "model_comparison.csv", index=False)

    monkeypatch.setattr(tuning, "PROJECT_ROOT", tmp_path)

    with pytest.raises(ValueError, match="empty"):
        tuning.load_selected_model_name()


def test_get_selected_model():
    params = {
        "models": {
            "linear_regression": {"fit_intercept": True},
            "random_forest": {
                "n_estimators": 10,
                "max_depth": 4,
                "random_state": 42
            }
        }
    }

    linear_model = tuning.get_selected_model(
        "Linear Regression", params
    )

    forest_model = tuning.get_selected_model(
        "Random Forest", params
    )

    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

    assert isinstance(linear_model, LinearRegression)
    assert isinstance(forest_model, RandomForestRegressor)


def test_get_selected_model_invalid():
    with pytest.raises(ValueError, match="Unsupported model"):
        tuning.get_selected_model("Unknown Model", {"models": {}})


def test_get_parameter_grid():
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

    params = {
        "hyperparameter_tuning": {
            "linear_regression": {
                "fit_intercept": [True, False]
            },
            "random_forest": {
                "n_estimators": [5, 10],
                "max_depth": [2, 4]
            }
        }
    }

    linear_grid = tuning.get_parameter_grid(
        LinearRegression(), params
    )

    forest_grid = tuning.get_parameter_grid(
        RandomForestRegressor(), params
    )

    assert linear_grid == {
        "model__fit_intercept": [True, False]
    }

    assert forest_grid == {
        "model__n_estimators": [5, 10],
        "model__max_depth": [2, 4]
    }


def test_get_parameter_grid_invalid():
    with pytest.raises(ValueError, match="No tuning configuration"):
        tuning.get_parameter_grid(object(), {
            "hyperparameter_tuning": {}
        })


def test_tune_selected_model(tmp_path, monkeypatch):
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    X = pd.DataFrame({
        "distance": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "payment": [
            "Cash", "Card", "Cash",
            "Card", "Cash", "Card",
            "Cash", "Card", "Cash"
        ]
    })

    y = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90])

    pipeline = training.create_pipeline(
        LinearRegression(),
        ["distance"],
        ["payment"]
    )

    params = {"cv_folds": 3}

    monkeypatch.setattr(tuning, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        tuning,
        "log_tuning_results",
        lambda **kwargs: None
    )

    best_model = tuning.tune_selected_model(
        pipeline=pipeline,
        parameter_grid={
            "model__fit_intercept": [True, False]
        },
        X=X,
        y=y,
        params=params,
        model_name="Linear Regression"
    )

    assert isinstance(best_model, Pipeline)
    assert (tmp_path / "models" / "best_model.joblib").exists()
    assert (tmp_path / "reports" / "grid_search_results.csv").exists()
    assert (tmp_path / "reports" / "best_parameters.csv").exists()


# =========================
# MODEL EVALUATION TESTS
# =========================

def test_load_model(tmp_path, monkeypatch):
    import joblib
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model_path = tmp_path / "models" / "best_model.joblib"
    model_path.parent.mkdir(parents=True)

    joblib.dump(model, model_path)

    monkeypatch.setattr(evaluation, "PROJECT_ROOT", tmp_path)

    loaded_model = evaluation.load_model()

    assert isinstance(loaded_model, LinearRegression)


def test_load_model_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "PROJECT_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError):
        evaluation.load_model()


def test_load_test_data(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)

    pd.DataFrame({
        "distance": [1.0, 2.0],
        "Fare_amount": [10.0, 20.0]
    }).to_csv(data_dir / "test.csv", index=False)

    monkeypatch.setattr(evaluation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        evaluation,
        "load_params",
        lambda: {"target": "Fare_amount"}
    )

    df = evaluation.load_test_data()

    assert len(df) == 2
    assert "Fare_amount" in df.columns


def test_load_test_data_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        evaluation,
        "load_params",
        lambda: {"target": "Fare_amount"}
    )

    with pytest.raises(FileNotFoundError):
        evaluation.load_test_data()


def test_load_test_data_missing_target(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)

    pd.DataFrame({
        "distance": [1.0, 2.0]
    }).to_csv(data_dir / "test.csv", index=False)

    monkeypatch.setattr(evaluation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        evaluation,
        "load_params",
        lambda: {"target": "Fare_amount"}
    )

    with pytest.raises(ValueError, match="Target column"):
        evaluation.load_test_data()


def test_evaluate_model(monkeypatch):
    from sklearn.dummy import DummyRegressor

    df = pd.DataFrame({
        "distance": [1.0, 2.0, 3.0, 4.0],
        "Fare_amount": [10.0, 20.0, 30.0, 40.0]
    })

    model = DummyRegressor(strategy="mean")
    model.fit(
        df[["distance"]],
        df["Fare_amount"]
    )

    monkeypatch.setattr(
        evaluation,
        "load_params",
        lambda: {"target": "Fare_amount"}
    )

    results = evaluation.evaluate_model(model, df)

    assert list(results.columns) == ["MAE", "RMSE", "R2"]
    assert len(results) == 1
    assert results["MAE"].iloc[0] >= 0
    assert results["RMSE"].iloc[0] >= 0
    assert isinstance(results["R2"].iloc[0], float)


# =========================
# PREDICTION TESTS
# =========================

def test_prediction_load_model(tmp_path, monkeypatch):
    import joblib
    from sklearn.dummy import DummyRegressor

    model = DummyRegressor(strategy="constant", constant=25.0)
    model.fit([[1], [2]], [25.0, 25.0])

    model_path = tmp_path / "models" / "best_model.joblib"
    model_path.parent.mkdir(parents=True)

    joblib.dump(model, model_path)

    monkeypatch.setattr(prediction, "PROJECT_ROOT", tmp_path)

    loaded_model = prediction.load_model()

    assert isinstance(loaded_model, DummyRegressor)


def test_prediction_load_model_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "PROJECT_ROOT", tmp_path)

    with pytest.raises(FileNotFoundError):
        prediction.load_model()


def test_prepare_input():
    data = {
        "vendorid": 2,
        "rate_code": 1,
        "Trip_type": 1,
        "Trip_distance": 1.2,
        "pickup_hour": 14
    }

    result = prediction.prepare_input(data)

    assert len(result) == 1
    assert result.loc[0, "vendorid"] == 2
    assert result["vendorid"].dtype == "object"
    assert result["rate_code"].dtype == "object"
    assert result["Trip_type"].dtype == "object"


def test_predict_fare():
    from sklearn.dummy import DummyRegressor

    model = DummyRegressor(strategy="constant", constant=25.0)
    model.fit([[1], [2]], [25.0, 25.0])

    data = {"distance": 1.5}

    result = prediction.predict_fare(model, data)

    assert isinstance(result, float)
    assert result == 25.0