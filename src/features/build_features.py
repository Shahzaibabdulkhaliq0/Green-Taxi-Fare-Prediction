
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params():
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)["feature_engineering"]


def load_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    print("Loaded:", file_path, df.shape)
    return df


def create_features(df, params):
    df = df.copy()
    target = params["target"]

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"],
        format=params["pickup_datetime_format"],
        errors="coerce",
    )

    df = df.dropna(subset=["pickup_datetime"])
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["pickup_datetime"].dt.day_of_week
    df["pickup_month"] = df["pickup_datetime"].dt.month

    drop_columns = [
        column
        for column in params["drop_columns"]
        if column != target
    ]

    df = df.drop(columns=drop_columns, errors="ignore")
    df = df.drop(columns=["pickup_datetime"], errors="ignore")

    return df


def save_data(train, test, output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path / "train.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)

    print("Processed train:", train.shape)
    print("Processed test:", test.shape)
    print("Saved files to:", output_path)


def main():
    params = load_params()
    input_path = PROJECT_ROOT / params["input_dir"]
    output_path = PROJECT_ROOT / params["output_dir"]

    train = load_data(input_path / "train.csv")
    test = load_data(input_path / "test.csv")

    train = create_features(train, params)
    test = create_features(test, params)

    save_data(train, test, output_path)


if __name__ == "__main__":
    main()