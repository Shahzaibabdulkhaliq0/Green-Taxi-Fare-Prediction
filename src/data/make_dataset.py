
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params():
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)["data_preparation"]


def load_data(params):
    raw_path = PROJECT_ROOT / params["raw_data_path"]
    df = pd.read_csv(raw_path, low_memory=False)
    print("Dataset loaded:", df.shape)
    return df


def clean_data(df, target):
    df.columns = df.columns.str.strip()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    df = df[df[target] > 0]
    df = df.drop_duplicates()

    print("Dataset cleaned:", df.shape)
    return df

def save_data(train, test, params):
    output_dir = PROJECT_ROOT / params["interim_data_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    print("Train data:", train.shape)
    print("Test data:", test.shape)
    print("Files saved successfully")


def main():
    params = load_params()

    df = load_data(params)
    df = clean_data(df, params["target"])

    # ===== TEMPORARY TESTING MODE =====
    # Testing ke liye uncomment karo:
    # df = df.sample(n=20000, random_state=42)  = Isko Comment & Uncomment krna hai
    #  Is se Hum Data ki Just 20000 Rows Use kr sake ge
    # Full dataset ke liye upar wali line comment rakho.
    # ==================================

    train, test = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )

    save_data(train, test, params)


if __name__ == "__main__":
    main()