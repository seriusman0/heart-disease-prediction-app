import pandas as pd
import os

EXPECTED_COLUMNS = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak",
    "ST_Slope", "HeartDisease"
]


def load_raw_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def get_feature_target_split(df: pd.DataFrame):
    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]
    return X, y


def describe_dataset(df: pd.DataFrame) -> dict:
    n_pos = int((df["HeartDisease"] == 1).sum())
    n_neg = int((df["HeartDisease"] == 0).sum())
    zero_chol = int((df["Cholesterol"] == 0).sum())
    return {
        "shape": df.shape,
        "n_rows": len(df),
        "n_features": len(df.columns) - 1,
        "null_counts": df.isnull().sum().to_dict(),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "class_balance": {"Heart Disease": n_pos, "No Heart Disease": n_neg},
        "zero_cholesterol": zero_chol,
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
