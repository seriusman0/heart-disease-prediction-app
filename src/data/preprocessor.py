import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
NUMERIC_COLS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]


def fix_cholesterol_zeros(df: pd.DataFrame, median: float = None) -> tuple:
    df = df.copy()
    if median is None:
        median = df.loc[df["Cholesterol"] != 0, "Cholesterol"].median()
    df.loc[df["Cholesterol"] == 0, "Cholesterol"] = median
    return df, median


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)


def scale_numerics(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    num_cols = [c for c in NUMERIC_COLS if c in X_train.columns]
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    return X_train_scaled, X_test_scaled, scaler


def full_pipeline(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    from src.data.loader import get_feature_target_split

    X, y = get_feature_target_split(df)

    # Train-test split before any fitting
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fix cholesterol zeros (fit median on train only)
    X_train_fixed, chol_median = fix_cholesterol_zeros(X_train_raw)
    X_test_fixed, _ = fix_cholesterol_zeros(X_test_raw, median=chol_median)

    # Combine for encoding to ensure same columns
    combined = pd.concat([X_train_fixed, X_test_fixed], axis=0)
    combined_encoded = encode_categoricals(combined)
    X_train_enc = combined_encoded.iloc[:len(X_train_fixed)].reset_index(drop=True)
    X_test_enc = combined_encoded.iloc[len(X_train_fixed):].reset_index(drop=True)

    feature_columns = list(X_train_enc.columns)

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_numerics(X_train_enc, X_test_enc)

    return (
        X_train_scaled.values,
        X_test_scaled.values,
        y_train.reset_index(drop=True).values,
        y_test.reset_index(drop=True).values,
        scaler,
        feature_columns,
        float(chol_median),
    )


def transform_single_row(
    input_dict: dict,
    scaler: StandardScaler,
    feature_columns: list,
    chol_median: float,
) -> np.ndarray:
    """Convert user form input to model-ready array."""
    # Build a single-row DataFrame
    row = pd.DataFrame([input_dict])

    # Fix cholesterol zero
    if row["Cholesterol"].iloc[0] == 0:
        row["Cholesterol"] = chol_median

    # Encode
    row_encoded = encode_categoricals(row)

    # Align to training feature columns (fill missing dummies with 0)
    aligned = pd.DataFrame(0, index=[0], columns=feature_columns)
    for col in row_encoded.columns:
        if col in aligned.columns:
            aligned[col] = row_encoded[col].values

    # Scale numeric cols
    num_cols = [c for c in NUMERIC_COLS if c in aligned.columns]
    aligned[num_cols] = scaler.transform(aligned[num_cols].values.reshape(1, -1).repeat(1, axis=0))

    return aligned.values
