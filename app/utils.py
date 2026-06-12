"""Shared utilities for all Streamlit pages."""
import os
import sys

# Ensure app/ is in sys.path so src.* resolves to app/src/
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(APP_DIR, "..")
for _p in (APP_DIR, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
from src.utils.io_helpers import load_model, load_json, load_numpy

MODELS_DIR = os.path.join(ROOT, "model")
DATASET_PATH = os.path.join(ROOT, "dataset", "heart.csv")

PLOTLY_TEMPLATE = "plotly_white"
COLOR_POSITIVE = "#E84545"   # red — heart disease
COLOR_NEGATIVE = "#2B2D42"   # dark — no heart disease
COLOR_PALETTE = [COLOR_NEGATIVE, COLOR_POSITIVE]


def models_exist() -> bool:
    required = ["scaler.joblib", "feature_columns.json",
                "logistic_regression.joblib", "random_forest.joblib",
                "xgboost.joblib", "kmeans.joblib"]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


@st.cache_resource
def load_all_models() -> dict:
    meta = load_json(os.path.join(MODELS_DIR, "feature_columns.json"))
    return {
        "lr": load_model(os.path.join(MODELS_DIR, "logistic_regression.joblib")),
        "rf": load_model(os.path.join(MODELS_DIR, "random_forest.joblib")),
        "xgb": load_model(os.path.join(MODELS_DIR, "xgboost.joblib")),
        "kmeans": load_model(os.path.join(MODELS_DIR, "kmeans.joblib")),
        "scaler": load_model(os.path.join(MODELS_DIR, "scaler.joblib")),
        "feature_columns": meta["feature_columns"],
        "chol_median": meta["chol_median"],
    }


@st.cache_data
def load_eval_results() -> dict:
    return load_json(os.path.join(MODELS_DIR, "eval_results.json"))


@st.cache_data
def load_test_data():
    X_test = load_numpy(os.path.join(MODELS_DIR, "X_test.npy"))
    y_test = load_numpy(os.path.join(MODELS_DIR, "y_test.npy"))
    return X_test, y_test


@st.cache_data
def load_clustering_data() -> dict:
    return load_json(os.path.join(MODELS_DIR, "clustering_data.json"))


@st.cache_data
def load_shap_values():
    path = os.path.join(MODELS_DIR, "shap_values_xgboost.npy")
    if os.path.exists(path):
        return load_numpy(path)
    return None


@st.cache_data
def load_raw_df():
    import pandas as pd
    return pd.read_csv(DATASET_PATH)


MODEL_DISPLAY_NAMES = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}

MODEL_KEYS = {
    "Logistic Regression": "lr",
    "Random Forest": "rf",
    "XGBoost": "xgb",
}
