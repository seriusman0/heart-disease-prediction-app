import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_explainer(model, X_train_sample: np.ndarray):
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, X_train_sample)
    else:
        return shap.TreeExplainer(model)


def compute_shap_values(explainer, X: np.ndarray) -> np.ndarray:
    vals = explainer.shap_values(X)
    # For tree models with binary classification, shap_values returns list [neg_class, pos_class]
    if isinstance(vals, list):
        return vals[1]
    return vals


def get_summary_data(shap_values: np.ndarray, feature_names: list) -> dict:
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    return {
        "features": [feature_names[i] for i in sorted_idx],
        "mean_abs_shap": mean_abs[sorted_idx].tolist(),
    }


def get_single_prediction_shap(explainer, row: np.ndarray, feature_names: list) -> dict:
    if row.ndim == 1:
        row = row.reshape(1, -1)
    vals = explainer.shap_values(row)
    if isinstance(vals, list):
        vals = vals[1]
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1])
    return {
        "base_value": float(base_value),
        "shap_values": vals[0].tolist(),
        "feature_names": feature_names,
        "feature_values": row[0].tolist(),
    }
