import json
import os
import joblib
import numpy as np


def save_model(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON artifact not found: {path}")
    with open(path) as f:
        return json.load(f)


def save_numpy(arr: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def load_numpy(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Numpy artifact not found: {path}")
    return np.load(path, allow_pickle=True)
