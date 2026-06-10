"""
Training pipeline — run once to produce all model artifacts in models/.
Usage: python scripts/train_pipeline.py
"""
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.data.loader import load_raw_data
from src.data.preprocessor import full_pipeline
from src.models.trainer import train_all
from src.models.evaluator import evaluate_all
from src.models.clustering import (
    compute_elbow_data, compute_silhouette_scores,
    get_optimal_k, run_kmeans,
)
from src.explainability.shap_explainer import get_explainer, compute_shap_values
from src.utils.io_helpers import save_model, save_json, save_numpy

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "heart.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    print("=" * 60)
    print("Heart Disease Risk Prediction — Training Pipeline")
    print("=" * 60)

    # ── 1. Load & Preprocess ──────────────────────────────────────
    print("\n[1/5] Loading and preprocessing data...")
    df = load_raw_data(DATASET_PATH)
    print(f"  Dataset: {df.shape[0]} rows × {df.shape[1]} cols")

    X_train, X_test, y_train, y_test, scaler, feature_columns, chol_median = full_pipeline(df)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Cholesterol zero median: {chol_median:.1f}")
    print(f"  Features: {len(feature_columns)}")

    save_model(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    save_json(
        {"feature_columns": feature_columns, "chol_median": chol_median},
        os.path.join(MODELS_DIR, "feature_columns.json"),
    )
    print("  Saved: scaler.joblib, feature_columns.json")

    # ── 2. Train Classifiers ──────────────────────────────────────
    print("\n[2/5] Training classifiers (GridSearchCV)...")
    models = train_all(X_train, y_train)
    for name, model in models.items():
        save_model(model, os.path.join(MODELS_DIR, f"{name}.joblib"))
    print("  Saved: logistic_regression.joblib, random_forest.joblib, xgboost.joblib")

    # ── 3. Evaluate ───────────────────────────────────────────────
    print("\n[3/5] Evaluating models...")
    results = evaluate_all(models, X_test, y_test)
    for name, metrics in results.items():
        print(f"  {name:25s}  F1={metrics['f1']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  Acc={metrics['accuracy']:.4f}")
    save_json(results, os.path.join(MODELS_DIR, "eval_results.json"))
    save_numpy(X_test, os.path.join(MODELS_DIR, "X_test.npy"))
    save_numpy(y_test, os.path.join(MODELS_DIR, "y_test.npy"))
    save_numpy(X_train, os.path.join(MODELS_DIR, "X_train.npy"))
    print("  Saved: eval_results.json, X_test.npy, y_test.npy, X_train.npy")

    # ── 4. Clustering ─────────────────────────────────────────────
    print("\n[4/5] Running clustering analysis...")
    X_all_scaled = np.vstack([X_train, X_test])
    elbow = compute_elbow_data(X_all_scaled)
    sil = compute_silhouette_scores(X_all_scaled)
    optimal_k = get_optimal_k(sil)
    print(f"  Optimal k (silhouette): {optimal_k}")
    km_model, km_labels = run_kmeans(X_all_scaled, optimal_k)
    save_model(km_model, os.path.join(MODELS_DIR, "kmeans.joblib"))
    save_json(
        {"elbow": elbow, "silhouette": sil, "optimal_k": optimal_k},
        os.path.join(MODELS_DIR, "clustering_data.json"),
    )
    save_numpy(km_labels, os.path.join(MODELS_DIR, "kmeans_labels.npy"))
    save_numpy(X_all_scaled, os.path.join(MODELS_DIR, "X_all_scaled.npy"))
    print("  Saved: kmeans.joblib, clustering_data.json, kmeans_labels.npy")

    # ── 5. SHAP ───────────────────────────────────────────────────
    print("\n[5/5] Computing SHAP values...")
    xgb_model = models["xgboost"]
    explainer = get_explainer(xgb_model, X_train[:100])
    shap_vals = compute_shap_values(explainer, X_test)
    save_numpy(shap_vals, os.path.join(MODELS_DIR, "shap_values_xgboost.npy"))
    print(f"  SHAP values shape: {shap_vals.shape}")
    print("  Saved: shap_values_xgboost.npy")

    print("\n" + "=" * 60)
    print("Training complete. All artifacts saved to models/")
    print("Run: streamlit run app/main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
