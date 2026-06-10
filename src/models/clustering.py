import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def compute_elbow_data(X_scaled, k_range=range(2, 11)) -> dict:
    inertias = []
    ks = list(k_range)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return {"k": ks, "inertia": inertias}


def compute_silhouette_scores(X_scaled, k_range=range(2, 11)) -> dict:
    scores = []
    ks = list(k_range)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores.append(float(silhouette_score(X_scaled, labels)))
    return {"k": ks, "score": scores}


def get_optimal_k(silhouette_dict: dict) -> int:
    idx = int(np.argmax(silhouette_dict["score"]))
    return silhouette_dict["k"][idx]


def run_kmeans(X_scaled, n_clusters: int):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def get_cluster_profiles(df_original: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    df = df_original.copy()
    df["Cluster"] = labels
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df.groupby("Cluster")[numeric_cols].mean().round(2)


def get_pca_projection(X_scaled) -> np.ndarray:
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X_scaled)
