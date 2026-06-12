import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.utils import (
    load_all_models, load_clustering_data, load_raw_df,
    PLOTLY_TEMPLATE, COLOR_POSITIVE, COLOR_NEGATIVE, models_exist,
)
from src.utils.io_helpers import load_numpy
from src.models.clustering import get_pca_projection, get_cluster_profiles

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../.. /model")

st.set_page_config(page_title="Clustering | Heart Disease", page_icon="🔵", layout="wide")
st.title("🔵 Analisis Clustering K-Means")

if not models_exist():
    st.error("Model artifacts tidak ditemukan. Jalankan `python scripts/train_pipeline.py`.")
    st.stop()

clustering_data = load_clustering_data()
optimal_k = clustering_data["optimal_k"]

st.info(f"Optimal k (berdasarkan Silhouette Score): **{optimal_k}**")

# ── Elbow & Silhouette ───────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    elbow = clustering_data["elbow"]
    fig_elbow = px.line(
        x=elbow["k"], y=elbow["inertia"],
        markers=True, template=PLOTLY_TEMPLATE,
        title="Elbow Curve (Inertia vs k)",
        labels={"x": "k (jumlah cluster)", "y": "Inertia"},
    )
    fig_elbow.update_traces(line_color=COLOR_POSITIVE)
    st.plotly_chart(fig_elbow, use_container_width=True)

with col2:
    sil = clustering_data["silhouette"]
    sil_df = pd.DataFrame(sil)
    fig_sil = px.line(
        sil_df, x="k", y="score",
        markers=True, template=PLOTLY_TEMPLATE,
        title="Silhouette Score vs k",
        labels={"k": "k (jumlah cluster)", "score": "Silhouette Score"},
    )
    fig_sil.update_traces(line_color="#00CC96")
    # Highlight optimal k
    opt_score = sil_df.loc[sil_df["k"] == optimal_k, "score"].values[0]
    fig_sil.add_scatter(
        x=[optimal_k], y=[opt_score],
        mode="markers", marker=dict(size=14, color="red", symbol="star"),
        name=f"Optimal k={optimal_k}",
    )
    st.plotly_chart(fig_sil, use_container_width=True)

st.markdown("---")

# ── PCA Scatter ──────────────────────────────────────────────────
st.subheader("🗺️ Visualisasi Cluster (PCA 2D)")

try:
    X_all = load_numpy(os.path.join(MODELS_DIR, "X_all_scaled.npy"))
    labels = load_numpy(os.path.join(MODELS_DIR, "kmeans_labels.npy"))
    pca_coords = get_pca_projection(X_all)

    df_pca = pd.DataFrame({
        "PC1": pca_coords[:, 0],
        "PC2": pca_coords[:, 1],
        "Cluster": labels.astype(str),
    })

    fig_pca = px.scatter(
        df_pca, x="PC1", y="PC2", color="Cluster",
        template=PLOTLY_TEMPLATE,
        title=f"Distribusi Cluster (k={optimal_k}) — PCA 2D",
        opacity=0.75,
    )
    st.plotly_chart(fig_pca, use_container_width=True)
except Exception as e:
    st.warning(f"Tidak dapat memuat data PCA: {e}")

st.markdown("---")

# ── Cluster Profiles ─────────────────────────────────────────────
st.subheader("📋 Profil Cluster")
df_raw = load_raw_df()

try:
    # Only use numeric subset of raw data + labels
    chol_median_val = df_raw.loc[df_raw["Cholesterol"] != 0, "Cholesterol"].median()
    df_num = df_raw.copy()
    df_num.loc[df_num["Cholesterol"] == 0, "Cholesterol"] = chol_median_val
    numeric_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "HeartDisease"]
    df_num_subset = df_num[numeric_cols].copy()

    n_rows = len(df_num_subset)
    n_labels = len(labels)
    labels_subset = labels[:n_rows] if n_labels >= n_rows else labels

    profiles = get_cluster_profiles(df_num_subset, labels_subset)
    st.dataframe(profiles, use_container_width=True)

    # HeartDisease prevalence per cluster
    df_num_subset_copy = df_num_subset.copy()
    df_num_subset_copy["Cluster"] = labels_subset
    prevalence = df_num_subset_copy.groupby("Cluster")["HeartDisease"].mean().reset_index()
    prevalence.columns = ["Cluster", "HeartDisease Prevalence"]
    prevalence["Cluster"] = prevalence["Cluster"].astype(str)

    fig_prev = px.bar(
        prevalence, x="Cluster", y="HeartDisease Prevalence",
        color="HeartDisease Prevalence",
        color_continuous_scale="Reds",
        template=PLOTLY_TEMPLATE,
        title="Prevalensi Penyakit Jantung per Cluster",
        text_auto=".2%",
    )
    st.plotly_chart(fig_prev, use_container_width=True)
except Exception as e:
    st.warning(f"Tidak dapat memuat profil cluster: {e}")
