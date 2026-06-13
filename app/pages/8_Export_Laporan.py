"""
Halaman khusus export semua gambar untuk laporan.
Semua Gbr. 1-9 ditampilkan berurutan, bersih, siap screenshot.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils import (
    load_all_models, load_eval_results, load_test_data,
    load_raw_df, load_clustering_data,
    PLOTLY_TEMPLATE, COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_PALETTE, models_exist,
)
from src.utils.io_helpers import load_numpy
from src.models.evaluator import get_feature_importances
from src.models.clustering import get_pca_projection

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "model")

st.set_page_config(page_title="Export Gambar Laporan", page_icon=None, layout="wide")
st.title("Export Gambar untuk Laporan")
st.markdown("Semua gambar (Gbr. 1–9) tersaji berurutan. Screenshot tiap bagian untuk melengkapi laporan.")

if not models_exist():
    st.error("Model artifacts tidak ditemukan.")
    st.stop()

# ── Load semua data ───────────────────────────────────────────────
artifacts     = load_all_models()
results       = load_eval_results()
X_test, y_test = load_test_data()
df_raw        = load_raw_df()
clustering_data = load_clustering_data()
optimal_k     = clustering_data["optimal_k"]
feature_columns = artifacts["feature_columns"]

MODEL_KEYS_LIST = ["logistic_regression", "random_forest", "xgboost"]
MODEL_NAMES     = ["Logistic Regression", "Random Forest", "XGBoost"]

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# GBR. 1 — Overview Dataset
# ════════════════════════════════════════════════════════════════
st.subheader("Gbr. 1 — Overview Dataset")

col_l, col_r = st.columns([1, 2])
with col_l:
    info_rows = [
        ("Jumlah Observasi", "918"),
        ("Jumlah Fitur Prediktor", "11"),
        ("Label Target", "HeartDisease (0/1)"),
        ("Sumber", "Cleveland, Hungarian, Switzerland, Long Beach VA, Stalog"),
        ("Missing Values", "0 (kolesterol 0 diimputasi)"),
    ]
    st.table(pd.DataFrame(info_rows, columns=["Atribut", "Nilai"]))

with col_r:
    st.dataframe(df_raw.head(10), use_container_width=True)

st.caption(
    "Gbr. 1 — Cuplikan dataset rekam medis: 918 observasi dan 11 fitur klinis "
    "setelah proses pembersihan awal."
)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# GBR. 2 — Distribusi Kelas Target
# ════════════════════════════════════════════════════════════════
st.subheader("Gbr. 2 — Distribusi Kelas Target (HeartDisease)")

counts = df_raw["HeartDisease"].value_counts().sort_index()
label_map = {0: "Normal (0)", 1: "Terindikasi Penyakit (1)"}
df_counts = pd.DataFrame({
    "Kelas": [label_map[k] for k in counts.index],
    "Jumlah Pasien": counts.values,
})

fig2 = px.bar(
    df_counts, x="Kelas", y="Jumlah Pasien",
    color="Kelas",
    color_discrete_map={
        "Normal (0)": COLOR_NEGATIVE,
        "Terindikasi Penyakit (1)": COLOR_POSITIVE,
    },
    template=PLOTLY_TEMPLATE,
    title="Distribusi Kelas Target — HeartDisease",
    text="Jumlah Pasien",
)
fig2.update_traces(textposition="outside", textfont_size=14)
fig2.update_layout(showlegend=False, yaxis=dict(range=[0, 600]))
st.plotly_chart(fig2, use_container_width=True)
st.caption(
    "Gbr. 2 — Grafik Bar Chart jumlah pasien berdasarkan kelas target: "
    f"{int(counts[1])} terindikasi penyakit (55,3%) dan {int(counts[0])} normal (44,7%)."
)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# GBR. 3 — Matriks Korelasi Pearson
# ════════════════════════════════════════════════════════════════
st.subheader("Gbr. 3 — Matriks Korelasi Pearson")

num_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak", "HeartDisease"]
corr = df_raw[num_cols].corr(method="pearson").round(2)

fig3 = px.imshow(
    corr, text_auto=True, template=PLOTLY_TEMPLATE,
    color_continuous_scale="RdBu_r",
    title="Matriks Korelasi Pearson — Fitur Medis vs Target (HeartDisease)",
    zmin=-1, zmax=1,
)
fig3.update_layout(
    coloraxis_colorbar=dict(title="Pearson r"),
    height=500,
)
st.plotly_chart(fig3, use_container_width=True)
st.caption(
    "Gbr. 3 — Matriks Korelasi Pearson antar fitur medis dengan variabel target "
    "penyakit jantung. Warna merah = korelasi positif, biru = korelasi negatif."
)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# GBR. 4 — Feature Importance XGBoost
# ════════════════════════════════════════════════════════════════
st.subheader("Gbr. 4 — Grafik Feature Importance")

fi_df = get_feature_importances(artifacts["xgb"], feature_columns).head(15)
fig4 = px.bar(
    fi_df, x="importance", y="feature", orientation="h",
    template=PLOTLY_TEMPLATE,
    title="Feature Importance — XGBoost (Top 15 Fitur Medis)",
    color="importance",
    color_continuous_scale="Reds",
)
fig4.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=480)
st.plotly_chart(fig4, use_container_width=True)
st.caption(
    "Gbr. 4 — Grafik Feature Importance yang memeringkatkan kontribusi masing-masing "
    "atribut medis hasil ekstraksi seleksi fitur (model XGBoost)."
)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# GBR. 5 — Elbow Curve + Silhouette Score
# ════════════════════════════════════════════════════════════════
st.subheader("Gbr. 5 — Kurva Elbow dan Grafik Silhouette Score")

col5a, col5b = st.columns(2)

with col5a:
    elbow = clustering_data["elbow"]
    fig5a = px.line(
        x=elbow["k"], y=elbow["inertia"],
        markers=True, template=PLOTLY_TEMPLATE,
        title="Elbow Curve — Inertia vs k",
        labels={"x": "k (Jumlah Cluster)", "y": "Inertia"},
    )
    fig5a.update_traces(line_color=COLOR_POSITIVE, line_width=2, marker_size=8)
    fig5a.update_layout(height=360)
    st.plotly_chart(fig5a, use_container_width=True)

with col5b:
    sil = clustering_data["silhouette"]
    sil_df = pd.DataFrame(sil)
    opt_score = sil_df.loc[sil_df["k"] == optimal_k, "score"].values[0]
    fig5b = px.line(
        sil_df, x="k", y="score",
        markers=True, template=PLOTLY_TEMPLATE,
        title="Silhouette Score vs k",
        labels={"k": "k (Jumlah Cluster)", "score": "Silhouette Score"},
    )
    fig5b.update_traces(line_color="#00CC96", line_width=2, marker_size=8)
    fig5b.add_scatter(
        x=[optimal_k], y=[opt_score],
        mode="markers",
        marker=dict(size=16, color="red", symbol="star"),
        name=f"Optimal k={optimal_k}",
    )
    fig5b.update_layout(height=360)
    st.plotly_chart(fig5b, use_container_width=True)

st.caption(
    f"Gbr. 5 — Kurva Elbow (kiri) menunjukkan penurunan inersia melandai pada K=3. "
    f"Grafik Silhouette Score (kanan) mencapai puncak pada K={optimal_k}, "
    "memvalidasi jumlah klaster optimal."
)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# GBR. 6 — PCA 2D Scatter
# ════════════════════════════════════════════════════════════════
st.subheader("Gbr. 6 — Plot Sebaran PCA 2 Dimensi")

CLUSTER_LABELS = {
    "0": "Klaster 0 — Risiko Rendah",
    "1": "Klaster 1 — Risiko Moderat",
    "2": "Klaster 2 — Risiko Kritis",
}

try:
    X_all   = load_numpy(os.path.join(MODELS_DIR, "X_all_scaled.npy"))
    km_lbls = load_numpy(os.path.join(MODELS_DIR, "kmeans_labels.npy"))
    pca_coords = get_pca_projection(X_all)

    df_pca = pd.DataFrame({
        "PC1": pca_coords[:, 0],
        "PC2": pca_coords[:, 1],
        "Profil Risiko": [CLUSTER_LABELS.get(str(l), str(l)) for l in km_lbls],
    })

    fig6 = px.scatter(
        df_pca, x="PC1", y="PC2", color="Profil Risiko",
        color_discrete_map={
            "Klaster 0 — Risiko Rendah":  "#2B2D42",
            "Klaster 1 — Risiko Moderat": "#F4A261",
            "Klaster 2 — Risiko Kritis":  "#E84545",
        },
        template=PLOTLY_TEMPLATE,
        title=f"Sebaran Pasien — PCA 2D (K-Means, k={optimal_k})",
        opacity=0.72,
    )
    fig6.update_layout(
        legend_title="Profil Risiko",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig6, use_container_width=True)
    st.caption(
        "Gbr. 6 — Plot Sebaran PCA 2 Dimensi yang memvisualisasikan batas pemisahan "
        "tiga profil risiko pasien hasil K-Means: Risiko Rendah (hitam), "
        "Risiko Moderat (oranye), Risiko Kritis (merah)."
    )
except Exception as e:
    st.error(f"PCA error: {e}")

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# GBR. 7 — Komparasi Metrik Kinerja
# ════════════════════════════════════════════════════════════════
st.subheader("Gbr. 7 — Komparasi Metrik Kinerja Semua Model")

metrics_map = {
    "Akurasi":  "accuracy",
    "Presisi":  "precision",
    "Recall":   "recall",
    "F1-Score": "f1",
}
bar_colors7 = ["#636EFA", "#00CC96", "#EF553B", "#AB63FA"]

fig7 = go.Figure()
for (label, key), color in zip(metrics_map.items(), bar_colors7):
    vals = [results[m][key] for m in MODEL_KEYS_LIST]
    fig7.add_trace(go.Bar(
        name=label,
        x=MODEL_NAMES,
        y=vals,
        marker_color=color,
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
        textfont_size=11,
    ))

fig7.update_layout(
    barmode="group",
    template=PLOTLY_TEMPLATE,
    title="Komparasi Metrik Kinerja — Logistic Regression vs Random Forest vs XGBoost",
    yaxis=dict(range=[0.7, 1.05], title="Score"),
    legend_title="Metrik",
    height=460,
)
st.plotly_chart(fig7, use_container_width=True)
st.caption(
    "Gbr. 7 — Diagram Batang komparasi metrik kinerja (Akurasi, Presisi, Recall, F1-Score) "
    "antara Logistic Regression, Random Forest, dan XGBoost."
)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# GBR. 8 — Confusion Matrix XGBoost
# ════════════════════════════════════════════════════════════════
st.subheader("Gbr. 8 — Confusion Matrix XGBoost")

xr   = results["xgboost"]
cm8  = np.array(xr["confusion_matrix"])

col8a, col8b = st.columns([2, 1])
with col8a:
    fig8 = px.imshow(
        cm8,
        labels=dict(x="Prediksi", y="Aktual"),
        x=["No Disease", "Heart Disease"],
        y=["No Disease", "Heart Disease"],
        text_auto=True,
        color_continuous_scale="Blues",
        template=PLOTLY_TEMPLATE,
        title="Confusion Matrix — XGBoost",
    )
    fig8.update_coloraxes(showscale=False)
    fig8.update_layout(height=420, font=dict(size=14))
    fig8.update_traces(textfont_size=20)
    st.plotly_chart(fig8, use_container_width=True)

with col8b:
    st.markdown("**Metrik XGBoost**")
    st.metric("Accuracy",  f"{xr['accuracy']:.4f}")
    st.metric("F1-Score",  f"{xr['f1']:.4f}")
    st.metric("Precision", f"{xr['precision']:.4f}")
    st.metric("Recall",    f"{xr['recall']:.4f}")
    st.metric("ROC-AUC",   f"{xr['roc_auc']:.4f}")
    st.markdown("---")
    st.markdown(f"**TN** = {int(cm8[0,0])}  &nbsp; **FP** = {int(cm8[0,1])}")
    st.markdown(f"**FN** = {int(cm8[1,0])}  &nbsp; **TP** = {int(cm8[1,1])}")

st.caption(
    "Gbr. 8 — Visualisasi Confusion Matrix model XGBoost yang merincikan "
    "proporsi klasifikasi prediksi benar (TN, TP) dan salah (FP, FN)."
)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# GBR. 9 — SHAP Summary Plot
# ════════════════════════════════════════════════════════════════
st.subheader("Gbr. 9 — SHAP Summary Plot")

from src.explainability.shap_explainer import get_summary_data

@st.cache_data
def _load_shap():
    path = os.path.join(MODELS_DIR, "shap_values_xgboost.npy")
    if os.path.exists(path):
        return load_numpy(path)
    return None

shap_vals = _load_shap()

if shap_vals is not None:
    summary = get_summary_data(shap_vals, feature_columns)
    summary_df = pd.DataFrame({
        "Fitur": summary["features"][:15],
        "Mean |SHAP|": summary["mean_abs_shap"][:15],
    })

    fig9 = px.bar(
        summary_df, x="Mean |SHAP|", y="Fitur", orientation="h",
        color="Mean |SHAP|",
        color_continuous_scale="Reds",
        template=PLOTLY_TEMPLATE,
        title="SHAP Summary Plot — Kontribusi Fitur terhadap Prediksi XGBoost (Top 15)",
    )
    fig9.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        height=480,
    )
    st.plotly_chart(fig9, use_container_width=True)
    st.caption(
        "Gbr. 9 — Grafik SHAP Summary Plot yang menginterpretasikan besaran dan arah "
        "kontribusi marjinal setiap fitur medis terhadap prediksi model XGBoost. "
        "ST_Slope Flat dan ChestPainType ASY memiliki kontribusi positif terbesar."
    )

    # Tambahan: beeswarm via matplotlib untuk laporan
    st.markdown("**Versi Beeswarm (untuk laporan):**")
    try:
        import shap, matplotlib.pyplot as plt
        from src.explainability.shap_explainer import get_explainer

        X_train = load_numpy(os.path.join(MODELS_DIR, "X_train.npy"))
        explainer = get_explainer(artifacts["xgb"], X_train[:100])
        base_val  = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[1])

        explanation = shap.Explanation(
            values=shap_vals,
            base_values=np.full(len(shap_vals), base_val),
            data=X_test,
            feature_names=feature_columns,
        )
        fig_bee, ax = plt.subplots(figsize=(10, 6))
        shap.plots.beeswarm(explanation, max_display=15, show=False)
        plt.title("SHAP Summary Plot (Beeswarm) — XGBoost", fontsize=12, fontweight="bold")
        st.pyplot(fig_bee)
        plt.close()
    except Exception as e:
        st.info(f"Beeswarm tidak tersedia: {e}")
else:
    st.warning("SHAP values tidak ditemukan. Jalankan ulang training pipeline.")

st.markdown("---")
st.success("Semua gambar (Gbr. 1-9) telah tersaji. Screenshot tiap section untuk laporan.")
