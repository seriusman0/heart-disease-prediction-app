import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.utils import models_exist

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("❤️ Heart Disease\nRisk Prediction")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**Proyek UAS Data Mining**
Metode: Classification + Clustering
Framework: CRISP-DM
Dataset: Heart Failure Prediction (Kaggle)
"""
)

if not models_exist():
    st.warning(
        "⚠️ Model artifacts not found. "
        "Run `python scripts/train_pipeline.py` first.",
        icon="⚠️",
    )

st.title("❤️ Prediksi Risiko Penyakit Jantung")
st.markdown(
    """
Selamat datang di aplikasi **Prediksi dan Segmentasi Risiko Penyakit Jantung**
menggunakan teknik Data Mining.

Gunakan navigasi di sidebar untuk menjelajahi:
- **EDA** — Eksplorasi dan visualisasi data
- **Model Performance** — Evaluasi model klasifikasi
- **Clustering** — Segmentasi pasien K-Means
- **Prediction** — Prediksi risiko untuk pasien baru
- **SHAP Explainability** — Interpretasi model dengan SHAP
"""
)

col1, col2, col3 = st.columns(3)
col1.metric("Total Pasien", "918")
col2.metric("Fitur", "11")
col3.metric("Model Terbaik", "XGBoost")
