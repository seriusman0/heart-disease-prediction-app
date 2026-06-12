import streamlit as st
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils import load_raw_df, PLOTLY_TEMPLATE, COLOR_PALETTE
from src.data.loader import describe_dataset

st.set_page_config(page_title="Home | Heart Disease", page_icon=None, layout="wide")
st.title("Beranda — Ringkasan Proyek")

df = load_raw_df()
info = describe_dataset(df)

# ── Dataset Summary ──────────────────────────────────────────────
st.subheader("Ringkasan Dataset")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Record", info["n_rows"])
c2.metric("Fitur", info["n_features"])
c3.metric("Positif (HeartDisease=1)", info["n_positive"])
c4.metric("Kolesterol = 0 (akan diimputasi)", info["zero_cholesterol"])

# ── Class Balance ────────────────────────────────────────────────
st.subheader("Distribusi Kelas Target")
col_a, col_b = st.columns([1, 2])
with col_a:
    balance_df = {
        "Label": ["Heart Disease", "No Heart Disease"],
        "Count": [info["n_positive"], info["n_negative"]],
    }
    fig_pie = px.pie(
        balance_df, values="Count", names="Label",
        color_discrete_sequence=COLOR_PALETTE,
        template=PLOTLY_TEMPLATE,
        title="Class Balance",
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    st.markdown("#### Deskripsi Fitur")
    feature_info = {
        "Age": "Usia pasien (tahun)",
        "Sex": "Jenis kelamin (M/F)",
        "ChestPainType": "Tipe nyeri dada (ATA/NAP/ASY/TA)",
        "RestingBP": "Tekanan darah istirahat (mmHg)",
        "Cholesterol": "Kolesterol serum (mm/dl)",
        "FastingBS": "Gula darah puasa > 120 mg/dl (0/1)",
        "RestingECG": "Hasil ECG istirahat",
        "MaxHR": "Detak jantung maksimum",
        "ExerciseAngina": "Angina saat olahraga (Y/N)",
        "Oldpeak": "Depresi ST akibat latihan",
        "ST_Slope": "Kemiringan segmen ST puncak",
    }
    import pandas as pd
    feat_df = pd.DataFrame(list(feature_info.items()), columns=["Fitur", "Deskripsi"])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ── CRISP-DM Phases ──────────────────────────────────────────────
st.subheader("Tahapan CRISP-DM")
phases = [
    ("1. Business Understanding", "Definisi masalah: deteksi dini penyakit jantung untuk intervensi medis."),
    ("2. Data Understanding", "EDA: distribusi, korelasi, outlier, nilai nol pada Kolesterol."),
    ("3. Data Preparation", "Imputasi Cholesterol=0 → median, encoding, scaling (StandardScaler)."),
    ("4. Modeling", "Classification: LR, Random Forest, XGBoost. Clustering: K-Means."),
    ("5. Evaluation", "Metrik: F1-Score, ROC-AUC. Explainability: SHAP."),
    ("6. Deployment", "Streamlit web app dengan 6 halaman interaktif."),
]
for title, desc in phases:
    with st.expander(title):
        st.write(desc)

# ── Raw Data Preview ─────────────────────────────────────────────
st.subheader("Sampel Data")
st.dataframe(df.head(10), use_container_width=True)
