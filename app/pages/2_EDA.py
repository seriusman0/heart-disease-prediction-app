import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.utils import load_raw_df, PLOTLY_TEMPLATE, COLOR_PALETTE, COLOR_POSITIVE, COLOR_NEGATIVE

st.set_page_config(page_title="EDA | Heart Disease", page_icon="🔍", layout="wide")
st.title("🔍 Eksplorasi Data (EDA)")

df_raw = load_raw_df()

# Cleaned version
df_clean = df_raw.copy()
chol_median = df_clean.loc[df_clean["Cholesterol"] != 0, "Cholesterol"].median()
df_clean.loc[df_clean["Cholesterol"] == 0, "Cholesterol"] = chol_median

view = st.radio("Tampilkan data:", ["Data Asli", "Data Setelah Preprocessing"], horizontal=True)
df = df_raw if view == "Data Asli" else df_clean

# ── Statistics ───────────────────────────────────────────────────
st.subheader("📈 Statistik Deskriptif")
st.dataframe(df.describe().round(2), use_container_width=True)

st.markdown("---")

# ── Missing / Zero Values ────────────────────────────────────────
st.subheader("🔎 Nilai Nol & Missing")
col1, col2 = st.columns(2)
with col1:
    nulls = df.isnull().sum().reset_index()
    nulls.columns = ["Fitur", "Null Count"]
    st.dataframe(nulls, use_container_width=True, hide_index=True)
with col2:
    if view == "Data Asli":
        zero_chol = int((df["Cholesterol"] == 0).sum())
        st.metric("Kolesterol = 0", zero_chol, delta=f"akan diimputasi dengan median {chol_median:.1f}")
    else:
        st.info(f"Kolesterol nol telah diimputasi dengan median: {chol_median:.1f}")

st.markdown("---")

# ── Distribution Plots ───────────────────────────────────────────
st.subheader("📊 Distribusi Fitur Numerik")
numeric_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
selected_num = st.selectbox("Pilih fitur numerik:", numeric_cols)
fig_hist = px.histogram(
    df, x=selected_num, color="HeartDisease",
    color_discrete_map={0: COLOR_NEGATIVE, 1: COLOR_POSITIVE},
    barmode="overlay", template=PLOTLY_TEMPLATE,
    title=f"Distribusi {selected_num} berdasarkan HeartDisease",
    labels={"HeartDisease": "Heart Disease"},
    opacity=0.75,
)
st.plotly_chart(fig_hist, use_container_width=True)

# ── Categorical ──────────────────────────────────────────────────
st.subheader("📊 Distribusi Fitur Kategorikal")
cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope", "FastingBS"]
selected_cat = st.selectbox("Pilih fitur kategorikal:", cat_cols)
cat_counts = df.groupby([selected_cat, "HeartDisease"]).size().reset_index(name="count")
fig_bar = px.bar(
    cat_counts, x=selected_cat, y="count", color="HeartDisease",
    color_discrete_map={0: COLOR_NEGATIVE, 1: COLOR_POSITIVE},
    barmode="group", template=PLOTLY_TEMPLATE,
    title=f"Distribusi {selected_cat} berdasarkan HeartDisease",
)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ── Correlation Heatmap ──────────────────────────────────────────
st.subheader("🔥 Heatmap Korelasi")
num_df = df[["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak", "HeartDisease"]]
corr = num_df.corr().round(2)
fig_corr = px.imshow(
    corr, text_auto=True, template=PLOTLY_TEMPLATE,
    color_continuous_scale="RdBu_r",
    title="Matriks Korelasi Fitur Numerik",
    zmin=-1, zmax=1,
)
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# ── Box Plots ────────────────────────────────────────────────────
st.subheader("📦 Box Plot Fitur Numerik vs HeartDisease")
selected_box = st.selectbox("Pilih fitur untuk box plot:", numeric_cols, key="box")
df_box = df.copy()
df_box["HeartDisease"] = df_box["HeartDisease"].map({0: "No Disease", 1: "Heart Disease"})
fig_box = px.box(
    df_box, x="HeartDisease", y=selected_box,
    color="HeartDisease",
    color_discrete_map={"No Disease": COLOR_NEGATIVE, "Heart Disease": COLOR_POSITIVE},
    template=PLOTLY_TEMPLATE,
    title=f"{selected_box} vs HeartDisease",
    points="outliers",
)
st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")

# ── Scatter Age vs MaxHR ─────────────────────────────────────────
st.subheader("🔵 Age vs MaxHR (colored by HeartDisease)")
df_scatter = df.copy()
df_scatter["HeartDisease"] = df_scatter["HeartDisease"].map({0: "No Disease", 1: "Heart Disease"})
fig_scatter = px.scatter(
    df_scatter, x="Age", y="MaxHR",
    color="HeartDisease",
    color_discrete_map={"No Disease": COLOR_NEGATIVE, "Heart Disease": COLOR_POSITIVE},
    template=PLOTLY_TEMPLATE,
    title="Age vs MaxHR",
    opacity=0.7,
    hover_data=["Sex", "ChestPainType"],
)
st.plotly_chart(fig_scatter, use_container_width=True)
