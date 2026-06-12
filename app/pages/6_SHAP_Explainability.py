import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils import (
    load_all_models, load_shap_values, load_test_data,
    PLOTLY_TEMPLATE, COLOR_POSITIVE, COLOR_NEGATIVE, models_exist,
    MODEL_DISPLAY_NAMES, MODEL_KEYS,
)
from src.explainability.shap_explainer import get_explainer, compute_shap_values, get_summary_data
from src.utils.io_helpers import load_numpy

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "model")

st.set_page_config(page_title="SHAP | Heart Disease", page_icon=None, layout="wide")
st.title("SHAP Explainability")
st.markdown("Interpretasi model menggunakan SHAP (SHapley Additive exPlanations).")

if not models_exist():
    st.error("Model artifacts tidak ditemukan. Jalankan `python scripts/train_pipeline.py`.")
    st.stop()

artifacts = load_all_models()
X_test, y_test = load_test_data()
feature_columns = artifacts["feature_columns"]

# Model selector
model_choice = st.selectbox("Pilih Model:", ["XGBoost", "Random Forest", "Logistic Regression"])
model_key = MODEL_KEYS[model_choice]
model = artifacts[model_key]

@st.cache_data
def get_shap_vals(model_key: str):
    if model_key == "xgb":
        cached = load_shap_values()
        if cached is not None:
            return cached
    X_train = load_numpy(os.path.join(MODELS_DIR, "X_train.npy"))
    mdl = load_all_models()[model_key]
    exp = get_explainer(mdl, X_train[:100])
    X_test_data, _ = load_test_data()
    return compute_shap_values(exp, X_test_data)

with st.spinner("Menghitung SHAP values..."):
    shap_values = get_shap_vals(model_key)

# ── Global: Mean SHAP Bar ────────────────────────────────────────
st.subheader("Global Feature Importance (Mean |SHAP|)")
summary = get_summary_data(shap_values, feature_columns)
summary_df = pd.DataFrame({
    "Feature": summary["features"][:15],
    "Mean |SHAP|": summary["mean_abs_shap"][:15],
})
fig_global = px.bar(
    summary_df, x="Mean |SHAP|", y="Feature", orientation="h",
    color="Mean |SHAP|", color_continuous_scale="Reds",
    template=PLOTLY_TEMPLATE,
    title="Top 15 Fitur berdasarkan Mean Absolute SHAP Value",
)
fig_global.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_global, use_container_width=True)

# ── Global: SHAP Beeswarm (matplotlib) ──────────────────────────
st.subheader("Beeswarm Plot")
try:
    import shap
    X_train = load_numpy(os.path.join(MODELS_DIR, "X_train.npy"))
    explainer = get_explainer(model, X_train[:100])

    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(len(shap_values),
                            explainer.expected_value[1]
                            if isinstance(explainer.expected_value, (list, np.ndarray))
                            else explainer.expected_value),
        data=X_test,
        feature_names=feature_columns,
    )
    fig_bee, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    st.pyplot(fig_bee)
    plt.close()
except Exception as e:
    st.info(f"Beeswarm plot tidak tersedia: {e}")

st.markdown("---")

# ── Local: Waterfall ─────────────────────────────────────────────
st.subheader("Local Explanation — Satu Prediksi")
sample_idx = st.number_input(
    f"Pilih index sampel (0 – {len(X_test)-1}):",
    min_value=0, max_value=len(X_test) - 1, value=0, step=1,
)

row_shap = shap_values[sample_idx]
top_idx = np.argsort(np.abs(row_shap))[::-1][:10]

local_df = pd.DataFrame({
    "Feature": [feature_columns[i] for i in top_idx],
    "SHAP Value": [row_shap[i] for i in top_idx],
    "Feature Value": [X_test[sample_idx][i] for i in top_idx],
})
local_df["Direction"] = local_df["SHAP Value"].apply(
    lambda v: "Increase Risk" if v > 0 else "Decrease Risk"
)

col_l, col_r = st.columns([2, 1])
with col_l:
    fig_local = px.bar(
        local_df, x="SHAP Value", y="Feature", orientation="h",
        color="Direction",
        color_discrete_map={"Increase Risk": COLOR_POSITIVE, "Decrease Risk": COLOR_NEGATIVE},
        template=PLOTLY_TEMPLATE,
        title=f"Waterfall SHAP — Sampel #{sample_idx}",
    )
    fig_local.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_local, use_container_width=True)

with col_r:
    st.markdown("**Nilai Fitur Sampel**")
    st.dataframe(
        local_df[["Feature", "Feature Value", "SHAP Value"]].round(4),
        use_container_width=True,
        hide_index=True,
    )
    actual_label = "Heart Disease" if y_test[sample_idx] == 1 else "No Disease"
    st.metric("Label Aktual", actual_label)

# ── Waterfall via shap library ───────────────────────────────────
try:
    import shap
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1])

    waterfall_exp = shap.Explanation(
        values=row_shap,
        base_values=base_val,
        data=X_test[sample_idx],
        feature_names=feature_columns,
    )
    fig_wf, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(waterfall_exp, max_display=12, show=False)
    st.pyplot(fig_wf)
    plt.close()
except Exception as e:
    st.info(f"Waterfall SHAP (matplotlib) tidak tersedia: {e}")
