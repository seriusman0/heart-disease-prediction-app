import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.utils import load_all_models, PLOTLY_TEMPLATE, COLOR_POSITIVE, COLOR_NEGATIVE, models_exist, MODEL_KEYS
from src.data.preprocessor import transform_single_row
from src.explainability.shap_explainer import get_explainer, get_single_prediction_shap

st.set_page_config(page_title="Prediction | Heart Disease", page_icon="🫀", layout="wide")
st.title("🫀 Prediksi Risiko Penyakit Jantung")
st.markdown("Masukkan data pasien untuk mendapatkan prediksi risiko penyakit jantung.")

if not models_exist():
    st.error("Model artifacts tidak ditemukan. Jalankan `python scripts/train_pipeline.py`.")
    st.stop()

artifacts = load_all_models()

# ── Input Form ───────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("📋 Data Pasien")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age (Usia)", 18, 80, 50)
        sex = st.selectbox("Sex (Jenis Kelamin)", ["M", "F"])
        chest_pain = st.selectbox("ChestPainType", ["ATA", "NAP", "ASY", "TA"],
                                   help="ATA=Atypical Angina, NAP=Non-Anginal Pain, ASY=Asymptomatic, TA=Typical Angina")
        resting_bp = st.number_input("RestingBP (mmHg)", min_value=80, max_value=200, value=120)

    with col2:
        cholesterol = st.number_input("Cholesterol (mm/dl)", min_value=0, max_value=600, value=200,
                                       help="Nilai 0 akan diimputasi dengan median training data")
        fasting_bs = st.radio("FastingBS (Gula darah puasa > 120 mg/dl)", [0, 1],
                               format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")
        resting_ecg = st.selectbox("RestingECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("MaxHR (Detak jantung maks)", 60, 202, 150)

    with col3:
        exercise_angina = st.selectbox("ExerciseAngina", ["N", "Y"],
                                        format_func=lambda x: "Ya (Y)" if x == "Y" else "Tidak (N)")
        oldpeak = st.number_input("Oldpeak (ST depression)", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)
        st_slope = st.selectbox("ST_Slope", ["Up", "Flat", "Down"])
        model_choice = st.selectbox("Model Prediksi", ["XGBoost", "Random Forest", "Logistic Regression"])

    submitted = st.form_submit_button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)

# ── Prediction ───────────────────────────────────────────────────
if submitted:
    input_dict = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }

    model_key = MODEL_KEYS[model_choice]
    model = artifacts[model_key]
    scaler = artifacts["scaler"]
    feature_columns = artifacts["feature_columns"]
    chol_median = artifacts["chol_median"]

    row = transform_single_row(input_dict, scaler, feature_columns, chol_median)
    prob = float(model.predict_proba(row)[0][1])
    prediction = int(prob >= 0.5)

    st.markdown("---")
    st.subheader("📊 Hasil Prediksi")

    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        if prediction == 1:
            st.error(f"⚠️ **Risiko Tinggi Penyakit Jantung**\nProbabilitas: {prob:.1%}")
        else:
            st.success(f"✅ **Risiko Rendah Penyakit Jantung**\nProbabilitas: {prob:.1%}")

        risk_level = "Tinggi" if prob >= 0.7 else ("Sedang" if prob >= 0.4 else "Rendah")
        st.metric("Probabilitas Risiko", f"{prob:.1%}", delta=f"Risiko {risk_level}")

    with col_res2:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "Probabilitas Penyakit Jantung"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLOR_POSITIVE if prediction == 1 else COLOR_NEGATIVE},
                "steps": [
                    {"range": [0, 40], "color": "#d4edda"},
                    {"range": [40, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(template=PLOTLY_TEMPLATE, height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # SHAP for this prediction
    st.subheader("🔍 Kontribusi Fitur (SHAP)")
    try:
        from src.utils.io_helpers import load_numpy
        import matplotlib.pyplot as plt
        import shap

        MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
        X_train = load_numpy(os.path.join(MODELS_DIR, "X_train.npy"))

        explainer = get_explainer(model, X_train[:100])
        shap_result = get_single_prediction_shap(explainer, row, feature_columns)

        shap_vals = np.array(shap_result["shap_values"])
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:10]

        import plotly.express as px
        shap_df_data = {
            "Feature": [feature_columns[i] for i in top_idx],
            "SHAP Value": [shap_vals[i] for i in top_idx],
        }
        import pandas as pd
        shap_df = pd.DataFrame(shap_df_data)
        shap_df["Direction"] = shap_df["SHAP Value"].apply(lambda v: "Increase Risk" if v > 0 else "Decrease Risk")

        fig_shap = px.bar(
            shap_df, x="SHAP Value", y="Feature", orientation="h",
            color="Direction",
            color_discrete_map={"Increase Risk": COLOR_POSITIVE, "Decrease Risk": COLOR_NEGATIVE},
            template=PLOTLY_TEMPLATE,
            title="Top 10 Fitur yang Mempengaruhi Prediksi Ini",
        )
        fig_shap.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_shap, use_container_width=True)
    except Exception as e:
        st.info(f"SHAP plot tidak tersedia: {e}")
