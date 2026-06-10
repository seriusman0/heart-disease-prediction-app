import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.utils import (
    load_all_models, load_eval_results, load_test_data,
    PLOTLY_TEMPLATE, COLOR_POSITIVE, COLOR_NEGATIVE, models_exist,
)
from src.models.evaluator import get_feature_importances

st.set_page_config(page_title="Model Performance | Heart Disease", page_icon="📈", layout="wide")
st.title("📈 Evaluasi Model Klasifikasi")

if not models_exist():
    st.error("Model artifacts tidak ditemukan. Jalankan `python scripts/train_pipeline.py`.")
    st.stop()

artifacts = load_all_models()
results = load_eval_results()
X_test, y_test = load_test_data()

# ── Comparison Table ─────────────────────────────────────────────
st.subheader("🏆 Perbandingan Semua Model")
compare_rows = []
for key, name in [("logistic_regression", "Logistic Regression"),
                  ("random_forest", "Random Forest"),
                  ("xgboost", "XGBoost")]:
    r = results[key]
    compare_rows.append({
        "Model": name,
        "Accuracy": f"{r['accuracy']:.4f}",
        "F1-Score": f"{r['f1']:.4f}",
        "ROC-AUC": f"{r['roc_auc']:.4f}",
        "Precision": f"{r['precision']:.4f}",
        "Recall": f"{r['recall']:.4f}",
    })
st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

# Bar chart comparison
metrics = ["f1", "roc_auc", "accuracy"]
metric_labels = {"f1": "F1-Score", "roc_auc": "ROC-AUC", "accuracy": "Accuracy"}
model_names = ["logistic_regression", "random_forest", "xgboost"]
display_names = ["Logistic Regression", "Random Forest", "XGBoost"]
colors = ["#636EFA", "#00CC96", "#EF553B"]

fig_compare = go.Figure()
for metric in metrics:
    fig_compare.add_trace(go.Bar(
        name=metric_labels[metric],
        x=display_names,
        y=[results[m][metric] for m in model_names],
    ))
fig_compare.update_layout(
    barmode="group", template=PLOTLY_TEMPLATE,
    title="Perbandingan Metrik Semua Model",
    yaxis=dict(range=[0.7, 1.0]),
)
st.plotly_chart(fig_compare, use_container_width=True)

st.markdown("---")

# ── Per-Model Detail ─────────────────────────────────────────────
st.subheader("🔬 Detail Per Model")
tab_lr, tab_rf, tab_xgb = st.tabs(["Logistic Regression", "Random Forest", "XGBoost"])

model_map = {
    "logistic_regression": (tab_lr, artifacts["lr"]),
    "random_forest": (tab_rf, artifacts["rf"]),
    "xgboost": (tab_xgb, artifacts["xgb"]),
}

feature_columns = artifacts["feature_columns"]

for key, (tab, model) in model_map.items():
    r = results[key]
    with tab:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{r['accuracy']:.4f}")
        col2.metric("F1-Score", f"{r['f1']:.4f}")
        col3.metric("ROC-AUC", f"{r['roc_auc']:.4f}")
        col4.metric("Precision", f"{r['precision']:.4f}")
        col5.metric("Recall", f"{r['recall']:.4f}")

        c_left, c_right = st.columns(2)

        # Confusion Matrix
        with c_left:
            cm = np.array(r["confusion_matrix"])
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual"),
                x=["No Disease", "Heart Disease"],
                y=["No Disease", "Heart Disease"],
                text_auto=True,
                color_continuous_scale="Blues",
                template=PLOTLY_TEMPLATE,
                title="Confusion Matrix",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # ROC Curve
        with c_right:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=r["fpr"], y=r["tpr"],
                mode="lines",
                name=f"ROC (AUC={r['roc_auc']:.4f})",
                line=dict(color=COLOR_POSITIVE, width=2),
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(color="gray", dash="dash"),
            ))
            fig_roc.update_layout(
                template=PLOTLY_TEMPLATE,
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        # Feature Importance
        st.subheader("Feature Importance")
        fi_df = get_feature_importances(model, feature_columns).head(15)
        fig_fi = px.bar(
            fi_df, x="importance", y="feature", orientation="h",
            template=PLOTLY_TEMPLATE,
            title="Top 15 Feature Importances",
            color="importance",
            color_continuous_scale="Reds",
        )
        fig_fi.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fi, use_container_width=True)
