"""
Stroke Risk AI — Streamlit Application
=======================================
Built from:
  - 4_Model_Development.ipynb   (tabular Logistic Regression stroke model)
  - 5_LIME_Explainability.ipynb (LIME explanations for the tabular model)
  - 7_CT_Model_Development.ipynb (fine-tuned ResNet50 CT scan stroke model)

No model is retrained or modified — this app only loads saved artifacts and
reproduces the exact preprocessing / inference pipelines from the notebooks.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import utils
from utils import (
    MODELS_DIR, DATA_DIR, CATEGORY_OPTIONS, TABULAR_REPORTED_METRICS,
    CT_REPORTED_METRICS, CT_BASELINE_CM, CT_FINETUNED_CM,
)

# ---------------------------------------------------------------------------
# Page config & global style
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Stroke Risk AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main { background-color: #f7f9fc; }
    .metric-card {
        background: white; border-radius: 14px; padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06); border: 1px solid #eef1f6;
    }
    .risk-banner {
        border-radius: 14px; padding: 1.4rem 1.6rem; color: white;
        font-weight: 600; font-size: 1.15rem; margin: 0.6rem 0 1rem 0;
    }
    .section-title {
        font-size: 1.4rem; font-weight: 700; color: #1a2b4c;
        margin-bottom: 0.3rem;
    }
    .disclaimer-box {
        background: #fff4e5; border-left: 5px solid #e67e22; border-radius: 8px;
        padding: 0.9rem 1.1rem; font-size: 0.92rem; color: #6b4a1e;
    }
    .missing-box {
        background: #fdecea; border-left: 5px solid #e74c3c; border-radius: 8px;
        padding: 0.9rem 1.1rem; font-size: 0.9rem; color: #7a1f1f;
    }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

MEDICAL_DISCLAIMER = (
    "⚠️ **Medical Disclaimer:** This application is an educational / research "
    "demonstration of machine learning models and is **not a certified medical "
    "device**. It must not be used for real clinical diagnosis or treatment "
    "decisions. Always consult a qualified healthcare professional for medical "
    "advice."
)


def disclaimer():
    st.markdown(f'<div class="disclaimer-box">{MEDICAL_DISCLAIMER}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🧠 Stroke Risk AI")
    st.caption("ML/DL-powered stroke risk assessment dashboard")
    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "📋 Tabular Risk Prediction",
            "📊 Dataset Exploration",
            "📚 Disease Information",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Artifacts expected in `/models` and `/data` — see README.")

# ===========================================================================
# HOME
# ===========================================================================
if page == "🏠 Home":
    st.markdown('<div class="section-title">Stroke Risk AI — Overview</div>', unsafe_allow_html=True)

    st.write(
        "This dashboard provides a **stroke risk assessment system** based on "
        "patient clinical and demographic information."
    )

    st.markdown("---")

    st.markdown("#### 📋 Tabular Risk Model")
    st.write(
        "A **Logistic Regression** classifier trained on patient clinical/demographic "
        "data including age, glucose level, BMI, hypertension, heart disease, and "
        "lifestyle factors to estimate stroke risk. The model uses explainable "
        "machine learning with LIME to highlight the factors influencing each prediction."
    )

    disclaimer()

    st.markdown("#### System status")

    tab_art = utils.load_tabular_artifacts()

    s1, s2, s3, s4 = st.columns(4)

    def status_chip(col, label, ok):
        with col:
            st.metric(label, "✅ Ready" if ok else "❌ Missing")

    status_chip(s1, "LR Model", tab_art["model"] is not None)
    status_chip(s2, "Encoder", tab_art["encoder"] is not None)
    status_chip(s3, "Scaler", tab_art["scaler"] is not None)
    status_chip(
        s4,
        "Dataset CSV",
        os.path.exists(os.path.join(DATA_DIR, "stroke_data_cleaned.csv"))
    )

    missing = tab_art["errors"]

    if missing:
        st.markdown(
            '<div class="missing-box"><b>Missing artifacts detected:</b><br>' +
            "<br>".join(f"• {m}" for m in missing) +
            "<br><br>See the README for exactly which files to place in <code>/models</code> "
            "and <code>/data</code>.</div>",
            unsafe_allow_html=True,
        )


# ===========================================================================
# TABULAR RISK PREDICTION
# ===========================================================================
elif page == "📋 Tabular Risk Prediction":
    st.markdown('<div class="section-title">📋 Tabular Stroke Risk Prediction</div>', unsafe_allow_html=True)
    st.write("Fill in the patient's clinical and demographic details. The exact "
             "one-hot encoding + scaling pipeline from the notebook is applied before prediction.")
    disclaimer()

    tab_art = utils.load_tabular_artifacts()
    if tab_art["model"] is None:
        st.error("Logistic Regression model not available.")
        st.info("Place `logistic_regression_model.pkl` in `/models`. Ideally also provide "
                "`encoder.pkl`, `scaler.pkl`, and `feature_names.pkl` for exact fidelity.")
    else:
        if tab_art["errors"]:
            with st.expander("⚠️ Some artifacts are missing — click for details", expanded=False):
                for e in tab_art["errors"]:
                    st.write(f"- {e}")

        with st.form("tabular_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", min_value=0, max_value=120, value=50)
                hypertension = st.selectbox("Hypertension", ["No", "Yes"])
                heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
            with c2:
                avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)",
                                                      min_value=40.0, max_value=300.0, value=100.0)
                bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)
                gender = st.selectbox("Gender", CATEGORY_OPTIONS["gender"])
            with c3:
                ever_married = st.selectbox("Ever Married", CATEGORY_OPTIONS["ever_married"])
                work_type = st.selectbox("Work Type", CATEGORY_OPTIONS["work_type"])
                residence = st.selectbox("Residence Type", CATEGORY_OPTIONS["Residence_type"])
            smoking_status = st.selectbox("Smoking Status", CATEGORY_OPTIONS["smoking_status"])

            submitted = st.form_submit_button("Predict Stroke Risk", use_container_width=True)

        if submitted:
            form_values = {
                "age": age, "hypertension": hypertension == "Yes",
                "heart_disease": heart_disease == "Yes",
                "avg_glucose_level": avg_glucose_level, "bmi": bmi,
                "gender": gender, "ever_married": ever_married,
                "work_type": work_type, "Residence_type": residence,
                "smoking_status": smoking_status,
            }
            try:
                raw_df = utils.build_raw_tabular_row(form_values)
                X_processed, warns = utils.preprocess_tabular(raw_df, tab_art)
                for w in warns:
                    st.warning(w)

                prediction, stroke_prob, proba = utils.predict_tabular(tab_art["model"], X_processed)
                label = utils.TABULAR_CLASS_NAMES[prediction]
                risk_label, risk_color = utils.risk_level(stroke_prob)

                st.markdown(
                    f'<div class="risk-banner" style="background:{risk_color}">'
                    f'Prediction: {label} &nbsp;|&nbsp; Stroke Probability: {stroke_prob*100:.1f}% '
                    f'&nbsp;|&nbsp; {risk_label}</div>', unsafe_allow_html=True,
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Prediction", label)
                m2.metric("Stroke Probability", f"{stroke_prob*100:.1f}%")
                m3.metric("Risk Level", risk_label)
                st.markdown("**Recommendation**")
                st.info(utils.recommendation_text(stroke_prob, prediction))

                # --- LIME explanation ---
                st.markdown("---")
                st.markdown("#### 🔍 LIME Explanation — Feature Contributions")
                if tab_art["model"] is not None:
                    try:
                        import explainability as expl
                        # Build a small synthetic background/training set for LIME if we
                        # don't have the real training data on disk; prefer the real
                        # dataset if present for a faithful explainer.
                        bg_path = os.path.join(DATA_DIR, "stroke_data_cleaned.csv")
                        feature_names = tab_art["feature_names"]
                        if os.path.exists(bg_path):
                            bg_df = pd.read_csv(bg_path)
                            bg_df = bg_df.reindex(columns=feature_names, fill_value=0)
                            training_values = bg_df.values
                        else:
                            st.caption("No dataset found for LIME's background distribution — "
                                       "using the current input replicated as a fallback "
                                       "(explanations will be less statistically meaningful; "
                                       "place `stroke_data_cleaned.csv` in `/data` for accurate LIME).")
                            training_values = np.repeat(X_processed.values, 50, axis=0)

                        explainer = expl.build_lime_explainer(
                            tuple(map(tuple, training_values)), tuple(feature_names)
                        )
                        explanation, features, weights = expl.explain_instance(
                            explainer, tab_art["model"], X_processed.values[0], num_features=10
                        )

                        fig, ax = plt.subplots(figsize=(8, 5))
                        colors = ["#e74c3c" if w > 0 else "#2ecc71" for w in weights]
                        ax.barh(features, weights, color=colors)
                        ax.axvline(x=0, color="black", linewidth=0.8)
                        ax.set_xlabel("LIME Contribution (→ increases risk / → decreases risk)")
                        ax.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)

                        inc = [f for f, w in zip(features, weights) if w > 0]
                        dec = [f for f, w in zip(features, weights) if w < 0]
                        cA, cB = st.columns(2)
                        with cA:
                            st.markdown("**⬆️ Increases risk:**")
                            st.write(", ".join(inc) if inc else "None among top features")
                        with cB:
                            st.markdown("**⬇️ Decreases risk:**")
                            st.write(", ".join(dec) if dec else "None among top features")
                    except Exception as e:
                        st.warning(f"LIME explanation unavailable: {e}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ===========================================================================
# DATASET EXPLORATION
# ===========================================================================
elif page == "📊 Dataset Exploration":
    st.markdown('<div class="section-title">📊 Dataset Exploration</div>', unsafe_allow_html=True)
    st.write("Exploration of the training dataset used for the tabular stroke risk model "
             "(read-only — no user prediction here).")

    csv_path = os.path.join(DATA_DIR, "stroke_data_cleaned.csv")
    uploaded_csv = None
    if not os.path.exists(csv_path):
        st.warning("`stroke_data_cleaned.csv` not found in `/data`. You can upload it below "
                   "to explore it in this session (it will not be saved to disk).")
        uploaded_csv = st.file_uploader("Upload stroke_data_cleaned.csv", type=["csv"])

    df = None
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)

    if df is None:
        st.info("Provide the dataset to see exploration charts.")
    else:
        st.caption("Note: this file is the *already cleaned/scaled* dataset referenced in the "
                   "notebooks — numeric columns like age/bmi/avg_glucose_level are standardized "
                   "(z-scores), not raw units, since the raw-cleaning notebook wasn't provided.")

        st.markdown("#### Preview")
        st.dataframe(df.head(20), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing values", int(df.isna().sum().sum()))

        with st.expander("Column data types"):
            st.dataframe(df.dtypes.astype(str).rename("dtype"), use_container_width=True)

        st.markdown("#### Basic statistics")
        st.dataframe(df.describe().T, use_container_width=True)

        if "stroke" in df.columns:
            st.markdown("#### Class distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            counts = df["stroke"].value_counts().sort_index()
            sns.barplot(x=counts.index.astype(str), y=counts.values, palette=["#2ecc71", "#e74c3c"], ax=ax)
            ax.set_xlabel("Stroke (0=Normal, 1=Stroke)")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        st.markdown("#### Feature distributions")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        sel_cols = st.multiselect("Choose numeric features to plot", numeric_cols,
                                   default=[c for c in ["age", "avg_glucose_level", "bmi"] if c in numeric_cols])
        for i in range(0, len(sel_cols), 3):
            row_cols = st.columns(3)
            for j, colname in enumerate(sel_cols[i:i+3]):
                with row_cols[j]:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.histplot(df[colname], kde=True, ax=ax, color="#3498db")
                    ax.set_title(colname)
                    plt.tight_layout()
                    st.pyplot(fig)

        st.markdown("#### Correlation heatmap")
        corr_cols = [c for c in numeric_cols if df[c].nunique() > 1]
        if len(corr_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax)
            plt.tight_layout()
            st.pyplot(fig)

# ===========================================================================
# DISEASE INFORMATION
# ===========================================================================
elif page == "📚 Disease Information":
    st.markdown('<div class="section-title">📚 Stroke — Disease Information</div>', unsafe_allow_html=True)
    disclaimer()

    st.markdown("#### Overview")
    st.write(
        "A stroke occurs when the blood supply to part of the brain is interrupted or "
        "reduced, depriving brain tissue of oxygen and nutrients. Brain cells begin to "
        "die within minutes. Stroke is a medical emergency and prompt treatment is "
        "crucial to minimize brain damage and complications."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Common Symptoms")
        st.markdown(
            "- Sudden numbness or weakness of the face, arm, or leg (especially one-sided)\n"
            "- Sudden confusion or trouble understanding speech\n"
            "- Sudden trouble speaking\n"
            "- Sudden vision problems in one or both eyes\n"
            "- Sudden trouble walking, dizziness, loss of balance or coordination\n"
            "- Sudden severe headache with no known cause"
        )
        st.markdown("#### Risk Factors")
        st.markdown(
            "- Hypertension (high blood pressure)\n"
            "- Heart disease / atrial fibrillation\n"
            "- Diabetes / high blood glucose\n"
            "- High BMI / obesity\n"
            "- Smoking\n"
            "- Older age\n"
            "- Sedentary lifestyle, poor diet\n"
            "- Family history of stroke"
        )
    with c2:
        st.markdown("#### Prevention")
        st.markdown(
            "- Manage blood pressure and blood glucose levels\n"
            "- Maintain a healthy weight and balanced diet\n"
            "- Exercise regularly\n"
            "- Avoid smoking and limit alcohol\n"
            "- Regular medical check-ups, especially with risk factors present\n"
            "- Take prescribed medications for hypertension/diabetes/cholesterol consistently"
        )
        st.markdown("#### When to See a Doctor")
        st.markdown(
            "Seek **immediate emergency care** if you or someone else shows sudden stroke "
            "symptoms — remember **F.A.S.T.**: **F**ace drooping, **A**rm weakness, "
            "**S**peech difficulty, **T**ime to call emergency services. For risk-factor "
            "management (e.g., hypertension, diabetes, high BMI) without acute symptoms, "
            "schedule a routine visit with a physician."
        )

st.markdown("---")
st.caption("Stroke Risk AI — built on user-provided model notebooks. Not for clinical use.")
