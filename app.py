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

    /* --- Home page: hero & educational content --- */
    .hero-box {
        background: linear-gradient(135deg, #1a2b4c 0%, #2c4a7c 55%, #3d6bb3 100%);
        border-radius: 18px; padding: 2.6rem 2.4rem; color: white;
        margin-bottom: 1.6rem; box-shadow: 0 6px 24px rgba(26,43,76,0.25);
    }
    .hero-title { font-size: 2.3rem; font-weight: 800; margin: 0 0 0.4rem 0; }
    .hero-subtitle { font-size: 1.05rem; font-weight: 400; color: #dce6f7; max-width: 720px; }
    .hero-badges { margin-top: 1rem; }
    .hero-badge {
        display: inline-block; background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.28); border-radius: 999px;
        padding: 0.3rem 0.85rem; font-size: 0.82rem; margin-right: 0.5rem;
        margin-bottom: 0.4rem;
    }
    .info-card {
        background: white; border-radius: 14px; padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06); border: 1px solid #eef1f6;
        height: 100%;
    }
    .info-card h4 { margin: 0 0 0.5rem 0; color: #1a2b4c; font-size: 1.02rem; }
    .info-card ul { margin: 0; padding-left: 1.1rem; }
    .info-card li { margin-bottom: 0.35rem; font-size: 0.93rem; color: #33425c; }
    .subsection-title {
        font-size: 1.15rem; font-weight: 700; color: #1a2b4c;
        margin: 1.6rem 0 0.7rem 0;
    }
    .why-early-box {
        background: #eef4ff; border-left: 5px solid #3d6bb3; border-radius: 8px;
        padding: 1rem 1.2rem; font-size: 0.95rem; color: #1a2b4c;
    }
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

# Session state default so the Home page's "Start Prediction" button can
# jump to the prediction page without altering how navigation itself works.
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "🏠 Home"

# If a page switch was requested (e.g. via the Home page CTA button) on the
# previous run, apply it now — BEFORE the radio widget below is instantiated.
# (Streamlit forbids writing to a widget's session_state key after that
# widget has already been created in the same run, so the request is staged
# in "pending_nav" by the button and consumed here on the following rerun.)
if "pending_nav" in st.session_state:
    st.session_state["nav_page"] = st.session_state.pop("pending_nav")

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
        key="nav_page",
    )
    st.markdown("---")
    st.caption("Artifacts expected in `/models` and `/data` — see README.")

# ===========================================================================
# HOME
# ===========================================================================
if page == "🏠 Home":

    # --- Hero section --------------------------------------------------
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">🧠 Stroke Risk AI</div>
            <div class="hero-subtitle">
                An explainable machine learning &amp; deep learning platform that
                estimates stroke risk from patient clinical data and CT brain scans —
                built to make risk factors and predictions easy to understand.
            </div>
            <div class="hero-badges">
                <span class="hero-badge">📋 Clinical Risk Prediction</span>
                <span class="hero-badge">🩻 CT Scan Analysis</span>
                <span class="hero-badge">🔍 Explainable AI (LIME)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cta_col, _ = st.columns([1, 2])
    with cta_col:
        if st.button("🚀 Start Prediction", use_container_width=True, type="primary"):
            st.session_state["pending_nav"] = "📋 Tabular Risk Prediction"
            st.rerun()

    st.markdown("---")

    # --- What is a stroke? ----------------------------------------------
    st.markdown('<div class="section-title">What Is a Stroke?</div>', unsafe_allow_html=True)
    st.write(
        "A **stroke** occurs when the blood supply to part of the brain is interrupted "
        "or reduced, preventing brain tissue from getting oxygen and nutrients. Brain "
        "cells begin to die within minutes, which is why a stroke is a **medical "
        "emergency** — fast recognition and treatment can greatly reduce brain damage "
        "and other complications. Strokes are broadly grouped into **ischemic** "
        "(caused by a blocked artery) and **hemorrhagic** (caused by a leaking or "
        "burst blood vessel)."
    )

    # --- Symptoms & risk factors -----------------------------------------
    st.markdown('<div class="subsection-title">⚠️ Common Symptoms</div>', unsafe_allow_html=True)
    sym_col, risk_col = st.columns(2)

    with sym_col:
        st.markdown(
            """
            <div class="info-card">
                <h4>🚨 Recognize the warning signs</h4>
                <ul>
                    <li>Sudden weakness or numbness in the face, arm, or leg — often on one side of the body</li>
                    <li>Sudden difficulty speaking or understanding speech</li>
                    <li>Sudden vision problems in one or both eyes</li>
                    <li>Sudden severe headache with no known cause</li>
                    <li>Sudden dizziness, loss of balance, or trouble walking</li>
                    <li>Sudden confusion or trouble with coordination</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with risk_col:
        st.markdown(
            """
            <div class="info-card">
                <h4>📈 Major Risk Factors</h4>
                <ul>
                    <li>High blood pressure (hypertension)</li>
                    <li>Diabetes / high blood glucose</li>
                    <li>Smoking and tobacco use</li>
                    <li>Obesity and physical inactivity</li>
                    <li>High cholesterol</li>
                    <li>Heart disease and irregular heart rhythm</li>
                    <li>Increasing age and family history</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Prevention -------------------------------------------------------
    st.markdown('<div class="subsection-title">🛡️ Prevention Tips</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(
            """
            <div class="info-card">
                <h4>🥗 Healthy Lifestyle</h4>
                <ul>
                    <li>Eat a balanced, low-sodium, low-fat diet</li>
                    <li>Exercise regularly (aim for 150+ min/week)</li>
                    <li>Maintain a healthy body weight</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            """
            <div class="info-card">
                <h4>💊 Manage Conditions</h4>
                <ul>
                    <li>Control blood pressure and blood glucose</li>
                    <li>Manage cholesterol levels</li>
                    <li>Take prescribed medications consistently</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            """
            <div class="info-card">
                <h4>🩺 Stay Proactive</h4>
                <ul>
                    <li>Quit smoking and limit alcohol intake</li>
                    <li>Schedule regular medical checkups</li>
                    <li>Know your personal and family risk factors</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Why early prediction matters -------------------------------------
    st.markdown('<div class="subsection-title">⏱️ Why Early Prediction Matters</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="why-early-box">
            Early identification of stroke risk allows patients and clinicians to act
            <b>before</b> a stroke happens — adjusting lifestyle, starting preventive
            treatment, or scheduling closer monitoring. Because stroke damage progresses
            within minutes, catching risk factors early — through clinical data or
            CT-based screening — can support timely intervention, potentially
            reducing severity, long-term disability, and the chance of a first or
            recurrent stroke.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    disclaimer()

    # --- System status (existing functionality, unchanged) -----------------
    with st.expander("⚙️ System status — model & data artifacts", expanded=False):
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
