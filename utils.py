"""
utils.py
--------
Core pipeline logic extracted 1:1 from the source notebooks:
  - 4_Model_Development.ipynb        (tabular Logistic Regression stroke model)
  - 5_LIME_Explainability.ipynb      (LIME explanation pipeline)
  - 7_CT_Model_Development.ipynb     (ResNet50 CT scan stroke model)

No model is retrained here. This module only loads saved artifacts and
reproduces the exact preprocessing / inference steps used in the notebooks.
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------------------
# Constants extracted directly from the notebooks
# ---------------------------------------------------------------------------

# From 4_Model_Development.ipynb / 5_LIME_Explainability.ipynb
CATEGORICAL_COLS = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

# Only these numeric columns are continuous / were standardized upstream.
# hypertension and heart_disease are already binary 0/1 flags (not scaled).
NUMERIC_COLS_TO_SCALE = ["age", "avg_glucose_level", "bmi"]

# Categories inferred from the fitted OneHotEncoder's output columns
# (drop="first", sparse_output=False, handle_unknown="ignore") seen in the
# notebook's df.head() / feature_names.pkl output:
# ['age','hypertension','heart_disease','avg_glucose_level','bmi',
#  'gender_Male','gender_Other','ever_married_Yes','work_type_Never_worked',
#  'work_type_Private','work_type_Self-employed','work_type_children',
#  'Residence_type_Urban','smoking_status_formerly smoked',
#  'smoking_status_never smoked','smoking_status_smokes']
CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "ever_married": ["No", "Yes"],
    "work_type": ["Govt_job", "Never_worked", "Private", "Self-employed", "children"],
    "Residence_type": ["Rural", "Urban"],
    "smoking_status": ["Unknown", "formerly smoked", "never smoked", "smokes"],
}

# Expected final feature order (from feature_names.pkl / X_test.columns.tolist())
FALLBACK_FEATURE_NAMES = [
    "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
    "gender_Male", "gender_Other", "ever_married_Yes",
    "work_type_Never_worked", "work_type_Private", "work_type_Self-employed",
    "work_type_children", "Residence_type_Urban",
    "smoking_status_formerly smoked", "smoking_status_never smoked",
    "smoking_status_smokes",
]

TABULAR_CLASS_NAMES = ["Normal", "Stroke"]   # from 5_LIME_Explainability.ipynb
CT_CLASS_NAMES = ["Normal", "Stroke"]        # from 7_CT_Model_Development.ipynb (alphabetical, label_mode="binary")
CT_IMG_SIZE = (224, 224)                     # IMG_HEIGHT, IMG_WIDTH in notebook 7

# Decision threshold: the notebook's conclusion explicitly selects 0.5 as the
# most practical threshold for the final Logistic Regression model.
TABULAR_THRESHOLD = 0.5
CT_THRESHOLD = 0.5  # notebook 7 uses (y_prob >= 0.5) throughout

# Static evaluation numbers reported in the notebooks (used as a fallback in
# the Model Dashboard when no live test set is available in this app).
TABULAR_REPORTED_METRICS = {
    "Logistic Regression (final model)": {
        "Accuracy": 0.753, "Precision": None, "Recall": 0.80, "F1 Score": 0.241,
    },
    "Random Forest": {"Accuracy": 0.915, "Precision": None, "Recall": 0.16, "F1 Score": None},
    "LDA": {"Accuracy": None, "Precision": None, "Recall": 0.80, "F1 Score": None},
}

CT_REPORTED_METRICS = {
    "Baseline ResNet50":   {"Accuracy": 0.8124, "Precision": 0.6519, "Recall": 0.7923, "AUC": 0.8862},
    "Fine-Tuned ResNet50": {"Accuracy": 0.8696, "Precision": 0.7325, "Recall": 0.8846, "AUC": 0.9417},
}
CT_BASELINE_CM = np.array([[252, 55], [27, 103]])   # [[TN, FP], [FN, TP]]
CT_FINETUNED_CM = np.array([[265, 42], [15, 115]])


# ---------------------------------------------------------------------------
# Artifact loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_tabular_artifacts():
    """Load the LR model + preprocessing artifacts saved by notebook 4/5.
    Returns a dict; any missing artifact is reported as None so the UI can
    degrade gracefully and tell the user exactly what's missing.
    """
    artifacts = {"model": None, "encoder": None, "scaler": None, "feature_names": None, "errors": []}

    paths = {
        "model": os.path.join(MODELS_DIR, "logistic_regression_model.pkl"),
        "encoder": os.path.join(MODELS_DIR, "encoder.pkl"),
        "scaler": os.path.join(MODELS_DIR, "scaler.pkl"),
        "feature_names": os.path.join(MODELS_DIR, "feature_names.pkl"),
    }

    for key, path in paths.items():
        if os.path.exists(path):
            try:
                artifacts[key] = joblib.load(path)
            except Exception as e:
                artifacts["errors"].append(f"Failed to load {os.path.basename(path)}: {e}")
        else:
            artifacts["errors"].append(f"Missing file: {os.path.basename(path)}")

    if artifacts["feature_names"] is None:
        artifacts["feature_names"] = FALLBACK_FEATURE_NAMES

    return artifacts


@st.cache_resource(show_spinner=False)
def load_ct_model():
    """Load the fine-tuned (preferred) or baseline Keras ResNet50 model."""
    import tensorflow as tf

    candidates = [
        os.path.join(MODELS_DIR, "best_resnet50_finetuned.keras"),
        os.path.join(MODELS_DIR, "resnet50_baseline.keras"),
        os.path.join(MODELS_DIR, "best_resnet50_model.keras"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                from tensorflow.keras.layers import Dense

                class FixedDense(Dense):
                    def __init__(self, *args, **kwargs):
                        kwargs.pop("quantization_config", None)
                        super().__init__(*args, **kwargs)

                model = tf.keras.models.load_model(
                    path,
                    compile=False,
                    safe_mode=False,
                    custom_objects={
                        "Dense": FixedDense
                    }
                )
                return model, os.path.basename(path), None
            except Exception as e:
                return None, None, f"Failed to load {os.path.basename(path)}: {e}"
    return None, None, "No CT model file found in /models (expected best_resnet50_finetuned.keras or resnet50_baseline.keras)."


# ---------------------------------------------------------------------------
# Tabular preprocessing (mirrors notebook 4 / 5 exactly)
# ---------------------------------------------------------------------------

def build_raw_tabular_row(form_values: dict) -> pd.DataFrame:
    """Assemble a single-row DataFrame in the raw column layout the
    notebook's encoder/scaler expect, straight from the Streamlit form."""
    row = {
        "age": form_values["age"],
        "hypertension": int(form_values["hypertension"]),
        "heart_disease": int(form_values["heart_disease"]),
        "avg_glucose_level": form_values["avg_glucose_level"],
        "bmi": form_values["bmi"],
        "gender": form_values["gender"],
        "ever_married": form_values["ever_married"],
        "work_type": form_values["work_type"],
        "Residence_type": form_values["Residence_type"],
        "smoking_status": form_values["smoking_status"],
    }
    return pd.DataFrame([row])


def preprocess_tabular(raw_df: pd.DataFrame, artifacts: dict):
    """Reproduce the exact preprocessing pipeline from notebook 4/5:
    1. One-hot encode categorical columns (saved encoder, or manual fallback)
    2. Scale continuous numeric columns (saved scaler, if available)
    3. Reindex to the training feature order, filling missing with 0
    Returns (X_processed_df, warnings_list)
    """
    warnings = []
    df = raw_df.copy()

    encoder = artifacts.get("encoder")
    scaler = artifacts.get("scaler")
    feature_names = artifacts.get("feature_names") or FALLBACK_FEATURE_NAMES

    # --- 1. Categorical encoding ---
    if encoder is not None:
        encoded = encoder.transform(df[CATEGORICAL_COLS])
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
            index=df.index,
        )
    else:
        warnings.append(
            "encoder.pkl not found — using a manual one-hot encoding reconstructed "
            "from the notebook's output columns (drop='first'). For exact fidelity, "
            "supply the original encoder.pkl."
        )
        encoded_df = _manual_one_hot(df)

    df_num = df.drop(columns=CATEGORICAL_COLS)
    X_processed = pd.concat([df_num, encoded_df], axis=1)

    # --- 2. Scale continuous numeric columns ---
    if scaler is not None:
        cols = [c for c in NUMERIC_COLS_TO_SCALE if c in X_processed.columns]
        X_processed[cols] = scaler.transform(X_processed[cols])
    else:
        warnings.append(
            "scaler.pkl not found — numeric features (age, avg_glucose_level, bmi) "
            "are being passed through UNSCALED. The training data was standardized, "
            "so predictions will likely be unreliable until the real scaler.pkl is "
            "supplied in /models."
        )

    # --- 3. Match training feature order ---
    X_processed = X_processed.reindex(columns=feature_names, fill_value=0)

    return X_processed, warnings


def _manual_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback one-hot encoding matching the categories/order the fitted
    encoder produced in the notebook (drop='first', alphabetical order)."""
    encoded_cols = {}
    for col in CATEGORICAL_COLS:
        categories = CATEGORY_OPTIONS[col]
        dropped, kept = categories[0], categories[1:]
        for cat in kept:
            colname = f"{col}_{cat}"
            encoded_cols[colname] = (df[col] == cat).astype(float)
    return pd.DataFrame(encoded_cols, index=df.index)


def predict_tabular(model, X_processed: pd.DataFrame):
    """Run inference exactly as in the notebook: predict_proba + threshold."""
    proba = model.predict_proba(X_processed.values)[0]
    stroke_proba = float(proba[1])
    prediction = int(stroke_proba >= TABULAR_THRESHOLD)
    return prediction, stroke_proba, proba


# ---------------------------------------------------------------------------
# CT image preprocessing (mirrors notebook 7 exactly)
# ---------------------------------------------------------------------------

def preprocess_ct_image(pil_image):
    """Resize to 224x224 and apply ResNet50's preprocess_input, exactly as
    in notebook 7 (tf.keras.applications.resnet50.preprocess_input)."""
    from tensorflow.keras.applications.resnet50 import preprocess_input

    img = pil_image.convert("RGB").resize(CT_IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)      # (1, 224, 224, 3)
    arr = preprocess_input(arr)            # caffe-style mean subtraction, BGR order
    return arr


def predict_ct(model, preprocessed_arr):
    """Sigmoid output: probability of class 1 = 'Stroke' (alphabetical class
    order from image_dataset_from_directory, label_mode='binary')."""
    prob = float(model.predict(preprocessed_arr, verbose=0)[0][0])
    prediction = int(prob >= CT_THRESHOLD)
    return prediction, prob


# ---------------------------------------------------------------------------
# Risk tiers & recommendations
# NOTE: These thresholds/recommendation texts are not part of the notebooks —
# the notebooks only produce a probability. They are added here as a
# reasonable, clearly-labelled clinical-communication layer for the app.
# ---------------------------------------------------------------------------

def risk_level(probability: float) -> tuple:
    """Returns (label, color) for a given stroke probability."""
    if probability < 0.30:
        return "Low Risk", "#2ecc71"
    elif probability < 0.60:
        return "Moderate Risk", "#f1c40f"
    elif probability < 0.80:
        return "High Risk", "#e67e22"
    else:
        return "Very High Risk", "#e74c3c"


def recommendation_text(probability: float, prediction: int) -> str:
    label, _ = risk_level(probability)
    if prediction == 0 and probability < 0.30:
        return ("No strong indicators of stroke risk were detected. Maintain a healthy "
                "lifestyle (balanced diet, regular exercise, routine check-ups) and monitor "
                "blood pressure and glucose periodically.")
    elif label == "Moderate Risk":
        return ("Some risk factors are present. Consider scheduling a check-up with a "
                "physician to review blood pressure, glucose, cholesterol, and lifestyle "
                "factors, and discuss preventive measures.")
    elif label == "High Risk":
        return ("Multiple significant risk factors detected. It is strongly recommended to "
                "consult a physician or neurologist soon for a comprehensive stroke risk "
                "assessment and possible diagnostic workup.")
    else:
        return ("The model indicates a very high likelihood of stroke risk / findings. "
                "Seek prompt medical evaluation. If experiencing sudden numbness, confusion, "
                "trouble speaking, vision loss, or severe headache, treat this as a medical "
                "emergency and call local emergency services immediately.")
