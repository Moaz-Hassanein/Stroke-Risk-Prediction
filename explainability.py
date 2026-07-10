"""
explainability.py
------------------
LIME explanation pipeline extracted from 5_LIME_Explainability.ipynb.
Explains individual predictions of the tabular Logistic Regression model.
"""

import numpy as np
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

from utils import TABULAR_CLASS_NAMES


@st.cache_resource(show_spinner=False)
def build_lime_explainer(training_data_values: tuple, feature_names: tuple):
    """Build (and cache) the LimeTabularExplainer exactly as in the notebook.
    training_data_values must be a hashable tuple-of-tuples for caching;
    converted back to an ndarray here."""
    data = np.array(training_data_values)
    explainer = LimeTabularExplainer(
        training_data=data,
        feature_names=list(feature_names),
        class_names=TABULAR_CLASS_NAMES,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    return explainer


def explain_instance(explainer, model, sample_row_values: np.ndarray, num_features: int = 10):
    """Mirrors notebook 5's explain_prediction(): returns the LIME
    explanation object plus a (features, weights) list for plotting."""
    explanation = explainer.explain_instance(
        data_row=sample_row_values,
        predict_fn=lambda x: model.predict_proba(x),
        num_features=num_features,
    )
    features, weights = [], []
    for feature, weight in explanation.as_list():
        features.append(feature)
        weights.append(weight)
    return explanation, features, weights
