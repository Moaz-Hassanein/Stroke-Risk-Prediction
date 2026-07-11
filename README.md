# 🧠 Stroke Risk Prediction

A two-track machine learning project for stroke risk assessment: **(1)** a tabular clinical-data pipeline that predicts stroke risk from patient health records, and **(2)** a deep learning pipeline that classifies brain CT scans as *Normal* or *Stroke* using a fine-tuned ResNet50, with explainability provided via LIME (tabular) and Grad-CAM (imaging).

---

## 📌 Project Overview

Stroke is a leading cause of death and long-term disability worldwide, and early risk identification is critical for timely intervention. This project approaches the problem from two complementary angles:

| Track | Input | Goal |
|---|---|---|
| **Clinical / Tabular** | Patient demographic & health records | Predict stroke risk from structured clinical features |
| **Medical Imaging / CT** | Brain CT scan images | Classify CT scans as *Normal* or *Stroke* |

Both pipelines go beyond raw prediction to focus on **model interpretability** — a critical requirement in healthcare AI — using LIME for the tabular model and Grad-CAM for the CNN-based image classifier.

---

## 📂 Datasets

| Dataset | Description | Source |
|---|---|---|
| **Stroke Prediction Dataset** | 5,110 patient records with demographic, lifestyle, and clinical attributes (age, hypertension, heart disease, glucose level, BMI, smoking status, etc.) and a binary `stroke` target | [Kaggle — fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) |
| **Brain Stroke CT Scan Image Dataset** | Brain CT scan images organized into `Train` / `Validation` / `Test` splits, each with `Normal` and `Stroke` classes (650×650 grayscale images) | [Kaggle — iashiqul/brain-stroke-prediction-ct-scan-image-dataset](https://www.kaggle.com/datasets/iashiqul/brain-stroke-prediction-ct-scan-image-dataset) |

> Datasets are not included in this repository due to size and licensing. Download them from the links above and place them under a local `Data/` directory (see [Project Structure](#-project-structure)).

---

## 🤗 Pretrained Models

The trained artifacts for this project (fine-tuned ResNet50 CT classifier and/or supporting model files) are published on Hugging Face:

🔗 **[Moaz-Hassanein/stroke-risk-prediction](https://huggingface.co/Moaz-Hassanein/stroke-risk-prediction)**

---

## 🗂 Project Structure

```
Stroke-Risk-Prediction/
│
├── Data/
│   ├── healthcare-dataset-stroke-data.csv     # Raw tabular dataset
│   ├── stroke_data_cleaned.csv                # Cleaned/preprocessed tabular dataset
│   └── Brain_Stroke_CT-SCAN_image/            # CT image dataset (Train/Validation/Test)
│
├── Models/
│   ├── scaler.pkl                             # StandardScaler for numeric features
│   ├── encoder.pkl                            # OneHotEncoder for categorical features
│   ├── feature_names.pkl                      # Saved feature order for inference
│   ├── logistic_regression_model.pkl          # Final tabular model
│   ├── best_resnet50_finetuned.keras          # Final CT image classification model
│   └── resnet50_baseline.keras                # Baseline (frozen-backbone) CT model
│
├── Notebooks/
│   ├── 1_EDA.ipynb                            # Tabular data exploration
│   ├── 2_Data_Cleaning.ipynb                  # Tabular data cleaning & preprocessing
│   ├── 3_Data_Analysis.ipynb                  # Statistical analysis of risk factors
│   ├── 4_Model_Development.ipynb              # Tabular model training & comparison
│   ├── 5_LIME_Explainability.ipynb            # LIME explainability for tabular model
│   ├── 6_CT_EDA.ipynb                         # CT image dataset exploration
│   ├── 7_CT_Model_Development.ipynb           # ResNet50 transfer learning & fine-tuning
│   └── 8_GradCam_Explainability_CT_Images.ipynb  # Grad-CAM explainability for CT model
│
└── README.md
```

---

## 🧪 Track 1 — Clinical / Tabular Stroke Prediction

### 1️⃣ Exploratory Data Analysis (`1_EDA.ipynb`)
- 5,110 records, 12 attributes (numerical + categorical), no duplicates.
- Missing values only in `bmi` (201 missing, ~3.9%).
- Outlier analysis (IQR) on `age`, `avg_glucose_level`, `bmi` — the latter two show notable outliers.
- Correlation and distribution analysis identify `age`, `heart_disease`, and `avg_glucose_level` as the features most associated with stroke.

### 2️⃣ Data Cleaning (`2_Data_Cleaning.ipynb`)
- Dropped the non-informative `id` column.
- Imputed missing `bmi` values with the **median** (robust to skew/outliers).
- Capped outliers in `avg_glucose_level` and `bmi` using the **IQR method**.
- Standardized continuous features (`age`, `avg_glucose_level`, `bmi`) via **Z-score scaling** (`StandardScaler`, saved as `scaler.pkl`).
- Exported the cleaned dataset to `stroke_data_cleaned.csv`.

### 3️⃣ Statistical Data Analysis (`3_Data_Analysis.ipynb`)
- Confirmed strong **class imbalance** (~4.9% stroke cases, ~20:1 ratio).
- **t-test / Mann-Whitney U** tests show `age`, `avg_glucose_level`, and `bmi` differ significantly between stroke/non-stroke groups.
- **Chi-square tests** show `hypertension`, `heart_disease`, and `ever_married` have the strongest categorical associations with stroke.
- Combined risk-factor analysis shows a compounding effect: patients with **both** hypertension and heart disease have markedly higher stroke rates.
- Key predictors identified: **age, heart_disease, avg_glucose_level, ever_married, hypertension.**

### 4️⃣ Model Development (`4_Model_Development.ipynb`)
- One-hot encoding of categorical features (`encoder.pkl`), feature order saved (`feature_names.pkl`).
- Class imbalance addressed with **SMOTE** oversampling on the training set.
- Five models trained and tuned via `GridSearchCV` (F1-optimized): **XGBoost, Random Forest, AdaBoost, Linear Discriminant Analysis (LDA), Logistic Regression.**
- Models compared on accuracy, precision, recall, F1, and ROC-AUC, with **recall prioritized** since missed stroke cases (false negatives) carry high clinical cost.

**Result summary:**

| Model | Accuracy | Recall | F1-Score | Notes |
|---|---|---|---|---|
| Random Forest | 91.5% | 16% | Low | High accuracy, but misses most stroke cases |
| XGBoost | 85.1% | Improved | Low | Better recall than RF, still limited |
| AdaBoost | 88.1% | Improved | Improved | Similar limitation |
| LDA | 73.7% | 80% | 22.9% | High recall, lower accuracy |
| **Logistic Regression (Final)** | 75.3% | **80%** | **0.241 (highest)** | Best balance for clinical priority on recall |

> **Final model:** Logistic Regression was selected as the production model, prioritizing recall (sensitivity) to minimize missed stroke diagnoses — the most clinically important failure mode — while retaining a competitive F1-score. Saved as `logistic_regression_model.pkl`.

### 5️⃣ Explainability — LIME (`5_LIME_Explainability.ipynb`)
- Uses **LIME (Local Interpretable Model-agnostic Explanations)** to explain individual predictions of the Logistic Regression model.
- Loads the saved encoder, scaler, and feature names to reproduce the exact training-time preprocessing pipeline before generating explanations.
- For each patient instance, LIME identifies which features pushed the prediction toward *Stroke* vs. *Normal* and by how much, visualized as a feature-contribution bar chart.
- Age consistently emerges as the dominant driver of stroke-risk predictions, consistent with the statistical analysis in notebook 3.

---

## 🩻 Track 2 — CT Brain Image Classification

### 6️⃣ CT Image EDA (`6_CT_EDA.ipynb`)
- Dataset organized into `Train` / `Validation` / `Test`, each with `Normal` and `Stroke` classes.
- Moderate class imbalance (more `Normal` than `Stroke` samples).
- All images: **650×650**, single-channel **grayscale**.
- Integrity check confirms **no corrupted images** across all splits.

### 7️⃣ CT Model Development (`7_CT_Model_Development.ipynb`)
- Images loaded via `image_dataset_from_directory`, resized to **224×224**, normalized with ResNet50's `preprocess_input`.
- **Data augmentation** (horizontal flip, small rotation, zoom, translation) applied to the training set only.
- **Transfer learning** with a pre-trained **ResNet50** (ImageNet weights):
  - **Phase 1 — Baseline:** convolutional base frozen; custom head (`GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)`) trained for binary classification.
  - **Phase 2 — Fine-tuning:** last 30 layers of the ResNet50 backbone unfrozen and retrained with a much smaller learning rate (`1e-5`) to adapt ImageNet features to CT imagery.
- Training used `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau` callbacks.

**Baseline vs. Fine-tuned performance (Test set):**

| Metric | Baseline | Fine-Tuned | Improvement |
|---|---|---|---|
| Accuracy | 81.24% | **86.96%** | +5.72 |
| Precision | 65.19% | **73.25%** | +8.06 |
| Recall | 79.23% | **88.46%** | +9.23 |
| AUC | 88.62% | **94.17%** | +5.55 |
| False Negatives (missed strokes) | 27 | **15** | −12 |

> **Final model:** the fine-tuned ResNet50 (`best_resnet50_finetuned.keras`) correctly classified 115 of 130 stroke cases in the test set, cutting false negatives nearly in half compared to the frozen-backbone baseline — a clinically significant improvement given the cost of missed stroke diagnoses.

### 8️⃣ Explainability — Grad-CAM (`8_GradCam_Explainability_CT_Images.ipynb`)
- Implements **Grad-CAM** (Gradient-weighted Class Activation Mapping) on the fine-tuned ResNet50 to visualize which regions of a CT scan drove the model's prediction.
- Builds a feature-extraction model from the `resnet50` backbone's last convolutional layer (`conv5_block3_out`) and re-applies the model's own classification head inside a `GradientTape`, computing gradients of the predicted class w.r.t. those feature maps — required for compatibility with **Keras 3's** functional-graph construction rules (see implementation note below).
- The resulting heatmap is overlaid on the original CT image to highlight influential regions (warm colors = high importance).

**Finding:** the model correctly attends to portions of the head, but in several cases also relies on skull boundaries and image edges rather than brain tissue exclusively — indicating room to improve localization quality despite strong classification metrics.

**Planned follow-up:** upgrading to **Grad-CAM++** (via `tf-keras-vis`) or **Score-CAM**, and/or adding brain-region extraction / background removal as a preprocessing step, to better concentrate model attention on clinically relevant brain tissue.

> **Keras 3 compatibility note:** In Keras 3, mixing an inner submodel's intermediate output with the outer model's final output inside a single `tf.keras.Model(...)` call raises a `KeyError` due to changes in nested submodel graph-merging behavior (vs. Keras 2). The workaround used here builds the Grad-CAM feature extractor directly from the backbone's input/output, then manually re-applies the trained head layers (`GlobalAveragePooling2D → Dense → Dropout → Dense`) inside the `GradientTape`, preserving the exact fine-tuned weights without reconstructing or reloading any part of the model.

---

## 🛠 Tech Stack

- **Language:** Python 3.13
- **Tabular ML:** `scikit-learn`, `XGBoost`, `imbalanced-learn` (SMOTE), `LIME`
- **Deep Learning:** `TensorFlow` / `Keras 3`, `ResNet50` (transfer learning)
- **Explainability:** LIME (tabular), Grad-CAM (imaging)
- **Data/Viz:** `pandas`, `NumPy`, `matplotlib`, `seaborn`, `OpenCV`, `Pillow`
- **Model persistence:** `joblib`, Keras `.keras` format

---

## ⚙️ Setup & Usage

```bash
# Clone the repository
git clone <repo-url>
cd Stroke-Risk-Prediction

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn \
            lime tensorflow opencv-python pillow joblib
```

1. Download both datasets from the Kaggle links above and place them in `Data/` following the structure shown in [Project Structure](#-project-structure).
2. Run the notebooks in numerical order (`1` → `8`). The tabular pipeline (1–5) and CT pipeline (6–8) are independent and can be run separately.
3. Trained artifacts (scaler, encoder, feature names, and models) are saved to `Models/` and reused by downstream notebooks (e.g., LIME and Grad-CAM notebooks load these artifacts to ensure inference matches training).

---

## 🔮 Future Work

- Replace Grad-CAM with **Grad-CAM++** or **Score-CAM** for sharper, more anatomically-focused CT heatmaps.
- Add brain-region extraction / skull-stripping as a CT preprocessing step to reduce background/skull activation.
- Explore ensemble or multimodal approaches combining tabular clinical risk with CT imaging findings.
- Deploy both models behind a unified inference API, using the artifacts published on [Hugging Face](https://huggingface.co/Moaz-Hassanein/stroke-risk-prediction).

---

## 📄 License & Disclaimer

This project is for **educational and research purposes only** and is **not intended for clinical or diagnostic use**. Predictions from these models should never substitute professional medical judgment.

## 🚀 Live Demo

Try the deployed application here:

**🔗 https://moaz-hassanein-stroke-risk-prediction-app-9gaexv.streamlit.app/**
