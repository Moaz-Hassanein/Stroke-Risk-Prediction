# Stroke Risk AI — Streamlit Application

Built from three notebooks:
- `4_Model_Development.ipynb` — tabular Logistic Regression stroke model
- `5_LIME_Explainability.ipynb` — LIME explanations for the tabular model
- `7_CT_Model_Development.ipynb` — fine-tuned ResNet50 CT scan stroke model

**No model was retrained.** This app only loads saved artifacts and reproduces
the exact preprocessing/inference code from the notebooks.

## 1. Required files (you must supply these)

The notebooks describe training code but the actual saved binary artifacts
were **not included** in the uploaded files. Place these in `models/`:

| File | Produced by | Purpose |
|---|---|---|
| `logistic_regression_model.pkl` | notebook 4 (`joblib.dump(grid_lr.best_estimator_, ...)`) | Tabular stroke model |
| `encoder.pkl` | notebook 4 (`joblib.dump(encoder, "encoder.pkl")`) | OneHotEncoder for categorical fields |
| `feature_names.pkl` | notebook 4 (`joblib.dump(x.columns.tolist(), ...)`) | Exact training feature order |
| `scaler.pkl` | **referenced but not created in the provided notebooks** — see below | Scales age/avg_glucose_level/bmi |
| `best_resnet50_finetuned.keras` (preferred) or `resnet50_baseline.keras` | notebook 7 (`model.save(...)`) | CT scan model |

Place the training dataset in `data/`:

| File | Purpose |
|---|---|
| `stroke_data_cleaned.csv` | Dataset Exploration tab + LIME background distribution |

The app **runs without these files** and will clearly flag what's missing in
the sidebar/Home page and inside each affected page, rather than crashing.

## 2. Missing information (found while analyzing the notebooks)

- **`scaler.pkl` is never fit or saved in `4_Model_Development.ipynb`.**
  `StandardScaler` is imported but unused; the loaded CSV (`stroke_data_cleaned.csv`)
  already contains standardized values for `age`, `avg_glucose_level`, and `bmi`
  (e.g. `age = 1.051434`), meaning scaling happened in an earlier, not-provided
  data-cleaning notebook. `5_LIME_Explainability.ipynb` loads this scaler from
  `../Models/scaler.pkl` — that file exists in the original project but wasn't
  part of your upload. **Without it, numeric inputs are passed through unscaled**
  and predictions will likely be unreliable. The app detects this and shows a
  warning; supply the real `scaler.pkl` for correct results.
- **No trained model binaries were uploaded** (only notebook *code*). The app
  cannot function for real inference until you add the `.pkl`/`.keras` files
  listed above.
- **No live test set / held-out probabilities** were included, so the Model
  Dashboard's CT metrics/confusion matrices use the **static numbers reported
  in the notebook's markdown/output cells** rather than a live computation.
  The tabular dashboard's feature importance *is* computed live from the
  loaded model's coefficients.
- **Risk-level tiers and recommendation text are not part of the notebooks**
  (the notebooks only output a probability). Reasonable clinical-communication
  thresholds were added in `utils.py` (`risk_level()`) and are clearly
  documented there — adjust as needed.
- Class order for the CT model (`Normal`=0, `Stroke`=1) is inferred from
  `image_dataset_from_directory`'s alphabetical labeling with `label_mode="binary"`,
  confirmed by the notebook's printed `Class Names: ['Normal', 'Stroke']`.
- Categorical value options (e.g. `work_type` categories) were reverse-engineered
  from the encoder's output column names in the notebook — if your real
  `encoder.pkl` differs, load it and it will take priority over the app's
  fallback manual encoder.

## 3. Extracted pipeline summary

**Tabular model (Logistic Regression):**
1. Raw fields → one-hot encode `gender`, `ever_married`, `work_type`,
   `Residence_type`, `smoking_status` (`drop="first"`, `handle_unknown="ignore"`)
2. Scale `age`, `avg_glucose_level`, `bmi` with the saved `StandardScaler`
3. Reindex to the saved `feature_names.pkl` order, fill missing with 0
4. `model.predict_proba()` → threshold at **0.5** (selected in the notebook's
   final conclusion) → `Normal` / `Stroke`
5. LIME (`LimeTabularExplainer`, `discretize_continuous=True`) explains each
   prediction against the training distribution.

**CT scan model (ResNet50, fine-tuned):**
1. Resize image to **224×224**
2. Apply `tensorflow.keras.applications.resnet50.preprocess_input`
3. `model.predict()` → sigmoid probability of class **1 = Stroke**
   (classes ordered alphabetically: `Normal`, `Stroke`)
4. Threshold at **0.5** (as used throughout notebook 7)
5. No image-level explainability, per app requirements.

## 4. Install & run

```bash
cd stroke_app
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# add your model/data files to models/ and data/ as described above

streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## 5. Project structure

```
stroke_app/
├── app.py              # Main Streamlit app (all 5 pages)
├── utils.py             # Model loading, exact preprocessing, prediction helpers
├── explainability.py     # LIME pipeline (notebook 5)
├── requirements.txt
├── models/               # <- put .pkl / .keras files here
└── data/                 # <- put stroke_data_cleaned.csv here
```
