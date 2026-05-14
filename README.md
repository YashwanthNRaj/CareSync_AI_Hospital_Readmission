# CareGuard AI: Hospital Readmission Risk Scorer

CareGuard AI is a complete end-to-end MLOps project that predicts whether a diabetic patient is likely to be readmitted within 30 days after hospital discharge.

This project is designed for a hackathon-style clinical AI mission. It includes data preprocessing, feature engineering, model training, MLflow experiment tracking, recall-focused threshold tuning, saved model artifacts, a FastAPI risk scoring service, a Streamlit demo dashboard, tests, Docker support, documentation, and presentation content.

> **Clinical disclaimer:** This project is for educational and hackathon demonstration purposes only. It is not a medical device and must not be used as a substitute for professional clinical judgment.

---

## Project Overview

Hospital readmission is expensive, stressful for patients, and often preventable with better follow-up planning. For diabetic patients, early identification of high-risk discharge cases can help care teams schedule follow-ups, review medications, and coordinate post-discharge support.

The project predicts:

- **1:** Patient is readmitted within 30 days
- **0:** Patient is not readmitted within 30 days

The positive class is:

```text
Readmitted within 30 days
```

---

## Why False Negatives Matter

In this problem, a false negative means the model predicts that a patient is low risk, but the patient is actually readmitted within 30 days.

That is more dangerous than a false positive because a high-risk patient may miss extra care, follow-up, medication review, and discharge counseling.

Because of this, CareGuard AI prioritizes:

- Recall / sensitivity for the positive class
- Reduction of false negatives
- Threshold tuning instead of using only the default 0.50 threshold

---

## Dataset

Dataset used:

**Diabetes 130-US Hospitals for Years 1999-2008**  
UCI Machine Learning Repository

Dataset page:

```text
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
```

The dataset is not included in this ZIP.

Place the file manually here:

```text
data/raw/diabetic_data.csv
```

Target column:

```text
readmitted
```

Target conversion:

```text
readmitted == "<30"  -> 1
readmitted == "NO"   -> 0
readmitted == ">30"  -> 0
```

---

## Project Architecture

```text
Raw Dataset
   ↓
Data Cleaning + Missing Value Handling
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Model Training + MLflow Tracking
   ↓
Threshold Tuning
   ↓
Saved Model Artifact
   ↓
FastAPI Risk Scoring API
   ↓
Streamlit Demo Dashboard
   ↓
Risk Explanation + Care Recommendation
```

---

## Folder Structure

```text
CareGuard_AI_Hospital_Readmission/
├── README.md
├── requirements.txt
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── notebooks/
├── src/
├── api/
├── app/
├── models/
├── reports/
├── docs/
└── tests/
```

---

## Windows PowerShell Setup

From inside the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

## Dataset Placement

Download the dataset from UCI and place it here:

```text
CareGuard_AI_Hospital_Readmission/data/raw/diabetic_data.csv
```

Do not rename the file.

---

## Train the Model

```powershell
python src/train.py
```

Training will:

- Load the raw dataset
- Replace `?` values with missing values
- Create the binary target
- Add engineered clinical utilization features
- Train multiple models
- Track experiments with MLflow
- Tune the decision threshold for recall
- Save model artifacts and evaluation reports

Saved outputs:

```text
models/careguard_readmission_model.joblib
models/threshold.json
reports/metrics.json
reports/figures/confusion_matrix.png
reports/figures/roc_curve.png
reports/figures/pr_curve.png
```

---

## MLflow Tracking

Start the MLflow UI:

```powershell
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

---

## Run FastAPI

```powershell
uvicorn api.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

---

## Run Streamlit Dashboard

```powershell
streamlit run app/streamlit_app.py
```

---

## API Sample Request

Endpoint:

```text
POST /predict
```

Sample JSON:

```json
{
  "race": "Caucasian",
  "gender": "Female",
  "age": "[60-70)",
  "admission_type_id": 1,
  "discharge_disposition_id": 1,
  "admission_source_id": 7,
  "time_in_hospital": 5,
  "num_lab_procedures": 45,
  "num_procedures": 1,
  "num_medications": 18,
  "number_outpatient": 1,
  "number_emergency": 0,
  "number_inpatient": 2,
  "diag_1": "250.83",
  "diag_2": "401",
  "diag_3": "428",
  "number_diagnoses": 8,
  "max_glu_serum": "None",
  "A1Cresult": ">8",
  "metformin": "No",
  "insulin": "Up",
  "change": "Ch",
  "diabetesMed": "Yes"
}
```

## API Sample Response

```json
{
  "risk_probability": 0.78,
  "risk_level": "High Risk",
  "prediction": "Readmitted within 30 days",
  "threshold_used": 0.35,
  "recommendation": "Schedule early follow-up within 7 days, medication review, discharge counseling, and care coordinator call.",
  "explanation": "The patient shows elevated readmission risk based on hospital utilization, length of stay, medications, and prior visits.",
  "clinical_disclaimer": "This prediction is for decision support and educational demonstration only. It is not a diagnosis and must not replace clinician judgment."
}
```

---

## Model Evaluation

The training pipeline evaluates models using:

- Recall
- Precision
- F1 score
- ROC-AUC
- PR-AUC / average precision
- Confusion matrix
- False negatives
- False positives

Accuracy is not the main optimization target because hospital readmission is an imbalanced and clinically sensitive prediction task.

---

## Threshold Tuning Explanation

Most binary classifiers use `0.50` as the default decision threshold. In this project, that may miss too many high-risk patients.

CareGuard AI evaluates thresholds from `0.10` to `0.90` and selects a recall-focused threshold that reduces false negatives while keeping precision reasonable.

Clinical logic:

```text
In this clinical context, we lower the threshold to catch more at-risk patients, reducing false negatives.
```

---

## MLOps Components

This project includes:

- Reproducible training script
- Central config file using `pathlib`
- Saved model artifact with `joblib`
- Saved threshold artifact
- MLflow experiment tracking
- Evaluation reports and plots
- API deployment with FastAPI
- Demo dashboard with Streamlit
- Docker deployment support
- Basic pytest tests

---

## Docker Usage

Build and run the API:

```powershell
docker compose up --build
```

API will run at:

```text
http://127.0.0.1:8000
```

Before using prediction endpoints in Docker, train the model locally or mount trained model artifacts into the `models/` folder.

---

## Limitations

- The dataset is historical and may not represent current hospital workflows.
- The model does not replace clinical judgment.
- The dataset contains coded diagnosis fields that may need domain interpretation.
- Threshold tuning improves recall but can increase false positives.
- External validation is required before any real clinical use.

---

## Future Scope

- Add SHAP explanations for feature-level interpretability
- Add model monitoring and drift detection
- Add automated retraining pipeline
- Add CI/CD using GitHub Actions
- Add authentication for API usage
- Add database logging of predictions
- Add clinician feedback loop
- Validate on modern hospital data

---

## Clinical Disclaimer

CareGuard AI is a hackathon and educational project. It is not intended to diagnose, treat, cure, or prevent any disease. Predictions must be reviewed by qualified healthcare professionals before any clinical action is taken.
