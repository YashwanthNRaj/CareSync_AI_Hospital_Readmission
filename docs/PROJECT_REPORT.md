# CareSync AI: Hospital Readmission Risk Scorer - Project Report

## Abstract

CareSync AI is an end-to-end MLOps project that predicts whether a diabetic patient is likely to be readmitted within 30 days of hospital discharge. The project uses the Diabetes 130-US Hospitals dataset and focuses on recall because false negatives can be clinically harmful. The system includes data preprocessing, missing value handling, feature engineering, model training, MLflow tracking, threshold tuning, FastAPI deployment, Streamlit dashboarding, documentation, Docker support, and tests.

## Introduction

Hospital readmission is an important healthcare quality and cost issue. Diabetic patients are especially vulnerable after discharge because medication changes, glucose control, comorbidities, and prior hospital utilization can influence readmission risk. A predictive risk scoring tool can help teams prioritize follow-up and discharge support.

## Problem Statement

The goal is to build a binary classification system that predicts whether a diabetic patient will be readmitted within 30 days.

The target is derived from the `readmitted` column:

- `<30` is converted to `1`
- `NO` and `>30` are converted to `0`

The positive class is readmission within 30 days.

## Dataset Description

The project uses the Diabetes 130-US Hospitals for Years 1999-2008 dataset from the UCI Machine Learning Repository. It contains hospital encounter records with demographic information, admission details, diagnoses, procedures, medications, lab results, and readmission labels.

The dataset is not included in the project ZIP. The user must place `diabetic_data.csv` inside `data/raw/`.

## Data Preprocessing

Preprocessing includes:

1. Loading the raw CSV file
2. Replacing `?` values with missing values
3. Dropping extremely sparse columns when needed
4. Converting the readmission target into a binary label
5. Dropping ID columns such as `encounter_id` and `patient_nbr`
6. Splitting features and target
7. Imputing missing numeric values using the median
8. Imputing missing categorical values using the most frequent value
9. One-hot encoding categorical variables with unknown-category handling

The preprocessing code is robust to optional missing columns.

## Feature Engineering

The project adds clinically meaningful features:

- `total_visits`: Sum of inpatient, outpatient, and emergency visits
- `medication_change_flag`: Whether medications changed during the encounter
- `diabetes_med_flag`: Whether diabetes medication was prescribed
- `high_utilization_flag`: Whether prior utilization is elevated
- `long_stay_flag`: Whether length of stay is long
- `num_medications_per_day`: Medication burden normalized by hospital stay length

These features are designed to capture utilization, discharge complexity, and treatment intensity.

## Model Selection

The training pipeline evaluates multiple models:

- Logistic Regression with balanced class weights
- Random Forest with balanced class weights
- Gradient Boosting Classifier
- Optional XGBoost if installed

Each model is trained using a pipeline that includes preprocessing and oversampling. The best model is selected primarily using recall, then F1 score and PR-AUC for balance.

## Evaluation Metrics

The model is evaluated using:

- Recall
- Precision
- F1 score
- ROC-AUC
- Average precision / PR-AUC
- Confusion matrix
- False negatives
- False positives

Accuracy is intentionally not the main optimization metric because the positive class is clinically important and may be imbalanced.

## Why Recall Matters

A false negative occurs when the model predicts that a patient is not at risk but the patient is actually readmitted within 30 days. This is dangerous because the patient may not receive extra discharge planning or follow-up care.

A false positive may lead to extra monitoring, which is less harmful than missing a truly high-risk patient. Therefore, recall is prioritized.

## Threshold Tuning

The project evaluates decision thresholds from 0.10 to 0.90. Lowering the threshold can increase recall and reduce false negatives.

Clinical threshold logic:

> In this clinical context, we lower the threshold to catch more at-risk patients, reducing false negatives.

The selected threshold is saved in `models/threshold.json`.

## MLOps Pipeline

The MLOps flow includes:

1. Reproducible training script
2. Central configuration using `pathlib`
3. MLflow experiment tracking
4. Saved model artifact
5. Saved threshold artifact
6. Saved metrics and plots
7. FastAPI deployment
8. Streamlit dashboard
9. Docker support
10. Pytest tests

## API Deployment

The API is built using FastAPI. It supports:

- `GET /`
- `GET /health`
- `POST /predict`
- `POST /predict_batch`

The API returns risk probability, risk level, prediction, threshold used, explanation, recommendation, and clinical disclaimer.

## Results

After training, results are saved in:

```text
reports/metrics.json
reports/figures/confusion_matrix.png
reports/figures/roc_curve.png
reports/figures/pr_curve.png
```

The exact metrics depend on the dataset version and trained model.

## Limitations

- Dataset is historical and may not reflect modern care practices.
- The model has not been externally validated.
- The dataset may contain social and clinical bias.
- Diagnosis codes may need domain-specific interpretation.
- Lower thresholds may increase false positives.

## Future Scope

- Add SHAP explanations
- Add monitoring and drift detection
- Add CI/CD pipeline
- Add model registry workflow
- Add database logging
- Add authentication
- Validate on modern hospital data
- Add clinician feedback loop

## Ethical Considerations

Healthcare AI systems can influence care access and prioritization. Any real clinical use would require fairness testing, clinical validation, privacy review, and continuous monitoring. The model should support clinicians, not replace them.

## Clinical Disclaimer

CareSync AI is for educational and hackathon demonstration only. It is not a medical device and must not be used as a substitute for professional clinical judgment.
