# Presentation Content

## Slide 1: Title

**CareSync AI: Hospital Readmission Risk Scorer**

A recall-focused MLOps system for predicting 30-day readmission risk in diabetic patients.

## Slide 2: Problem

Hospital readmissions are costly and harmful. Diabetic patients may require extra support after discharge, but hospitals need a way to identify high-risk patients early.

## Slide 3: Why False Negatives Matter

A false negative means a high-risk patient is predicted as low risk. This can cause missed follow-up, missed medication review, and poor post-discharge care. Therefore, recall is prioritized over accuracy alone.

## Slide 4: Dataset

Dataset: Diabetes 130-US Hospitals for Years 1999-2008.  
Target: `readmitted`.  
`<30` becomes positive class.  
`NO` and `>30` become negative class.

## Slide 5: System Architecture

Raw data flows through cleaning, feature engineering, train/test split, model training, MLflow tracking, threshold tuning, saved artifacts, FastAPI deployment, and Streamlit dashboard.

## Slide 6: Preprocessing

- Replace `?` with missing values
- Drop ID columns
- Convert target to binary
- Impute numerical and categorical columns
- One-hot encode categorical variables
- Use robust preprocessing pipeline

## Slide 7: Feature Engineering

Created features:

- Total prior visits
- Medication change flag
- Diabetes medication flag
- High utilization flag
- Long stay flag
- Medications per hospital day

## Slide 8: Models Used

- Logistic Regression with balanced class weights
- Random Forest with balanced class weights
- Gradient Boosting
- Optional XGBoost if installed

## Slide 9: Metrics

Evaluation focuses on:

- Recall
- Precision
- F1 score
- ROC-AUC
- PR-AUC
- False negatives
- Confusion matrix

## Slide 10: Threshold Tuning

Instead of using the default 0.50 threshold, the system evaluates thresholds from 0.10 to 0.90. Lowering the threshold helps catch more high-risk patients and reduce false negatives.

## Slide 11: API Demo

FastAPI endpoint:

```text
POST /predict
```

The API returns probability, risk level, prediction, explanation, recommendation, and clinical disclaimer.

## Slide 12: Streamlit Demo

The Streamlit dashboard allows a user to enter patient details, click predict, and view risk probability, risk category, explanation, and care recommendation.

## Slide 13: MLOps Tracking

MLflow is used to log:

- Parameters
- Metrics
- Trained models
- Model comparison
- Final best model

## Slide 14: Limitations

- Historical dataset
- No external validation
- Possible dataset bias
- Not suitable for clinical use without validation
- Threshold tuning may increase false positives

## Slide 15: Future Enhancements

- SHAP explanations
- Model monitoring
- Data drift detection
- CI/CD
- Clinician feedback loop
- Database logging
- Real-world validation

## Slide 16: Conclusion

CareSync AI demonstrates a complete clinical AI MLOps pipeline that focuses on the metric that matters most for this use case: catching high-risk patients by improving recall and reducing false negatives.
