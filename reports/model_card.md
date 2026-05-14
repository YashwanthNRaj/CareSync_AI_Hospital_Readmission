# Model Card: CareGuard AI Hospital Readmission Risk Scorer

## Model Name

CareGuard AI Hospital Readmission Risk Scorer

## Intended Use

The model is intended for educational and hackathon demonstration use. It predicts whether a diabetic patient may be readmitted within 30 days after hospital discharge.

The output can support a demo workflow where high-risk patients receive additional follow-up recommendations.

## Not Intended Use

This model is not intended for real clinical decision-making, diagnosis, treatment planning, insurance decisions, or replacing clinician judgment.

## Dataset

Diabetes 130-US Hospitals for Years 1999-2008 from the UCI Machine Learning Repository.

## Target

Binary target derived from `readmitted`:

- `<30` → 1
- `NO` or `>30` → 0

Positive class: readmitted within 30 days.

## Metrics

The training pipeline saves the final metrics in:

```text
reports/metrics.json
```

Tracked metrics include:

- Recall
- Precision
- F1 score
- ROC-AUC
- PR-AUC / average precision
- Confusion matrix
- False negatives
- False positives

## Threshold

The tuned decision threshold is saved in:

```text
models/threshold.json
```

The threshold is tuned to prioritize recall and reduce false negatives.

## Limitations

- Historical dataset may not represent current clinical practice.
- Diagnosis codes and treatment patterns may vary by institution.
- The model requires external validation before real-world use.
- Lowering the threshold may increase false positives.
- The model may reflect biases present in the original dataset.

## Ethical Risks

- Potential demographic bias
- Over-reliance on automated predictions
- Incorrect care prioritization if used without clinical review
- Data drift if deployed on modern hospital populations

## Clinical Disclaimer

This project is for educational demonstration only. It is not a medical device and must not be used as a substitute for qualified healthcare professionals.
