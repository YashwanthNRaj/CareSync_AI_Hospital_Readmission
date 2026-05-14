# API Documentation

## Run the API

From the project root:

```powershell
uvicorn api.main:app --reload
```

Open interactive API docs:

```text
http://127.0.0.1:8000/docs
```

## Base URL

```text
http://127.0.0.1:8000
```

---

## GET /

Returns basic API information.

### Response

```json
{
  "message": "CareGuard AI Hospital Readmission Risk Scorer API",
  "docs": "/docs",
  "clinical_disclaimer": "..."
}
```

---

## GET /health

Checks service and artifact availability.

### Response

```json
{
  "status": "ok",
  "model_available": true,
  "threshold_available": true
}
```

---

## POST /predict

Predicts risk for one patient.

### Request Format

The API accepts flexible patient fields because the dataset has many columns.

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

You can also send:

```json
{
  "data": {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[60-70)",
    "time_in_hospital": 5
  }
}
```

### Response Format

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

## POST /predict_batch

Predicts risk for multiple patients.

### Request Format

```json
{
  "patients": [
    {
      "race": "Caucasian",
      "gender": "Female",
      "age": "[60-70)",
      "time_in_hospital": 5,
      "num_medications": 18,
      "number_inpatient": 2,
      "change": "Ch",
      "diabetesMed": "Yes"
    },
    {
      "race": "AfricanAmerican",
      "gender": "Male",
      "age": "[70-80)",
      "time_in_hospital": 3,
      "num_medications": 10,
      "number_inpatient": 0,
      "change": "No",
      "diabetesMed": "Yes"
    }
  ]
}
```

### Response Format

```json
{
  "count": 2,
  "predictions": [
    {
      "risk_probability": 0.78,
      "risk_level": "High Risk",
      "prediction": "Readmitted within 30 days",
      "threshold_used": 0.35,
      "recommendation": "...",
      "explanation": "...",
      "clinical_disclaimer": "..."
    }
  ],
  "clinical_disclaimer": "..."
}
```

---

## Error Handling

### Missing Model

If the model is not trained yet:

```json
{
  "detail": "Trained model was not found at models/careguard_readmission_model.joblib. Run `python src/train.py` before prediction."
}
```

### Empty Input

```json
{
  "detail": "Patient input cannot be empty."
}
```

### Batch Empty Input

```json
{
  "detail": "patients list cannot be empty."
}
```

## Clinical Disclaimer

All API responses include a clinical disclaimer. This project is for educational demonstration and not for real clinical decision-making.
