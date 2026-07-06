from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH, THRESHOLD_PATH
from src.feature_engineering import add_clinical_features

CLINICAL_DISCLAIMER = (
    "This prediction is for decision support and educational demonstration only. "
    "It is not a diagnosis and must not replace clinician judgment."
)


def load_model(path: Path | None = None):
    """Load the trained model artifact."""
    candidates = [
        path or MODEL_PATH,
        PROJECT_ROOT / "models" / "careguard_readmission_model.joblib",
        PROJECT_ROOT / "models" / "caresync_readmission_model.joblib",
    ]

    for candidate in candidates:
        if candidate.exists():
            return joblib.load(candidate)

    raise FileNotFoundError(
        "Trained model was not found. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )

def load_threshold(path: Path = THRESHOLD_PATH) -> float:
    """Load the tuned threshold. If missing, fall back to 0.50."""
    if not path.exists():
        return 0.50
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload.get("threshold", 0.50))


def _risk_level(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    if probability <= 0.60:
        return "Medium Risk"
    return "High Risk"


def _recommendation(risk_level: str) -> str:
    if risk_level == "High Risk":
        return (
            "Schedule early follow-up within 7 days, medication review, "
            "discharge counseling, and care coordinator call."
        )
    if risk_level == "Medium Risk":
        return "Schedule follow-up within 14 days, monitor glucose control and medication adherence."
    return "Continue standard discharge plan and routine follow-up."


def _align_columns_to_training_data(model, patient_df: pd.DataFrame) -> pd.DataFrame:
    """
    The trained sklearn pipeline remembers the columns used during training.
    During API prediction, users may send only some fields.
    This function adds missing columns safely so prediction does not crash.
    """

    expected_columns = None

    if hasattr(model, "feature_names_in_"):
        expected_columns = list(model.feature_names_in_)

    elif hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]
        if hasattr(preprocessor, "feature_names_in_"):
            expected_columns = list(preprocessor.feature_names_in_)

    if expected_columns is None:
        return patient_df

    for column in expected_columns:
        if column not in patient_df.columns:
            patient_df[column] = np.nan

    patient_df = patient_df[expected_columns]
    return patient_df


def predict_readmission(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Predict 30-day readmission risk for one patient dictionary."""
    model = load_model()
    threshold = load_threshold()

    patient_df = pd.DataFrame([patient_data])
    patient_df = patient_df.replace("?", np.nan)

    # Add engineered features used during training.
    patient_df = add_clinical_features(patient_df)

    # Add missing training columns automatically.
    patient_df = _align_columns_to_training_data(model, patient_df)

    probability = float(model.predict_proba(patient_df)[:, 1][0])
    risk_level = _risk_level(probability)
    predicted_positive = probability >= threshold

    return {
        "risk_probability": round(probability, 4),
        "risk_level": risk_level,
        "prediction": "Readmitted within 30 days"
        if predicted_positive
        else "Not readmitted within 30 days",
        "threshold_used": round(float(threshold), 4),
        "recommendation": _recommendation(risk_level),
        "explanation": (
            "The patient shows elevated readmission risk based on hospital utilization, "
            "length of stay, medications, and prior visits."
            if risk_level == "High Risk"
            else "The estimated risk is based on hospital utilization, length of stay, medications, and prior visits."
        ),
        "clinical_disclaimer": CLINICAL_DISCLAIMER,
    }