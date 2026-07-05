from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from api.schemas import BatchPredictionRequest, PatientInput, PredictionResponse
from src.config import MODEL_PATH, THRESHOLD_PATH
from src.predict import CLINICAL_DISCLAIMER, predict_readmission

app = FastAPI(
    title="CareSync AI: Hospital Readmission Risk Scorer",
    description="Recall-focused diabetic patient 30-day readmission risk scoring API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY", "caresync-secure-key-2026")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

def get_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API key")
    return api_key


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "CareSync AI Hospital Readmission Risk Scorer API",
        "docs": "/docs",
        "clinical_disclaimer": CLINICAL_DISCLAIMER,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
        "threshold_available": THRESHOLD_PATH.exists(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput, api_key: str = Depends(get_api_key)) -> Dict[str, Any]:
    try:
        patient_dict = patient.to_feature_dict()
        if not patient_dict:
            raise HTTPException(status_code=400, detail="Patient input cannot be empty.")
        return predict_readmission(patient_dict)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict_batch")
def predict_batch(request: BatchPredictionRequest, api_key: str = Depends(get_api_key)) -> Dict[str, Any]:
    if not request.patients:
        raise HTTPException(status_code=400, detail="patients list cannot be empty.")

    predictions: List[Dict[str, Any]] = []
    try:
        for patient in request.patients:
            predictions.append(predict_readmission(patient))
        return {
            "count": len(predictions),
            "predictions": predictions,
            "clinical_disclaimer": CLINICAL_DISCLAIMER,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc
