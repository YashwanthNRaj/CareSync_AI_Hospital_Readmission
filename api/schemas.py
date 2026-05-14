from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PatientInput(BaseModel):
    """
    Flexible patient input schema.

    Accepts:
    1. {"patient": {...}}
    2. {"data": {...}}
    3. direct fields like {"race": "Caucasian", "age": "[70-80)"}
    """

    model_config = ConfigDict(extra="allow")

    patient: Optional[Dict[str, Any]] = Field(default=None)
    data: Optional[Dict[str, Any]] = Field(default=None)

    def to_feature_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}

        if self.patient:
            payload.update(self.patient)

        if self.data:
            payload.update(self.data)

        if self.model_extra:
            payload.update(self.model_extra)

        return payload


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_level: str
    prediction: str
    threshold_used: float
    recommendation: str
    explanation: str
    clinical_disclaimer: str


class BatchPredictionRequest(BaseModel):
    patients: List[Dict[str, Any]]