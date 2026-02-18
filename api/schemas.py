"""Pydantic request/response schemas for the inference API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    threshold: float


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[Transaction] = Field(min_length=1)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    preprocessor_path: str
    threshold: float
