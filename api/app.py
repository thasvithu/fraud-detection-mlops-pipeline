from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
    Transaction,
)
from api.service import InferenceService, load_inference_service

logger = logging.getLogger("api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


@dataclass
class MonitoringState:
    total_requests: int = 0
    error_count: int = 0
    total_predictions: int = 0
    fraud_predictions: int = 0
    total_latency_ms: float = 0.0
    _lock: Lock = field(default_factory=Lock)

    def record_request(self, *, latency_ms: float, status_code: int) -> None:
        with self._lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            if status_code >= 400:
                self.error_count += 1

    def record_predictions(self, predictions: list[dict[str, object]]) -> None:
        fraud_count = sum(1 for p in predictions if bool(p.get("is_fraud")))
        with self._lock:
            self.total_predictions += len(predictions)
            self.fraud_predictions += fraud_count

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            avg_latency = self.total_latency_ms / self.total_requests if self.total_requests else 0.0
            error_rate = self.error_count / self.total_requests if self.total_requests else 0.0
            fraud_rate = (
                self.fraud_predictions / self.total_predictions if self.total_predictions else 0.0
            )
            return {
                "total_requests": self.total_requests,
                "error_count": self.error_count,
                "error_rate": float(error_rate),
                "total_predictions": self.total_predictions,
                "fraud_predictions": self.fraud_predictions,
                "fraud_prediction_rate": float(fraud_rate),
                "avg_latency_ms": float(avg_latency),
            }


app = FastAPI(title="Fraud Detection API", version="0.3.0")
monitoring_state = MonitoringState()


@app.middleware("http")
async def add_observability(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    start = time.perf_counter()

    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000
        monitoring_state.record_request(latency_ms=latency_ms, status_code=status_code)
        logger.exception(
            json.dumps(
                {
                    "event": "request_error",
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "latency_ms": round(latency_ms, 2),
                }
            )
        )
        raise

    latency_ms = (time.perf_counter() - start) * 1000
    monitoring_state.record_request(latency_ms=latency_ms, status_code=status_code)

    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
    response.headers["X-Request-ID"] = request_id

    logger.info(
        json.dumps(
            {
                "event": "request_complete",
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 2),
            }
        )
    )
    return response


def get_inference_service() -> InferenceService:
    try:
        return load_inference_service()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


ServiceDep = Annotated[InferenceService, Depends(get_inference_service)]


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health(service: ServiceDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_path=str(service.model_path),
        preprocessor_path=str(service.preprocessor_path),
        threshold=service.threshold,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(**monitoring_state.snapshot())


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction, service: ServiceDep) -> PredictionResponse:
    output = service.predict_records([transaction.model_dump()])[0]
    monitoring_state.record_predictions([output])
    logger.info(
        json.dumps(
            {
                "event": "prediction",
                "prediction_count": 1,
                "fraud_predictions": int(output["is_fraud"]),
                "avg_probability": round(float(output["fraud_probability"]), 6),
                "threshold": float(output["threshold"]),
            }
        )
    )
    return PredictionResponse(**output)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest, service: ServiceDep) -> BatchPredictionResponse:
    predictions = service.predict_records([record.model_dump() for record in request.transactions])
    monitoring_state.record_predictions(predictions)

    fraud_count = sum(1 for row in predictions if row["is_fraud"])
    avg_probability = sum(float(row["fraud_probability"]) for row in predictions) / len(predictions)
    logger.info(
        json.dumps(
            {
                "event": "prediction_batch",
                "prediction_count": len(predictions),
                "fraud_predictions": fraud_count,
                "avg_probability": round(avg_probability, 6),
                "threshold": float(predictions[0]["threshold"]),
            }
        )
    )

    return BatchPredictionResponse(predictions=[PredictionResponse(**row) for row in predictions])
