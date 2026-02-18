from __future__ import annotations

import time
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
    Transaction,
)
from api.service import InferenceService, load_inference_service

app = FastAPI(title="Fraud Detection API", version="0.2.0")


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
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


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction, service: ServiceDep) -> PredictionResponse:
    output = service.predict_records([transaction.model_dump()])[0]
    return PredictionResponse(**output)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest, service: ServiceDep) -> BatchPredictionResponse:
    predictions = service.predict_records([record.model_dump() for record in request.transactions])
    return BatchPredictionResponse(predictions=[PredictionResponse(**row) for row in predictions])
