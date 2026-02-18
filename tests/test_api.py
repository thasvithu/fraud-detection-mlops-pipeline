from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.app import app, get_inference_service


class DummyService:
    threshold = 0.74
    model_path = Path("models/model.pkl")
    preprocessor_path = Path("models/preprocessor.pkl")

    def predict_records(self, records):
        outputs = []
        for record in records:
            amount = float(record["Amount"])
            prob = 0.9 if amount > 200 else 0.1
            outputs.append(
                {
                    "is_fraud": prob >= self.threshold,
                    "fraud_probability": prob,
                    "risk_level": "high" if prob >= 0.7 else "low",
                    "threshold": self.threshold,
                }
            )
        return outputs


def _transaction(amount: float = 10.0) -> dict[str, float]:
    payload = {"Time": 0.0, "Amount": amount}
    for i in range(1, 29):
        payload[f"V{i}"] = 0.0
    return payload


def test_health_endpoint() -> None:
    app.dependency_overrides[get_inference_service] = lambda: DummyService()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    app.dependency_overrides.clear()


def test_predict_endpoint_valid_payload() -> None:
    app.dependency_overrides[get_inference_service] = lambda: DummyService()
    client = TestClient(app)

    response = client.post("/predict", json=_transaction(amount=350.0))

    assert response.status_code == 200
    body = response.json()
    assert body["is_fraud"] is True
    assert body["risk_level"] == "high"
    assert response.headers.get("X-Request-ID")
    app.dependency_overrides.clear()


def test_predict_endpoint_invalid_payload() -> None:
    app.dependency_overrides[get_inference_service] = lambda: DummyService()
    client = TestClient(app)

    payload = _transaction()
    payload.pop("V28")
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    app.dependency_overrides.clear()


def test_batch_prediction_endpoint() -> None:
    app.dependency_overrides[get_inference_service] = lambda: DummyService()
    client = TestClient(app)

    response = client.post(
        "/predict/batch",
        json={"transactions": [_transaction(20.0), _transaction(300.0)]},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["is_fraud"] is False
    assert body["predictions"][1]["is_fraud"] is True
    app.dependency_overrides.clear()


def test_metrics_endpoint_tracks_predictions_and_requests() -> None:
    app.dependency_overrides[get_inference_service] = lambda: DummyService()
    client = TestClient(app)

    before = client.get("/metrics")
    assert before.status_code == 200
    before_body = before.json()

    predict_response = client.post("/predict", json=_transaction(amount=350.0))
    assert predict_response.status_code == 200

    after = client.get("/metrics")
    assert after.status_code == 200
    after_body = after.json()

    assert after_body["total_requests"] >= before_body["total_requests"] + 2
    assert after_body["total_predictions"] >= before_body["total_predictions"] + 1
    assert 0.0 <= after_body["error_rate"] <= 1.0
    assert 0.0 <= after_body["fraud_prediction_rate"] <= 1.0
    app.dependency_overrides.clear()


def test_health_returns_503_when_service_unavailable() -> None:
    def _raise():
        raise HTTPException(status_code=503, detail="Model artifact not found")

    app.dependency_overrides[get_inference_service] = _raise
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    assert "Model artifact not found" in response.json()["detail"]
    app.dependency_overrides.clear()
