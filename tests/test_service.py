from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from api.service import InferenceService, load_inference_service, resolve_threshold


class DummyPreprocessor:
    feature_names_in_ = np.array(["Time", *[f"V{i}" for i in range(1, 29)], "Amount"])

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame


class DummyModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probs = []
        for amount in frame["Amount"].tolist():
            if amount >= 300:
                probs.append([0.1, 0.9])
            elif amount >= 100:
                probs.append([0.55, 0.45])
            else:
                probs.append([0.95, 0.05])
        return np.array(probs)


def _record(amount: float) -> dict[str, float]:
    payload = {"Time": 0.0, "Amount": amount}
    for i in range(1, 29):
        payload[f"V{i}"] = 0.0
    return payload


def test_inference_service_predict_records_risk_levels() -> None:
    service = InferenceService(
        model=DummyModel(),
        preprocessor=DummyPreprocessor(),
        threshold=0.5,
        model_path=Path("models/model.pkl"),
        preprocessor_path=Path("models/preprocessor.pkl"),
        feature_columns=["Time", *[f"V{i}" for i in range(1, 29)], "Amount"],
    )

    outputs = service.predict_records([_record(20), _record(120), _record(320)])

    assert outputs[0]["risk_level"] == "low"
    assert outputs[1]["risk_level"] == "medium"
    assert outputs[2]["risk_level"] == "high"
    assert outputs[2]["is_fraud"] is True


def test_resolve_threshold_precedence(tmp_path) -> None:
    training_report = tmp_path / "model_training_report.json"
    model_report = tmp_path / "model_report.json"
    config_path = tmp_path / "train.yaml"

    config_path.write_text("threshold:\n  decision_threshold: 0.51\n", encoding="utf-8")
    model_report.write_text(
        json.dumps({"threshold_selection": {"selected_threshold": 0.63}}), encoding="utf-8"
    )
    training_report.write_text(
        json.dumps({"best_model": {"selected_threshold": 0.74}}), encoding="utf-8"
    )

    threshold = resolve_threshold(
        training_report_path=training_report,
        model_report_path=model_report,
        config_path=config_path,
    )

    assert threshold == 0.74


def test_load_inference_service_reads_artifacts_and_threshold(tmp_path) -> None:
    load_inference_service.cache_clear()

    model_path = tmp_path / "model.pkl"
    preprocessor_path = tmp_path / "preprocessor.pkl"
    training_report = tmp_path / "model_training_report.json"

    joblib.dump(DummyModel(), model_path)
    joblib.dump(DummyPreprocessor(), preprocessor_path)
    training_report.write_text(
        json.dumps({"best_model": {"selected_threshold": 0.66}}), encoding="utf-8"
    )

    service = load_inference_service(
        model_path=str(model_path),
        preprocessor_path=str(preprocessor_path),
        training_report_path=str(training_report),
        model_report_path=str(tmp_path / "missing_model_report.json"),
        config_path=str(tmp_path / "missing_config.yaml"),
    )

    assert service.threshold == 0.66
    outputs = service.predict_records([_record(300.0)])
    assert outputs[0]["is_fraud"] is True
