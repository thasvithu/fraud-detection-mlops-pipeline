"""Model loading and prediction service helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from src.data_ingestion import EXPECTED_COLUMNS

DEFAULT_MODEL_PATH = Path("models/model.pkl")
DEFAULT_PREPROCESSOR_PATH = Path("models/preprocessor.pkl")
DEFAULT_TRAINING_REPORT_PATH = Path("artifacts/model_training_report.json")
DEFAULT_MODEL_REPORT_PATH = Path("artifacts/model_report.json")
DEFAULT_CONFIG_PATH = Path("configs/train.yaml")
FEATURE_COLUMNS = [column for column in EXPECTED_COLUMNS if column != "Class"]


@dataclass
class InferenceService:
    """Encapsulate model/preprocessor runtime and prediction logic."""

    model: Any
    preprocessor: Any
    threshold: float
    model_path: Path
    preprocessor_path: Path
    feature_columns: list[str]

    def predict_records(self, records: list[dict[str, float]]) -> list[dict[str, Any]]:
        """Predict fraud labels/probabilities for input transaction records."""
        frame = pd.DataFrame(records)
        frame = frame[self.feature_columns]

        transformed = self.preprocessor.transform(frame)
        probabilities = self.model.predict_proba(transformed)[:, 1]

        outputs: list[dict[str, Any]] = []
        for prob in probabilities:
            probability = float(prob)
            outputs.append(
                {
                    "is_fraud": bool(probability >= self.threshold),
                    "fraud_probability": probability,
                    "risk_level": _risk_level(probability),
                    "threshold": float(self.threshold),
                }
            )
        return outputs


def _risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.3:
        return "medium"
    return "low"


def _threshold_from_training_report(training_report_path: Path) -> float | None:
    if not training_report_path.exists():
        return None
    payload = json.loads(training_report_path.read_text(encoding="utf-8"))
    best = payload.get("best_model", {})
    threshold = best.get("selected_threshold")
    return float(threshold) if threshold is not None else None


def _threshold_from_model_report(model_report_path: Path) -> float | None:
    if not model_report_path.exists():
        return None
    payload = json.loads(model_report_path.read_text(encoding="utf-8"))
    selection = payload.get("threshold_selection", {})
    threshold = selection.get("selected_threshold")
    return float(threshold) if threshold is not None else None


def _threshold_from_config(config_path: Path) -> float | None:
    if not config_path.exists():
        return None
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    threshold_cfg = config.get("threshold", {})
    threshold = threshold_cfg.get("decision_threshold")
    return float(threshold) if threshold is not None else None


def resolve_threshold(
    *,
    training_report_path: Path = DEFAULT_TRAINING_REPORT_PATH,
    model_report_path: Path = DEFAULT_MODEL_REPORT_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> float:
    """Resolve runtime threshold from artifacts, then fallback config/default."""
    value = _threshold_from_training_report(training_report_path)
    if value is not None:
        return value
    value = _threshold_from_model_report(model_report_path)
    if value is not None:
        return value
    value = _threshold_from_config(config_path)
    if value is not None:
        return value
    return 0.5


@lru_cache(maxsize=1)
def load_inference_service(
    *,
    model_path: str = str(DEFAULT_MODEL_PATH),
    preprocessor_path: str = str(DEFAULT_PREPROCESSOR_PATH),
    training_report_path: str = str(DEFAULT_TRAINING_REPORT_PATH),
    model_report_path: str = str(DEFAULT_MODEL_REPORT_PATH),
    config_path: str = str(DEFAULT_CONFIG_PATH),
) -> InferenceService:
    """Load model + preprocessor + threshold and cache service singleton."""
    model_file = Path(model_path)
    preprocessor_file = Path(preprocessor_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_file}")
    if not preprocessor_file.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {preprocessor_file}")

    model = joblib.load(model_file)
    preprocessor = joblib.load(preprocessor_file)
    threshold = resolve_threshold(
        training_report_path=Path(training_report_path),
        model_report_path=Path(model_report_path),
        config_path=Path(config_path),
    )

    feature_names_in = getattr(preprocessor, "feature_names_in_", FEATURE_COLUMNS)
    feature_columns = list(feature_names_in)

    return InferenceService(
        model=model,
        preprocessor=preprocessor,
        threshold=threshold,
        model_path=model_file,
        preprocessor_path=preprocessor_file,
        feature_columns=feature_columns,
    )
