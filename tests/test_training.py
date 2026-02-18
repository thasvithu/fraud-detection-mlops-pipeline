from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from src.evaluate import rank_models
from src.train import run_training_pipeline, train_single_model


def _synthetic_df(rows: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    data: dict[str, np.ndarray] = {
        "Time": rng.normal(loc=1000, scale=250, size=rows),
        "Amount": rng.normal(loc=80, scale=20, size=rows),
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(size=rows)

    y = np.zeros(rows, dtype=int)
    fraud_indices = rng.choice(rows, size=max(8, rows // 20), replace=False)
    y[fraud_indices] = 1

    # Inject weak signal for separability.
    data["Amount"][fraud_indices] += 40
    data["V3"][fraud_indices] += 1.5
    data["Class"] = y
    return pd.DataFrame(data)


def test_rank_models_orders_by_recall_then_precision() -> None:
    ranked = rank_models(
        [
            {"model_name": "a", "metrics": {"recall": 0.8, "precision": 0.9, "roc_auc": 0.9}},
            {"model_name": "b", "metrics": {"recall": 0.9, "precision": 0.7, "roc_auc": 0.95}},
            {"model_name": "c", "metrics": {"recall": 0.9, "precision": 0.8, "roc_auc": 0.85}},
        ]
    )
    assert [entry["model_name"] for entry in ranked] == ["c", "b", "a"]


def test_train_single_model_returns_expected_metrics() -> None:
    df = _synthetic_df(200)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Simple split for unit test.
    X_train, X_test = X.iloc[:160], X.iloc[160:]
    y_train, y_test = y.iloc[:160], y.iloc[160:]

    _, metrics = train_single_model(
        model_name="logistic_regression",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=42,
    )

    assert set(metrics.keys()) == {"precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix"}
    assert 0.0 <= metrics["recall"] <= 1.0


def test_run_training_pipeline_creates_report_and_model(tmp_path) -> None:
    df = _synthetic_df(240)
    data_path = tmp_path / "creditcard.csv"
    config_path = tmp_path / "train.yaml"
    model_path = tmp_path / "best_model.pkl"
    preprocessor_path = tmp_path / "preprocessor.pkl"
    report_path = tmp_path / "training_report.json"
    model_report_path = tmp_path / "model_report.json"
    validation_report_path = tmp_path / "data_validation.json"

    df.to_csv(data_path, index=False)

    config = {
        "experiment": {"name": "test-experiment"},
        "training": {
            "test_size": 0.2,
            "random_state": 42,
            "imbalance_method": "class_weight",
            "models": ["logistic_regression"],
        },
        "mlflow": {"tracking_uri": f"file:{tmp_path / 'mlruns'}"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report = run_training_pipeline(
        config_path=config_path,
        data_path=data_path,
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        report_path=report_path,
        model_report_path=model_report_path,
        validation_report_path=validation_report_path,
    )

    assert model_path.exists()
    assert preprocessor_path.exists()
    assert report_path.exists()
    assert model_report_path.exists()
    assert validation_report_path.exists()
    assert report["best_model"]["model_name"] == "logistic_regression"
    assert 0.0 < report["best_model"]["selected_threshold"] < 1.0

    stored = json.loads(report_path.read_text(encoding="utf-8"))
    assert stored["best_model"]["run_id"]
