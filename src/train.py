"""Training entrypoint for fraud detection models with MLflow tracking."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

from src.data_ingestion import load_data, run_data_validation
from src.evaluate import calculate_metrics, rank_models, select_best_threshold
from src.preprocessing import preprocess_for_training

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - handled at runtime
    XGBClassifier = None


DEFAULT_CONFIG_PATH = Path("configs/train.yaml")
DEFAULT_DATA_PATH = Path("data/raw/creditcard.csv")
DEFAULT_MODEL_PATH = Path("models/model.pkl")
DEFAULT_PREPROCESSOR_PATH = Path("models/preprocessor.pkl")
DEFAULT_REPORT_PATH = Path("artifacts/model_training_report.json")
DEFAULT_MODEL_REPORT_PATH = Path("artifacts/model_report.json")
DEFAULT_VALIDATION_REPORT_PATH = Path("artifacts/data_validation.json")


def load_training_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML training configuration."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    config.setdefault("experiment", {})
    config.setdefault("training", {})
    config.setdefault("mlflow", {})
    return config


def create_model(model_name: str, random_state: int) -> Any:
    """Create model instance from configured model name."""
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            class_weight="balanced",
            random_state=random_state,
        )

    if model_name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not available in the environment")
        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=2,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def train_single_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    random_state: int,
) -> tuple[Any, dict[str, Any]]:
    """Train one model and return model + metrics."""
    model = create_model(model_name, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    return model, metrics


def log_run_to_mlflow(
    *,
    experiment_name: str,
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    preprocessor_path: Path,
    model_temp_path: Path,
    artifact_dir: Path,
) -> str:
    """Log one training run to MLflow and return run id."""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        metric_values = {k: v for k, v in metrics.items() if isinstance(v, float)}
        mlflow.log_metrics(metric_values)

        # Structured artifacts for debugging and reproducibility.
        metrics_path = artifact_dir / f"metrics_{model_name}.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")
        mlflow.log_artifact(str(model_temp_path), artifact_path="model")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        return run.info.run_id


def save_model(model: Any, output_path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Save model artifact to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def run_training_pipeline(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    preprocessor_path: str | Path = DEFAULT_PREPROCESSOR_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    model_report_path: str | Path = DEFAULT_MODEL_REPORT_PATH,
    validation_report_path: str | Path = DEFAULT_VALIDATION_REPORT_PATH,
) -> dict[str, Any]:
    """Execute end-to-end training and experiment tracking pipeline."""
    config = load_training_config(config_path)

    experiment_name = config["experiment"].get("name", "fraud-detection-baseline")
    tracking_uri = config["mlflow"].get("tracking_uri", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    training_cfg = config["training"]
    random_state = int(training_cfg.get("random_state", 42))
    test_size = float(training_cfg.get("test_size", 0.2))
    imbalance_method = str(training_cfg.get("imbalance_method", "class_weight"))
    models = training_cfg.get("models") or [training_cfg.get("model", "logistic_regression")]
    threshold_cfg = config.get("threshold", {})
    min_recall_target = float(threshold_cfg.get("min_recall_target", 0.90))
    threshold_grid_size = int(threshold_cfg.get("grid_size", 99))
    threshold_min = float(threshold_cfg.get("min_threshold", 0.01))
    threshold_max = float(threshold_cfg.get("max_threshold", 0.99))

    run_data_validation(file_path=data_path, report_path=validation_report_path)
    raw_df = load_data(data_path)
    prep = preprocess_for_training(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        imbalance_method=imbalance_method,
        preprocessor_path=preprocessor_path,
    )

    results: list[dict[str, Any]] = []
    skipped_models: list[dict[str, str]] = []
    artifact_dir = Path(report_path).parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path_obj = Path(preprocessor_path)
    for model_name in models:
        try:
            model, metrics = train_single_model(
                model_name=model_name,
                X_train=prep["X_train"],
                y_train=prep["y_train"],
                X_test=prep["X_test"],
                y_test=prep["y_test"],
                random_state=random_state,
            )
        except RuntimeError as exc:
            skipped_models.append({"model_name": model_name, "reason": str(exc)})
            continue

        temp_model_path = Path(model_path).parent / f"{model_name}.pkl"
        save_model(model, temp_model_path)

        run_id = log_run_to_mlflow(
            experiment_name=experiment_name,
            model_name=model_name,
            params={
                "model_name": model_name,
                "test_size": test_size,
                "random_state": random_state,
                "imbalance_method": imbalance_method,
            },
            metrics=metrics,
            preprocessor_path=preprocessor_path_obj,
            model_temp_path=temp_model_path,
            artifact_dir=artifact_dir,
        )

        results.append({"model_name": model_name, "model": model, "metrics": metrics, "run_id": run_id})

    if not results:
        raise RuntimeError("No models were successfully trained.")

    ranked = rank_models(results)
    best = ranked[0]
    y_test_proba_best = best["model"].predict_proba(prep["X_test"])[:, 1]
    threshold_selection = select_best_threshold(
        prep["y_test"],
        y_test_proba_best,
        min_recall=min_recall_target,
        min_threshold=threshold_min,
        max_threshold=threshold_max,
        grid_size=threshold_grid_size,
    )

    model_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "best_model_name": best["model_name"],
        "default_threshold_metrics": best["metrics"],
        "threshold_selection": threshold_selection,
        "evaluation_summary": {
            "test_rows": int(len(prep["y_test"])),
            "min_recall_target": min_recall_target,
            "selection_reason": threshold_selection["selection_reason"],
        },
    }
    model_report_path_obj = Path(model_report_path)
    model_report_path_obj.parent.mkdir(parents=True, exist_ok=True)
    model_report_path_obj.write_text(json.dumps(model_report, indent=2), encoding="utf-8")

    final_model_path = save_model(best["model"], model_path)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_name,
        "tracking_uri": tracking_uri,
        "data_path": str(data_path),
        "preprocessor_path": str(preprocessor_path),
        "model_path": str(final_model_path),
        "model_report_path": str(model_report_path_obj),
        "best_model": {
            "model_name": best["model_name"],
            "run_id": best["run_id"],
            "metrics": best["metrics"],
            "selected_threshold": threshold_selection["selected_threshold"],
            "threshold_metrics": threshold_selection["selected_metrics"],
        },
        "all_results": [
            {"model_name": entry["model_name"], "run_id": entry["run_id"], "metrics": entry["metrics"]}
            for entry in ranked
        ],
        "skipped_models": skipped_models,
    }

    report_path_obj = Path(report_path)
    report_path_obj.parent.mkdir(parents=True, exist_ok=True)
    report_path_obj.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train fraud model and log to MLflow.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Training config YAML path.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Dataset CSV path.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Output model artifact path.")
    parser.add_argument(
        "--preprocessor-path",
        default=str(DEFAULT_PREPROCESSOR_PATH),
        help="Output preprocessor artifact path.",
    )
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH), help="Training report JSON path.")
    parser.add_argument(
        "--model-report-path",
        default=str(DEFAULT_MODEL_REPORT_PATH),
        help="Model evaluation report JSON path.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = run_training_pipeline(
        config_path=args.config,
        data_path=args.data_path,
        model_path=args.model_path,
        preprocessor_path=args.preprocessor_path,
        report_path=args.report_path,
        model_report_path=args.model_report_path,
    )

    best = report["best_model"]
    print("Training completed.")
    print(f"Best model: {best['model_name']}")
    print(f"Selected threshold: {best['selected_threshold']:.4f}")
    print(json.dumps(best["threshold_metrics"], indent=2))


if __name__ == "__main__":
    main()
