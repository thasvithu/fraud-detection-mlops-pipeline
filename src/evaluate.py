"""Model evaluation utilities."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(y_true, y_pred, y_pred_proba) -> dict[str, Any]:
    """Calculate classification metrics used for model comparison."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
        "confusion_matrix": cm.tolist(),
    }


def rank_models(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort candidate model results by recall, then precision, then roc_auc."""
    return sorted(
        results,
        key=lambda r: (r["metrics"]["recall"], r["metrics"]["precision"], r["metrics"]["roc_auc"]),
        reverse=True,
    )
