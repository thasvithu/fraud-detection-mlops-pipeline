"""Model evaluation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_roc_auc(y_true, y_pred_proba) -> float:
    try:
        return float(roc_auc_score(y_true, y_pred_proba))
    except ValueError:
        return float("nan")


def _safe_pr_auc(y_true, y_pred_proba) -> float:
    try:
        return float(average_precision_score(y_true, y_pred_proba))
    except ValueError:
        return float("nan")


def calculate_metrics(y_true, y_pred, y_pred_proba) -> dict[str, Any]:
    """Calculate classification metrics used for model comparison."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_pred_proba),
        "pr_auc": _safe_pr_auc(y_true, y_pred_proba),
        "confusion_matrix": cm.tolist(),
    }


def rank_models(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort candidate model results by recall, then precision, then roc_auc."""
    return sorted(
        results,
        key=lambda r: (r["metrics"]["recall"], r["metrics"]["precision"], r["metrics"]["roc_auc"]),
        reverse=True,
    )


def calculate_metrics_at_threshold(
    y_true,
    y_pred_proba,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Compute metrics using a probability threshold."""
    y_pred = (np.asarray(y_pred_proba) >= threshold).astype(int)
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    metrics["threshold"] = float(threshold)
    return metrics


def evaluate_thresholds(
    y_true,
    y_pred_proba,
    *,
    thresholds: list[float] | None = None,
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    grid_size: int = 99,
) -> list[dict[str, Any]]:
    """Evaluate model metrics across threshold grid."""
    if thresholds is None:
        thresholds = np.linspace(min_threshold, max_threshold, grid_size).tolist()
    return [
        calculate_metrics_at_threshold(y_true, y_pred_proba, threshold=t)
        for t in thresholds
    ]


def select_best_threshold(
    y_true,
    y_pred_proba,
    *,
    min_recall: float = 0.90,
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    grid_size: int = 99,
) -> dict[str, Any]:
    """Select threshold by maximizing precision while meeting recall target."""
    evaluations = evaluate_thresholds(
        y_true,
        y_pred_proba,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        grid_size=grid_size,
    )

    feasible = [m for m in evaluations if m["recall"] >= min_recall]
    search_space = feasible if feasible else evaluations
    selection_reason = "meets_min_recall" if feasible else "fallback_max_recall"

    best = sorted(
        search_space,
        key=lambda m: (m["precision"], m["f1"], m["recall"]),
        reverse=True,
    )[0]

    return {
        "selection_reason": selection_reason,
        "min_recall_target": float(min_recall),
        "selected_threshold": float(best["threshold"]),
        "selected_metrics": best,
        "threshold_grid_size": int(grid_size),
        "thresholds_evaluated": evaluations,
    }
