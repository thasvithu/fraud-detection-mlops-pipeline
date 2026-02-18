from __future__ import annotations

import numpy as np

from src.evaluate import (
    calculate_metrics_at_threshold,
    evaluate_thresholds,
    select_best_threshold,
)


def test_calculate_metrics_at_threshold_contains_threshold() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = calculate_metrics_at_threshold(y_true, y_prob, threshold=0.5)

    assert metrics["threshold"] == 0.5
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0


def test_evaluate_thresholds_returns_expected_grid_size() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])

    evaluated = evaluate_thresholds(y_true, y_prob, min_threshold=0.1, max_threshold=0.9, grid_size=9)

    assert len(evaluated) == 9
    assert evaluated[0]["threshold"] == 0.1


def test_select_best_threshold_prefers_precision_under_recall_constraint() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.02, 0.15, 0.20, 0.30, 0.55, 0.65, 0.80, 0.95])

    selected = select_best_threshold(
        y_true,
        y_prob,
        min_recall=0.75,
        min_threshold=0.1,
        max_threshold=0.9,
        grid_size=17,
    )

    assert selected["selected_metrics"]["recall"] >= 0.75
    assert 0.1 <= selected["selected_threshold"] <= 0.9
    assert selected["selection_reason"] in {"meets_min_recall", "fallback_max_recall"}
