from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.evaluate import calculate_metrics

THRESHOLD_EXPLANATION = (
    "In this clinical context, we lower the threshold to catch more at-risk patients, "
    "reducing false negatives."
)


def evaluate_thresholds(y_true, y_proba, start: float = 0.10, stop: float = 0.90, step: float = 0.05) -> pd.DataFrame:
    """Evaluate recall, precision, F1, false negatives, and false positives across thresholds."""
    thresholds = np.round(np.arange(start, stop + 1e-9, step), 2)
    rows: List[Dict] = []
    for threshold in thresholds:
        metrics = calculate_metrics(y_true, y_proba, threshold=float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "false_negatives": metrics["false_negatives"],
                "false_positives": metrics["false_positives"],
            }
        )
    return pd.DataFrame(rows)


def select_recall_focused_threshold(threshold_table: pd.DataFrame) -> Tuple[float, Dict]:
    """
    Select a threshold for clinical screening.

    Strategy:
    - Prefer thresholds close to the maximum observed recall.
    - Avoid thresholds with extremely poor precision when possible.
    - Break ties using fewer false negatives and stronger F1.
    """
    if threshold_table.empty:
        raise ValueError("Threshold table is empty.")

    max_recall = threshold_table["recall"].max()
    median_precision = threshold_table["precision"].median()
    precision_floor = max(0.05, median_precision * 0.50)

    candidates = threshold_table[
        (threshold_table["recall"] >= max(0.70, max_recall * 0.95))
        & (threshold_table["precision"] >= precision_floor)
    ].copy()

    if candidates.empty:
        candidates = threshold_table.copy()

    candidates = candidates.sort_values(
        by=["false_negatives", "f1", "precision"],
        ascending=[True, False, False],
    )
    best = candidates.iloc[0].to_dict()
    return float(best["threshold"]), best


def save_threshold(threshold: float, path: Path, details: Dict | None = None) -> None:
    """Save selected decision threshold to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": float(threshold),
        "explanation": THRESHOLD_EXPLANATION,
    }
    if details:
        payload["selection_details"] = details
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
