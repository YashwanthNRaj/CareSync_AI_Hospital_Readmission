from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(y_true, y_proba, threshold: float = 0.50) -> Dict[str, Any]:
    """Calculate classification metrics at a chosen threshold."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        roc_auc = None

    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except ValueError:
        pr_auc = None

    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "average_precision_pr_auc": pr_auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist(),
    }


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """Save metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def plot_confusion_matrix(y_true, y_proba, threshold: float, path: Path, title: Optional[str] = None) -> None:
    """Save a confusion matrix heatmap."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=["Not <30", "<30"],
        yticklabels=["Not <30", "<30"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title or f"Confusion Matrix at threshold {threshold:.2f}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_proba, path: Path) -> None:
    """Save ROC curve."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_value = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Recall")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pr_curve(y_true, y_proba, path: Path) -> None:
    """Save precision-recall curve."""
    path.parent.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap_value = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"Average Precision = {ap_value:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def print_classification_report(y_true, y_proba, threshold: float = 0.50) -> str:
    """Print and return a classification report."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Not readmitted <30", "Readmitted <30"],
        zero_division=0,
    )
    print(report)
    return report
