"""Metric panel: the numbers that actually describe model quality.

Reports macro-F1 (not just accuracy), PR-AUC, ROC-AUC, Brier score and
Expected Calibration Error, plus the confusion matrix. Accuracy alone hides
class-specific failure and over-confidence.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(
    y_true, y_prob, n_bins: int = 10
) -> float:
    """ECE: average |accuracy - confidence| across probability bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_bins(y_true, y_prob, n_bins: int = 10) -> list[dict]:
    """Per-bin accuracy/confidence, for a reliability diagram."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        out.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "count": int(mask.sum()),
                "accuracy": float(y_true[mask].mean()) if mask.any() else None,
                "confidence": float(y_prob[mask].mean()) if mask.any() else None,
            }
        )
    return out


def classification_report_dict(y_true, y_pred, y_prob=None) -> dict:
    """Full metric panel as a JSON-serialisable dict."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    report = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        try:
            report["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            report["roc_auc"] = None
        try:
            report["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            report["pr_auc"] = None
        try:
            report["brier"] = float(brier_score_loss(y_true, y_prob))
        except ValueError:
            report["brier"] = None
        report["ece"] = expected_calibration_error(y_true, y_prob)
    return report


def train_val_gap(train_acc: float, val_acc: float) -> float:
    """Positive gap => train beats val => (candidate) overfitting."""
    return float(train_acc - val_acc)


def best_threshold(y_true, y_prob, n_steps: int = 19) -> tuple[float, float]:
    """Sweep decision thresholds and return (threshold, f1_macro) at the best F1."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, n_steps):
        pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, pred, average="macro", zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), float(f)
    return best_t, best_f


def format_report(name: str, report: dict) -> str:
    """Human-readable block for the CLI."""
    cm = report.get("confusion_matrix") or [[0, 0], [0, 0]]
    lines = [
        f"  {name}",
        f"    n={report['n']}  accuracy={report['accuracy']:.4f}  "
        f"macro-F1={report['f1_macro']:.4f}  (weighted-F1={report['f1_weighted']:.4f})",
        f"    precision_macro={report['precision_macro']:.4f}  "
        f"recall_macro={report['recall_macro']:.4f}",
    ]
    if "pr_auc" in report:
        lines.append(
            f"    ROC-AUC={_fmt(report.get('roc_auc'))}  "
            f"PR-AUC={_fmt(report.get('pr_auc'))}  "
            f"Brier={_fmt(report.get('brier'))}  ECE={_fmt(report.get('ece'))}"
        )
    lines.append(f"    confusion=[[TN={cm[0][0]}, FP={cm[0][1]}], [FN={cm[1][0]}, TP={cm[1][1]}]]")
    return "\n".join(lines)


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"
