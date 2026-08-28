"""Shared utilities for anomaly-detection-style defect detection.
No torch dependency here so it can be imported by both the PyTorch
autoencoder and any lighter-weight fallback (e.g. PCA-based)."""

import numpy as np


def choose_threshold(errors: np.ndarray, is_defect: np.ndarray) -> dict:
    """Picks an anomaly threshold from the ROC curve on a labeled
    clean(0)/degraded(1) validation set. Returns the chosen threshold plus
    the ROC-AUC so it can be reported in EVALUATION.md."""
    from sklearn.metrics import roc_auc_score, roc_curve

    auc = roc_auc_score(is_defect, errors)
    fpr, tpr, thresholds = roc_curve(is_defect, errors)
    # Youden's J statistic: maximize (tpr - fpr) for a balanced operating point.
    best_idx = np.argmax(tpr - fpr)
    return {
        "threshold": float(thresholds[best_idx]),
        "roc_auc": float(auc),
        "tpr": float(tpr[best_idx]),
        "fpr": float(fpr[best_idx]),
    }
