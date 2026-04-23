from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


# Point-estimate metrics
def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(y_score)))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, np.asarray(y_score)))


def sensitivity_specificity(
    y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1
) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = [1 - positive, positive]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return float(sens), float(spec)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(np.asarray(y_true).astype(int),
                                         np.asarray(y_pred).astype(int)))


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)


# Bootstrap
def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = metric(y_true, y_score)
    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            stats[i] = metric(y_true[idx], y_score[idx])
        except ValueError:
            stats[i] = np.nan
    lo, hi = np.nanquantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


# Per-concept AUROC helper (Stage 4 Level 1)
def per_concept_auroc(
    concept_scores: np.ndarray,
    concept_labels: np.ndarray,
    concept_names: List[str],
    unknown_value: int = -1,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, name in enumerate(concept_names):
        labels = concept_labels[:, i]
        mask = labels != unknown_value
        if mask.sum() == 0:
            out[name] = float("nan")
            continue
        out[name] = auroc(labels[mask].astype(int), concept_scores[mask, i])
    return out


# Aggregate helper
def summarise(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> Dict[str, float]:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    sens, spec = sensitivity_specificity(y_true, y_pred)
    return {
        "auroc": auroc(y_true, y_prob),
        "auprc": auprc(y_true, y_prob),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "n": int(len(y_true)),
        "prevalence": float(np.mean(np.asarray(y_true).astype(int))),
    }
