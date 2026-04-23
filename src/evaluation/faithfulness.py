from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import get_device

# NumPy 2.0 
_trapz = getattr(np, "trapezoid", None) or np.trapz

logger = logging.getLogger(__name__)


# CBM concept-level
def _concept_importance(
    cbm,
    concept_vector: np.ndarray,
) -> np.ndarray:
    n_concepts = concept_vector.shape[-1]
    if hasattr(cbm, "concept_weights"):
        w = np.array(list(cbm.concept_weights().values()))
    elif hasattr(cbm, "model") and hasattr(cbm.model, "coef_"):
        w = cbm.model.coef_.ravel()
    else:
        w = np.ones(n_concepts)
    contrib = np.abs(w * concept_vector.ravel())
    return np.argsort(-contrib)   # most important first


def cbm_deletion_insertion(
    cbm,
    concept_scores: np.ndarray,
    diagnosis_true: np.ndarray,
    steps: Optional[int] = None,
    mask_value: float = 0.0,
) -> Dict[str, np.ndarray]:
    X = np.asarray(concept_scores, dtype=np.float32)
    y = np.asarray(diagnosis_true).astype(int)
    n_samples, n_concepts = X.shape
    steps = steps or n_concepts

    del_curve = np.zeros((n_samples, steps + 1), dtype=np.float32)
    ins_curve = np.zeros((n_samples, steps + 1), dtype=np.float32)

    neutral = np.full_like(X, fill_value=mask_value)

    for i in range(n_samples):
        order = _concept_importance(cbm, X[i:i + 1])
        # Deletion
        vec = X[i].copy()
        del_curve[i, 0] = cbm.predict_proba(vec[None, :])[0, 1]
        for s in range(steps):
            if s < len(order):
                vec[order[s]] = mask_value
            del_curve[i, s + 1] = cbm.predict_proba(vec[None, :])[0, 1]
        # Insertion
        vec = neutral[i].copy()
        ins_curve[i, 0] = cbm.predict_proba(vec[None, :])[0, 1]
        for s in range(steps):
            if s < len(order):
                vec[order[s]] = X[i, order[s]]
            ins_curve[i, s + 1] = cbm.predict_proba(vec[None, :])[0, 1]

    # Trapezoidal AUC over normalised x in [0, 1]
    xs = np.linspace(0, 1, steps + 1)
    del_auc = _trapz(del_curve, xs, axis=1)
    ins_auc = _trapz(ins_curve, xs, axis=1)
    return {
        "deletion_curve": del_curve,
        "insertion_curve": ins_curve,
        "deletion_auc": del_auc,
        "insertion_auc": ins_auc,
        "x": xs,
    }


# Pixel-level (Grad-CAM)
@torch.no_grad()
def _prob_at(model: torch.nn.Module, images: torch.Tensor, target: int = 1) -> np.ndarray:
    logits = model(images)
    return F.softmax(logits, dim=-1)[:, target].cpu().numpy()


def _rank_pixels(heatmap: np.ndarray) -> np.ndarray:
    flat = heatmap.reshape(-1)
    return np.argsort(-flat)


def gradcam_deletion_insertion(
    model: torch.nn.Module,
    images: torch.Tensor,
    heatmaps: np.ndarray,
    steps: int = 100,
    target_class: int = 1,
    mask_value: float = 0.0,
    device: Optional[str | torch.device] = None,
) -> Dict[str, np.ndarray]:
    device = torch.device(device) if device is not None else get_device()
    model = model.to(device).eval()
    images = images.to(device)
    n, c, h, w = images.shape
    px = h * w
    per_step = max(px // steps, 1)

    del_curve = np.zeros((n, steps + 1), dtype=np.float32)
    ins_curve = np.zeros((n, steps + 1), dtype=np.float32)

    neutral = torch.full_like(images, fill_value=mask_value)
    del_state = images.clone()
    ins_state = neutral.clone()

    # Initial probabilities
    del_curve[:, 0] = _prob_at(model, del_state, target_class)
    ins_curve[:, 0] = _prob_at(model, ins_state, target_class)

    for i in range(n):
        order = _rank_pixels(heatmaps[i])        # (px,)
        rows = torch.from_numpy(order // w).to(device)
        cols = torch.from_numpy(order % w).to(device)
        for s in range(steps):
            lo, hi = s * per_step, min((s + 1) * per_step, px)
            r = rows[lo:hi]; cc = cols[lo:hi]
            # Deletion: mask pixels
            del_state[i, :, r, cc] = mask_value
            # Insertion: restore pixels
            ins_state[i, :, r, cc] = images[i, :, r, cc]
            del_curve[i, s + 1] = _prob_at(model, del_state[i:i + 1], target_class)[0]
            ins_curve[i, s + 1] = _prob_at(model, ins_state[i:i + 1], target_class)[0]

    xs = np.linspace(0, 1, steps + 1)
    return {
        "deletion_curve": del_curve,
        "insertion_curve": ins_curve,
        "deletion_auc": _trapz(del_curve, xs, axis=1),
        "insertion_auc": _trapz(ins_curve, xs, axis=1),
        "x": xs,
    }


# Aggregation helper
def summarise_faithfulness(out: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        "mean_deletion_auc": float(np.nanmean(out["deletion_auc"])),
        "mean_insertion_auc": float(np.nanmean(out["insertion_auc"])),
        "median_deletion_auc": float(np.nanmedian(out["deletion_auc"])),
        "median_insertion_auc": float(np.nanmedian(out["insertion_auc"])),
        "faithfulness_score": float(
            np.nanmean(out["insertion_auc"]) - np.nanmean(out["deletion_auc"])
        ),
    }
