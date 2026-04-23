from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

UNKNOWN = -1


@dataclass
class InterventionResult:
    per_concept_rescue_rate: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_concept_sample_counts: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_k_rescue: pd.DataFrame = field(default_factory=pd.DataFrame)


def _predict(cbm, X: np.ndarray, threshold: float) -> np.ndarray:
    if hasattr(cbm, "predict_proba"):
        return (cbm.predict_proba(X)[:, 1] >= threshold).astype(int)
    return cbm.predict(X).astype(int)


def simulate_intervention(
    cbm,
    concept_scores: np.ndarray,
    concept_labels: np.ndarray,
    diagnosis_true: np.ndarray,
    fitzpatrick_bucket: np.ndarray,
    concept_names: Sequence[str],
    buckets: Sequence[str] = ("light", "medium", "dark"),
    threshold: float = 0.5,
    max_k: int = 3,
    unknown_value: int = UNKNOWN,
) -> InterventionResult:
    C_scores = np.asarray(concept_scores, dtype=np.float32).copy()
    C_labels = np.asarray(concept_labels, dtype=np.float32)
    y_true = np.asarray(diagnosis_true).astype(int)
    buckets_arr = np.asarray(fitzpatrick_bucket, dtype=object)

    y_pred = _predict(cbm, C_scores, threshold)
    mis_mask = y_pred != y_true
    n_concepts = C_scores.shape[1]

    rescue_counts = {b: np.zeros(n_concepts, dtype=np.int64) for b in buckets}
    sample_counts = {b: 0 for b in buckets}
    topk_rescued = {b: np.zeros(max_k + 1, dtype=np.int64) for b in buckets}

    # single-concept intervention
    for i in np.where(mis_mask)[0]:
        bucket = buckets_arr[i]
        if bucket not in buckets:
            continue
        sample_counts[bucket] += 1
        base_vec = C_scores[i:i + 1].copy()
        target = y_true[i]

        # Try each concept in isolation
        single_rescue = np.zeros(n_concepts, dtype=bool)
        for j in range(n_concepts):
            gt = C_labels[i, j]
            if gt == unknown_value:
                continue
            probe = base_vec.copy()
            probe[0, j] = 0.99 if gt == 1 else 0.01
            new_pred = _predict(cbm, probe, threshold)[0]
            if new_pred == target:
                rescue_counts[bucket][j] += 1
                single_rescue[j] = True

        # top-k joint intervention
        # Greedy: start from an all-VLM vector, repeatedly pick the concept
        # whose correction flips closer to the target class probability.
        current = base_vec.copy()
        corrected_mask = np.zeros(n_concepts, dtype=bool)
        for k in range(1, max_k + 1):
            best_idx, best_prob = None, -np.inf if target == 1 else np.inf
            for j in range(n_concepts):
                if corrected_mask[j]:
                    continue
                gt = C_labels[i, j]
                if gt == unknown_value:
                    continue
                probe = current.copy()
                probe[0, j] = 0.99 if gt == 1 else 0.01
                prob = cbm.predict_proba(probe)[0, 1] if hasattr(cbm, "predict_proba") \
                    else float(cbm.predict(probe)[0])
                if target == 1 and prob > best_prob:
                    best_prob, best_idx = prob, j
                elif target == 0 and prob < best_prob:
                    best_prob, best_idx = prob, j
            if best_idx is None:
                break
            gt = C_labels[i, best_idx]
            current[0, best_idx] = 0.99 if gt == 1 else 0.01
            corrected_mask[best_idx] = True
            new_pred = _predict(cbm, current, threshold)[0]
            if new_pred == target:
                topk_rescued[bucket][k] += 1

    # shape into DataFrames
    per_concept = pd.DataFrame({
        b: rescue_counts[b] / max(sample_counts[b], 1)
        for b in buckets
    }, index=list(concept_names))
    per_concept["overall_n_misclassified"] = [sum(sample_counts.values())] * n_concepts

    counts_df = pd.DataFrame({
        "n_misclassified": [sample_counts[b] for b in buckets]
    }, index=list(buckets))

    topk_rows = []
    for b in buckets:
        if sample_counts[b] == 0:
            continue
        row = {"bucket": b, "n_misclassified": sample_counts[b]}
        for k in range(1, max_k + 1):
            row[f"rescue_rate@k={k}"] = topk_rescued[b][k] / sample_counts[b]
        topk_rows.append(row)
    top_k_df = pd.DataFrame(topk_rows).set_index("bucket") if topk_rows else pd.DataFrame()

    return InterventionResult(
        per_concept_rescue_rate=per_concept,
        per_concept_sample_counts=counts_df,
        top_k_rescue=top_k_df,
    )
