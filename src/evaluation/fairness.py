from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import (
    auprc,
    auroc,
    bootstrap_ci,
    expected_calibration_error,
    per_concept_auroc,
    sensitivity_specificity,
    summarise,
)

logger = logging.getLogger(__name__)

DEFAULT_BUCKETS = ("light", "medium", "dark")


# Level 1: concept-level stratification
def concept_auroc_by_bucket(
    concept_scores: np.ndarray,
    concept_labels: np.ndarray,
    fitzpatrick_bucket: np.ndarray,
    concept_names: Sequence[str],
    buckets: Sequence[str] = DEFAULT_BUCKETS,
    unknown_value: int = -1,
) -> pd.DataFrame:
    rows = []
    for c_idx, c_name in enumerate(concept_names):
        row = {"concept": c_name}
        for b in buckets:
            mask = np.asarray(fitzpatrick_bucket) == b
            if mask.sum() == 0:
                row[b] = float("nan")
                continue
            c_scores = concept_scores[mask, c_idx]
            c_labels = concept_labels[mask, c_idx]
            known = c_labels != unknown_value
            if known.sum() == 0 or len(np.unique(c_labels[known])) < 2:
                row[b] = float("nan")
                continue
            row[b] = auroc(c_labels[known].astype(int), c_scores[known])
        rows.append(row)
    return pd.DataFrame(rows).set_index("concept")


# Level 2: diagnosis-level stratification
@dataclass
class DiagnosisFairnessReport:
    per_bucket: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_bucket_with_ci: pd.DataFrame = field(default_factory=pd.DataFrame)
    equalized_odds_gap: float = float("nan")
    demographic_parity_gap: float = float("nan")
    overall: Dict[str, float] = field(default_factory=dict)


def diagnosis_fairness_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fitzpatrick_bucket: np.ndarray,
    buckets: Sequence[str] = DEFAULT_BUCKETS,
    threshold: float = 0.5,
    n_bins: int = 10,
    bootstrap_resamples: int = 1000,
    alpha: float = 0.05,
) -> DiagnosisFairnessReport:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    buckets_arr = np.asarray(fitzpatrick_bucket, dtype=object)

    per_bucket_rows: List[Dict[str, Any]] = []
    ci_rows: List[Dict[str, Any]] = []
    tpr_by_bucket: Dict[str, float] = {}
    pos_rate_by_bucket: Dict[str, float] = {}

    for b in buckets:
        mask = buckets_arr == b
        n = int(mask.sum())
        row = {"bucket": b, "n": n,
               "prevalence": float(np.mean(y_true[mask])) if n else float("nan")}
        if n == 0:
            row.update({"auroc": float("nan"), "auprc": float("nan"),
                        "sensitivity": float("nan"), "specificity": float("nan"),
                        "ece": float("nan"), "positive_rate": float("nan")})
            per_bucket_rows.append(row)
            ci_rows.append({**row})
            continue

        yt, yp, yprob = y_true[mask], y_pred[mask], y_prob[mask]
        auroc_val = auroc(yt, yprob)
        auprc_val = auprc(yt, yprob)
        sens, spec = sensitivity_specificity(yt, yp)
        ece = expected_calibration_error(yt, yprob, n_bins=n_bins)
        pos_rate = float(np.mean(yp))

        tpr_by_bucket[b] = sens
        pos_rate_by_bucket[b] = pos_rate

        row.update({"auroc": auroc_val, "auprc": auprc_val,
                    "sensitivity": sens, "specificity": spec,
                    "ece": ece, "positive_rate": pos_rate})
        per_bucket_rows.append(row)

        # Bootstrap CIs (AUROC + sensitivity)
        _, auroc_lo, auroc_hi = bootstrap_ci(
            yt, yprob, auroc, n_resamples=bootstrap_resamples, alpha=alpha,
        )
        sens_metric = lambda yt_, yp_: sensitivity_specificity(yt_, (yp_ >= threshold).astype(int))[0]
        _, sens_lo, sens_hi = bootstrap_ci(
            yt, yprob, sens_metric, n_resamples=bootstrap_resamples, alpha=alpha,
        )
        ci_rows.append({**row,
                        "auroc_lo": auroc_lo, "auroc_hi": auroc_hi,
                        "sensitivity_lo": sens_lo, "sensitivity_hi": sens_hi})

    per_bucket_df = pd.DataFrame(per_bucket_rows).set_index("bucket")
    ci_df = pd.DataFrame(ci_rows).set_index("bucket")

    # Equalized odds / demographic parity gaps
    valid_tpr = [v for v in tpr_by_bucket.values() if not np.isnan(v)]
    eo_gap = max(valid_tpr) - min(valid_tpr) if len(valid_tpr) >= 2 else float("nan")
    valid_pr = [v for v in pos_rate_by_bucket.values() if not np.isnan(v)]
    dp_gap = max(valid_pr) - min(valid_pr) if len(valid_pr) >= 2 else float("nan")

    # Try fairlearn for consistency (optional dependency)
    try:
        from fairlearn.metrics import (
            MetricFrame, demographic_parity_difference, equalized_odds_difference,
        )
        eo_fl = float(equalized_odds_difference(y_true, y_pred,
                                                sensitive_features=buckets_arr))
        dp_fl = float(demographic_parity_difference(y_true, y_pred,
                                                    sensitive_features=buckets_arr))
        eo_gap = max(eo_gap, eo_fl) if not np.isnan(eo_gap) else eo_fl
        dp_gap = max(dp_gap, dp_fl) if not np.isnan(dp_gap) else dp_fl
    except Exception as exc:  # pragma: no cover
        logger.debug("fairlearn unavailable: %s", exc)

    overall = summarise(y_true, y_prob, threshold=threshold, n_bins=n_bins)

    return DiagnosisFairnessReport(
        per_bucket=per_bucket_df,
        per_bucket_with_ci=ci_df,
        equalized_odds_gap=float(eo_gap) if eo_gap is not None else float("nan"),
        demographic_parity_gap=float(dp_gap) if dp_gap is not None else float("nan"),
        overall=overall,
    )


# Convenience: full fairness audit
def full_audit(
    concept_scores: Optional[np.ndarray],
    concept_labels: Optional[np.ndarray],
    diagnosis_true: np.ndarray,
    diagnosis_prob: np.ndarray,
    fitzpatrick_bucket: np.ndarray,
    concept_names: Optional[Sequence[str]] = None,
    buckets: Sequence[str] = DEFAULT_BUCKETS,
    bootstrap_resamples: int = 1000,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    if concept_scores is not None and concept_labels is not None and concept_names is not None:
        result["concept_auroc_by_bucket"] = concept_auroc_by_bucket(
            concept_scores=concept_scores,
            concept_labels=concept_labels,
            fitzpatrick_bucket=fitzpatrick_bucket,
            concept_names=concept_names,
            buckets=buckets,
        )
        result["concept_auroc_overall"] = per_concept_auroc(
            concept_scores=concept_scores,
            concept_labels=concept_labels,
            concept_names=list(concept_names),
        )

    dx_report = diagnosis_fairness_report(
        y_true=diagnosis_true,
        y_prob=diagnosis_prob,
        fitzpatrick_bucket=fitzpatrick_bucket,
        buckets=buckets,
        bootstrap_resamples=bootstrap_resamples,
    )
    result["diagnosis_per_bucket"] = dx_report.per_bucket
    result["diagnosis_per_bucket_ci"] = dx_report.per_bucket_with_ci
    result["equalized_odds_gap"] = dx_report.equalized_odds_gap
    result["demographic_parity_gap"] = dx_report.demographic_parity_gap
    result["diagnosis_overall"] = dx_report.overall
    return result
