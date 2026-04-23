from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

_SINGLE_COL = (3.3, 2.7)
_DOUBLE_COL = (7.0, 3.5)


def _save(fig: plt.Figure, out_dir: str | Path, name: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    return out / f"{name}.pdf"


# Fig 1 : End-to-end pipeline diagram (schematic, using matplotlib boxes)
def plot_pipeline_diagram(out_dir: str | Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Style palettes: CBM track, baseline track, evaluation track.
    cbm_face, cbm_edge = "#DCE6F1", "#2F5496"     # blue
    cnn_face, cnn_edge = "#FCE4D6", "#C65911"     # orange
    eval_face, eval_edge = "#E2EFDA", "#548235"   # green

    def box(x, y, w, h, text, face, edge, fontsize=7.5):
        ax.add_patch(plt.Rectangle((x, y), w, h,
                                   facecolor=face, edgecolor=edge,
                                   linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", fontsize=fontsize)

    def arrow(x0, y0, x1, y1, color="black"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.1))

    # Input
    img_x, img_y, img_w, img_h = 0.02, 0.42, 0.09, 0.20
    box(img_x, img_y, img_w, img_h, "Image\n(derm /\nclinical)",
        "#F2F2F2", "#595959", fontsize=7.5)
    img_right = img_x + img_w
    img_cy = img_y + img_h / 2

    # Top track: CBM path (Stages 1-2)
    top_y, top_h = 0.68, 0.26
    top_cy = top_y + top_h / 2
    s1 = (0.14, top_y, 0.13, top_h)
    c  = (0.30, top_y, 0.13, top_h)
    s2 = (0.46, top_y, 0.13, top_h)
    box(*s1, "BiomedCLIP\n(frozen VLM)\nStage 1", cbm_face, cbm_edge)
    box(*c,  "9-dim\nconcept\nvector",            cbm_face, cbm_edge)
    box(*s2, "CBM-LR /\nCBM-MLP\nStage 2",        cbm_face, cbm_edge)

    # Bottom track: CNN baseline (Stage 3)
    bot_y, bot_h = 0.06, 0.26
    bot_cy = bot_y + bot_h / 2
    s3 = (0.30, bot_y, 0.13, bot_h)
    box(*s3, "EfficientNet-B0\n/ ResNet-50\nStage 3", cnn_face, cnn_edge)

    # Evaluation track: Stages 4, 5, 6
    ev_y, ev_h = 0.37, 0.26
    ev_cy = ev_y + ev_h / 2
    s4 = (0.615, ev_y, 0.12, ev_h)
    s5 = (0.745, ev_y, 0.12, ev_h)
    s6 = (0.875, ev_y, 0.12, ev_h)
    box(*s4, "Fitzpatrick\nfairness\naudit\nStage 4",   eval_face, eval_edge, fontsize=6.5)
    box(*s5, "Concept\nintervention\nStage 5",          eval_face, eval_edge, fontsize=6.5)
    box(*s6, "Faithfulness\n(RISE)\nStage 6\u2605",     eval_face, eval_edge, fontsize=6.5)

    # Arrows: input -> both tracks
    arrow(img_right, img_cy, s1[0],          top_cy)   # image -> S1
    arrow(img_right, img_cy, s3[0],          bot_cy)   # image -> S3

    # Arrows: within top track
    arrow(s1[0] + s1[2], top_cy, c[0],       top_cy)
    arrow(c[0]  + c[2],  top_cy, s2[0],      top_cy)

    # Arrows: tracks -> evaluation 
    arrow(s2[0] + s2[2], top_cy, s4[0],      ev_cy)    # CBM -> S4
    arrow(s3[0] + s3[2], bot_cy, s4[0],      ev_cy)    # CNN -> S4

    # Arrows: within evaluation track
    arrow(s4[0] + s4[2], ev_cy, s5[0],       ev_cy)
    arrow(s5[0] + s5[2], ev_cy, s6[0],       ev_cy)

    # Title 
    ax.text(0.5, 0.965,
            "VLM-CBM Dermatology Pipeline with Fitzpatrick Fairness Audit",
            ha="center", va="center", fontsize=10, fontweight="bold")

    _save(fig, out_dir, "fig1_pipeline")
    return fig


# Fig 2 : Concept-AUROC heatmap by Fitzpatrick bucket (headline)
def plot_concept_auroc_heatmap(
    concept_auroc_df: pd.DataFrame,
    out_dir: str | Path,
    cmap: str = "RdYlGn",
) -> plt.Figure:
    df = concept_auroc_df.copy() if concept_auroc_df is not None else pd.DataFrame()
    # If every cell is NaN (no dataset has both concept GT *and* FST labels),
    # producing a blank heatmap is worse than no figure. Skip cleanly.
    if df.size == 0 or df.apply(pd.to_numeric, errors="coerce").isna().all().all():
        logger.warning(
            "plot_concept_auroc_heatmap: no finite AUROCs (no dataset has "
            "both concept GT and Fitzpatrick labels) - skipping fig2.",
        )
        fig, _ = plt.subplots(figsize=_SINGLE_COL)
        plt.close(fig)
        return fig
    fig, ax = plt.subplots(figsize=_SINGLE_COL)
    sns.heatmap(
        df, annot=True, fmt=".2f",
        cmap=cmap, vmin=0.5, vmax=1.0,
        cbar_kws={"label": "AUROC"},
        linewidths=0.5, linecolor="white",
        ax=ax,
    )
    title = "Per-concept AUROC"
    if list(df.columns) != ["overall"]:
        title += " by Fitzpatrick bucket"
    ax.set_title(title)
    ax.set_xlabel("Fitzpatrick skin-tone bucket"
                  if list(df.columns) != ["overall"] else "")
    ax.set_ylabel("Clinical concept")
    plt.setp(ax.get_xticklabels(), rotation=0)
    _save(fig, out_dir, "fig2_concept_auroc_heatmap")
    return fig


# Fig 3 : Per-bucket diagnosis accuracy: CBM vs black-box
def plot_per_bucket_accuracy(
    per_bucket_by_model: Dict[str, pd.DataFrame],   # model_name -> df indexed by bucket
    metric: str = "auroc",
    out_dir: str | Path = "outputs/figures",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=_SINGLE_COL)
    models = list(per_bucket_by_model.keys())
    buckets = list(next(iter(per_bucket_by_model.values())).index)
    width = 0.8 / len(models)
    x = np.arange(len(buckets))
    all_vals = []
    for i, name in enumerate(models):
        df = per_bucket_by_model[name]
        vals = np.asarray(df[metric].values, dtype=float)
        ax.bar(x + i * width, vals, width=width, label=name)
        all_vals.append(vals)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(buckets)
    # Auto-fit the y-axis so near-chance bars are visible, rather than
    # clipping everything below 0.5. Clamp to [0, 1] since these are rates.
    stacked = np.concatenate(all_vals) if all_vals else np.array([0.5])
    stacked = stacked[np.isfinite(stacked)]
    lo = float(np.nanmin(stacked)) if stacked.size else 0.0
    hi = float(np.nanmax(stacked)) if stacked.size else 1.0
    pad = 0.05
    ymin = max(0.0, min(0.4, lo - pad))
    ymax = min(1.0, max(0.9, hi + pad))
    ax.set_ylim(ymin, ymax)
    if metric.lower() == "auroc" and ymin < 0.5 < ymax:
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Fitzpatrick bucket")
    ax.set_title(f"{metric.upper()} by skin tone, CBM vs black-box")
    ax.legend(frameon=False, fontsize=8)
    sns.despine()
    _save(fig, out_dir, f"fig3_per_bucket_{metric}")
    return fig


# Fig 4 : Per-bucket calibration plot
def plot_calibration(
    y_true_by_bucket: Dict[str, np.ndarray],
    y_prob_by_bucket: Dict[str, np.ndarray],
    model_name: str,
    n_bins: int = 10,
    out_dir: str | Path = "outputs/figures",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=_SINGLE_COL)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    bins = np.linspace(0, 1, n_bins + 1)
    mids = (bins[:-1] + bins[1:]) / 2
    for bucket, yt in y_true_by_bucket.items():
        yp = y_prob_by_bucket[bucket]
        accs = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (yp > lo) & (yp <= hi) if lo > 0 else (yp >= lo) & (yp <= hi)
            accs.append(yt[mask].mean() if mask.any() else np.nan)
        ax.plot(mids, accs, marker="o", label=bucket, linewidth=1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Calibration by skin tone — {model_name}")
    ax.legend(frameon=False, fontsize=8)
    sns.despine()
    _save(fig, out_dir, f"fig4_calibration_{model_name}")
    return fig


# Fig 5 : Intervention rescue plot
def plot_intervention_rescue(
    rescue_df: pd.DataFrame,                # rows=concepts, cols=buckets
    out_dir: str | Path = "outputs/figures",
    include_overall: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=_DOUBLE_COL)
    df = rescue_df.copy()
    # Keep plot columns: per-bucket rescue rates plus optional "overall"
    # (used when evaluating on a dataset without Fitzpatrick labels).
    keep = {"light", "medium", "dark", "overall"}
    bucket_cols = [c for c in df.columns if c in keep]
    if not bucket_cols:
        logger.warning(
            "plot_intervention_rescue: no recognised bucket columns in %s; "
            "skipping figure.", list(df.columns),
        )
        plt.close(fig)
        return fig
    df = df[bucket_cols]

    concepts = df.index.tolist()
    x = np.arange(len(concepts))
    width = 0.8 / max(len(bucket_cols), 1)
    colors = {
        "light": "#F2D17A", "medium": "#D98E3A", "dark": "#7A3A2C",
        "overall": "#4C72B0",
    }
    for i, b in enumerate(bucket_cols):
        ax.bar(x + i * width, df[b].values, width=width,
               label=b, color=colors.get(b, None))
    ax.set_xticks(x + width * (len(bucket_cols) - 1) / 2)
    ax.set_xticklabels(concepts, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Rescue rate")
    title = "Concept-intervention rescue rate"
    if bucket_cols != ["overall"]:
        title += " by skin tone"
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8)
    sns.despine()
    _save(fig, out_dir, "fig5_intervention_rescue")
    return fig


# Fig 6 : Qualitative: CBM concepts vs Grad-CAM panel
def plot_qualitative_panel(
    samples: List[Dict[str, Any]],  # each: {image, gradcam, concept_scores, concept_weights, label, pred}
    concept_names: Sequence[str],
    out_dir: str | Path = "outputs/figures",
) -> plt.Figure:
    import matplotlib.cm as cm
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(7.0, 2.3 * n))
    if n == 1:
        axes = axes[None, :]
    for i, s in enumerate(samples):
        img = s["image"]
        cam = s["gradcam"]
        ax_img, ax_cam, ax_bar = axes[i]
        ax_img.imshow(img); ax_img.set_axis_off()
        ax_img.set_title(f"True={s['label']}  CBM={s['pred']}")
        ax_cam.imshow(img); ax_cam.imshow(cam, cmap="jet", alpha=0.45)
        ax_cam.set_axis_off(); ax_cam.set_title("Grad-CAM")
        # Bar: top 5 concept contributions
        w = np.asarray(s.get("concept_weights", np.ones(len(concept_names))))
        x = np.asarray(s["concept_scores"])
        contrib = w * x
        order = np.argsort(-np.abs(contrib))[:5]
        ax_bar.barh([concept_names[j] for j in order][::-1],
                    contrib[order][::-1],
                    color=["#2F5496" if c >= 0 else "#C55" for c in contrib[order][::-1]])
        ax_bar.axvline(0, color="black", linewidth=0.5)
        ax_bar.set_title("CBM concept contrib.")
    fig.tight_layout()
    _save(fig, out_dir, "fig6_cbm_vs_gradcam_panel")
    return fig


# Fig 7 : Deletion/insertion curves (CBM vs Grad-CAM)
def plot_faithfulness_curves(
    cbm_result: Dict[str, np.ndarray],
    gradcam_result: Optional[Dict[str, np.ndarray]],
    out_dir: str | Path = "outputs/figures",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=_DOUBLE_COL, sharey=True)

    def _mean(curve):
        return curve.mean(axis=0)

    # Deletion
    axes[0].plot(cbm_result["x"], _mean(cbm_result["deletion_curve"]),
                 label=f"CBM (AUC={np.nanmean(cbm_result['deletion_auc']):.2f})",
                 color="#2F5496", linewidth=1.5)
    if gradcam_result is not None:
        axes[0].plot(gradcam_result["x"], _mean(gradcam_result["deletion_curve"]),
                     label=f"Grad-CAM (AUC={np.nanmean(gradcam_result['deletion_auc']):.2f})",
                     color="#C55", linewidth=1.5)
    axes[0].set_title("Deletion (lower = more faithful)")
    axes[0].set_xlabel("Fraction of features deleted")
    axes[0].set_ylabel("P(malignant)")
    axes[0].legend(frameon=False, fontsize=8)

    # Insertion
    axes[1].plot(cbm_result["x"], _mean(cbm_result["insertion_curve"]),
                 label=f"CBM (AUC={np.nanmean(cbm_result['insertion_auc']):.2f})",
                 color="#2F5496", linewidth=1.5)
    if gradcam_result is not None:
        axes[1].plot(gradcam_result["x"], _mean(gradcam_result["insertion_curve"]),
                     label=f"Grad-CAM (AUC={np.nanmean(gradcam_result['insertion_auc']):.2f})",
                     color="#C55", linewidth=1.5)
    axes[1].set_title("Insertion (higher = more faithful)")
    axes[1].set_xlabel("Fraction of features inserted")
    axes[1].legend(frameon=False, fontsize=8)

    sns.despine(fig)
    fig.tight_layout()
    _save(fig, out_dir, "fig7_faithfulness")
    return fig


# Fig 8 : Concept weights of the L1 logistic regression
def plot_concept_weights(
    concept_weights: Dict[str, float],
    out_dir: str | Path = "outputs/figures",
    title: str = "CBM-LR concept weights",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=_SINGLE_COL)
    items = sorted(concept_weights.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#2F5496" if v >= 0 else "#C55" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Weight (positive => melanoma)")
    ax.set_title(title)
    sns.despine(fig)
    _save(fig, out_dir, "fig8_concept_weights")
    return fig


# Driver
def generate_all_figures(data: Dict[str, Any], out_dir: str | Path) -> None:
    """Idempotent call invoked from notebook 08."""
    plot_pipeline_diagram(out_dir)
    if "concept_auroc_by_bucket" in data:
        plot_concept_auroc_heatmap(data["concept_auroc_by_bucket"], out_dir)
    if "per_bucket_by_model" in data:
        plot_per_bucket_accuracy(data["per_bucket_by_model"], out_dir=out_dir)
    if "rescue_df" in data:
        plot_intervention_rescue(data["rescue_df"], out_dir=out_dir)
    if "concept_weights" in data:
        plot_concept_weights(data["concept_weights"], out_dir=out_dir)
    if "cbm_faithfulness" in data:
        plot_faithfulness_curves(data["cbm_faithfulness"],
                                 data.get("gradcam_faithfulness"),
                                 out_dir=out_dir)
