from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping

import pandas as pd


def _save_table(df: pd.DataFrame, out_dir: str | Path, name: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{name}.csv"
    tex_path = out / f"{name}.tex"
    md_path = out / f"{name}.md"
    df.to_csv(csv_path)
    # LaTeX export via pandas requires jinja2 >= 3.1.2 on recent pandas.
    try:
        tex = df.to_latex(float_format="%.3f")
        Path(tex_path).write_text(tex)
    except Exception:
        # Fall back to a minimal booktabs-style writer so the pipeline never
        # crashes in reduced environments.
        Path(tex_path).write_text(_dataframe_to_latex_fallback(df))
    try:
        df.to_markdown(md_path, floatfmt=".3f")
    except Exception:
        Path(md_path).write_text(df.to_csv(sep="|"))
    return csv_path


def _dataframe_to_latex_fallback(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    col_spec = "l" + "r" * len(cols)
    header = " & ".join(["idx"] + cols) + r" \\"
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for idx, row in df.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append(" & ".join([str(idx)] + vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# Table 1 : Dataset summary
def table1_dataset_summary(out_dir: str | Path) -> Path:
    df = pd.DataFrame([
        {"Dataset": "HAM10000", "N images": 10015, "Classes": 7,
         "Fitzpatrick labels": "No", "Role": "Primary training"},
        {"Dataset": "Derm7pt", "N images": 2000, "Classes": "MEL / benign",
         "Fitzpatrick labels": "No", "Role": "Concept supervision"},
        {"Dataset": "Fitzpatrick17k", "N images": 16577, "Classes": 114,
         "Fitzpatrick labels": "Yes (I-VI)", "Role": "Fairness eval"},
        {"Dataset": "DDI", "N images": 656, "Classes": "malignant / benign",
         "Fitzpatrick labels": "Yes (expert)", "Role": "Fairness gold std"},
    ]).set_index("Dataset")
    return _save_table(df, out_dir, "table1_dataset_summary")


# Table 2 : Overall accuracy comparison
def table2_overall_accuracy(
    model_metrics: Mapping[str, Mapping[str, float]],
    out_dir: str | Path,
) -> Path:
    df = pd.DataFrame(model_metrics).T
    df.index.name = "model"
    return _save_table(df, out_dir, "table2_overall_accuracy")


# Table 3 : Per-bucket metrics across models
def table3_per_bucket(
    per_bucket_by_model: Mapping[str, pd.DataFrame],
    out_dir: str | Path,
    metrics: Iterable[str] = ("auroc", "sensitivity", "specificity", "ece"),
) -> Path:
    frames = []
    for model_name, df in per_bucket_by_model.items():
        sub = df[[m for m in metrics if m in df.columns]].copy()
        sub.columns = pd.MultiIndex.from_product([[model_name], sub.columns])
        frames.append(sub)
    wide = pd.concat(frames, axis=1)
    return _save_table(wide, out_dir, "table3_per_bucket")


# Table 4 : Equalized-odds / demographic-parity gaps
def table4_fairness_gaps(
    model_gaps: Mapping[str, Mapping[str, float]],
    out_dir: str | Path,
) -> Path:
    df = pd.DataFrame(model_gaps).T
    df.index.name = "model"
    return _save_table(df, out_dir, "table4_fairness_gaps")


# Table 5 : Per-concept AUROC on Derm7pt
def table5_concept_auroc(
    concept_auroc: Mapping[str, float],
    out_dir: str | Path,
) -> Path:
    df = pd.DataFrame(
        {"AUROC": concept_auroc.values()},
        index=list(concept_auroc.keys()),
    )
    df.index.name = "concept"
    return _save_table(df, out_dir, "table5_concept_auroc")


# Driver
def generate_all_tables(data: Dict, out_dir: str | Path) -> None:
    table1_dataset_summary(out_dir)
    if "model_metrics" in data:
        table2_overall_accuracy(data["model_metrics"], out_dir)
    if "per_bucket_by_model" in data:
        table3_per_bucket(data["per_bucket_by_model"], out_dir)
    if "model_gaps" in data:
        table4_fairness_gaps(data["model_gaps"], out_dir)
    if "concept_auroc" in data:
        table5_concept_auroc(data["concept_auroc"], out_dir)
