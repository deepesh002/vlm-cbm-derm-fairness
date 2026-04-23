# VLM-Powered Concept Bottleneck Model for Dermatology Classification
### with a Fitzpatrick Skin-Tone Fairness Audit

A Concept Bottleneck Model (CBM) for dermatology where a frozen BiomedCLIP VLM
extracts ABCDE-rule dermoscopic concepts, a sparse logistic-regression head
makes the melanoma-vs-benign call from concepts alone, and every metric is
stratified by Fitzpatrick skin type (I–II / III–IV / V–VI) using DDI and
Fitzpatrick17k.

## Pipeline

```
    ┌─────────────────────┐
    │ Dermoscopic image   │
    └──────────┬──────────┘
               ▼
 ┌─────────────────────────────┐
 │ Stage 1: BiomedCLIP (frozen)│  →  9-dim concept vector
 └──────────┬──────────────────┘
            ▼
 ┌─────────────────────────────┐
 │ Stage 2: CBM-LR / CBM-MLP   │  →  Diagnosis (MEL / benign)
 └──────────┬──────────────────┘
            ▼
 ┌─────────────────────────────┐
 │ Stage 4: Fitzpatrick audit  │  →  Concept-AUROC heatmap,
 │         + Stage 3 baselines │     equalized-odds gap, ECE
 └──────────┬──────────────────┘
            ▼
 ┌─────────────────────────────┐
 │ Stage 5: Concept intervention│ →  Rescue rate per skin tone
 │ Stage 6 (bonus): Faithfulness│    vs Grad-CAM (del/ins AUC)
 └─────────────────────────────┘
```

## Quickstart (MacBook with Apple Silicon, CUDA, or CPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Drop the three dataset archives you downloaded manually into `raw/`:

```
raw/
  release_v0.zip                    # Derm7pt   (https://derm.cs.sfu.ca)
  ddidiversedermatologyimages.zip   # DDI       (Stanford AIMI)
  fitzpatrick17k.csv                # Fitz17k   (mattgroh/fitzpatrick17k)
  fitzpatrick17k.zip                # Fitz17k images   (reach out per instructions on the git repo to get these images link)
```

HAM10000 is fetched automatically from Kaggle on first run (needs
`~/.kaggle/kaggle.json`), or drop a `raw/ham10000.zip` if you already
downloaded it.

Then execute the notebooks in order:

1. `notebooks/01_data_download.ipynb` — extract `raw/` archives into `data/` and print a readiness table
2. `notebooks/02_concept_extraction.ipynb` — run BiomedCLIP on all images, save `concepts_*.npz`
3. `notebooks/03_cbm_training.ipynb` — train CBM-LR and CBM-MLP on Derm7pt
4. `notebooks/04_baseline_training.ipynb` — fine-tune EfficientNet-B0 and ResNet-50
5. `notebooks/05_fairness_audit.ipynb` — Fitzpatrick-stratified metrics on DDI / Fitz17k
6. `notebooks/06_intervention.ipynb` — concept intervention simulation
7. `notebooks/07_faithfulness.ipynb` — CBM vs Grad-CAM deletion/insertion (bonus)
8. `notebooks/08_figures_tables.ipynb` — generate all 8 figures and 5 tables

### Device auto-selection

`src.utils.get_device()` picks MPS on Apple Silicon, CUDA on NVIDIA, else CPU, and sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so any op Metal doesn't implement yet falls back to CPU automatically. Every model accepts a `device=None` argument that means "auto".

## Repository Layout

```
vlm-cbm-derm-fairness/
├── config.yaml                  # All hyperparameters
├── requirements.txt
├── data/                        # unzipped raw data
├── notebooks/                   # 8 Colab-ready notebooks
├── raw/                         # dataset csv and zip files
├── src/
│   ├── data/        # download, label mapping, Dataset classes, transforms
│   ├── models/      # BiomedCLIP concept predictor, CBM heads, black-box CNN
│   ├── evaluation/  # metrics, fairness, intervention, faithfulness
│   └── visualization/  # plots.py, tables.py
├── outputs/         # figures/, tables/, concept_vectors/, model_checkpoints/
└── report/          # main.tex + references.bib (ACM format)
```

## Datasets

| Dataset          | Size              | Fitzpatrick | Role                             |
|------------------|-------------------|-------------|----------------------------------|
| HAM10000         | 10,015 dermoscopy | No          | Primary training                 |
| Derm7pt          | ~2,000            | No          | Concept supervision              |
| Fitzpatrick17k   | 16,577 clinical   | **Yes**     | Fairness evaluation              |
| DDI              | 656 clinical      | **Yes**     | Fairness gold standard (biopsy)  |

All open-access; no DUA required. See `src/data/download.py`.

## Research Questions

- **Q1** Does a VLM-powered CBM achieve competitive accuracy vs. a black-box
  baseline while providing interpretable concept-level explanations?
- **Q2** Do concept-prediction errors disproportionately affect darker skin
  tones (Fitzpatrick V–VI)?
- **Q3** Can concept-level interventions rescue misdiagnoses, and does the
  rescue rate differ by skin tone?
- **Q4** Are CBM explanations more faithful than Grad-CAM, and do they surface
  fairness issues that Grad-CAM hides?

## Citation

Key prior work:
- Koh et al., *Concept Bottleneck Models*, ICML 2020.
- Daneshjou et al., *Disparities in dermatology AI performance on a diverse,
  curated clinical image set*, Science Advances 2022.
- Patricio, Teixeira, Neves, *Towards Concept-based Interpretability of Skin
  Lesion Diagnosis Using VLMs*, CSBJ / MICCAI iMIMIC 2024.
