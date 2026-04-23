from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

from .label_mapping import (
    CONCEPTS,
    DERM7PT_COLUMN_NAMES,
    DERM7PT_SUPERVISED_CONCEPTS,
    DEFAULT_BUCKETS,
    FitzpatrickBuckets,
    ddi_canonical_dx,
    derm7pt_canonical_dx,
    derm7pt_concept_binary,
    ham10000_canonical_dx,
    binary_label,
)

logger = logging.getLogger(__name__)

UNKNOWN_CONCEPT = -1


# Base class
class _BaseDermDataset(Dataset):
    source: str = "base"

    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str | os.PathLike,
        transform: Optional[Callable] = None,
        buckets: FitzpatrickBuckets = DEFAULT_BUCKETS,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.buckets = buckets

    def __len__(self) -> int:
        return len(self.df)

    # Subclasses implement these two.
    def _row_image_path(self, row: pd.Series) -> Path:
        raise NotImplementedError

    def _row_to_record(self, row: pd.Series) -> Dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        path = self._row_image_path(row)
        try:
            img = Image.open(path).convert("RGB")
        except FileNotFoundError:
            logger.warning("Missing image: %s", path)
            # Return a zero image so batching still works; filter upstream.
            img = Image.new("RGB", (224, 224))
        if self.transform is not None:
            img = self.transform(img)
        record = self._row_to_record(row)
        record["image"] = img
        record["source"] = self.source
        return record


def _empty_concept_vector() -> torch.Tensor:
    return torch.full((len(CONCEPTS),), float(UNKNOWN_CONCEPT), dtype=torch.float32)


# HAM10000
class HAM10000Dataset(_BaseDermDataset):
    source = "ham10000"

    def __init__(
        self,
        root: str | os.PathLike,
        split: Optional[str] = None,  # not native, accepted for parity
        transform: Optional[Callable] = None,
        buckets: FitzpatrickBuckets = DEFAULT_BUCKETS,
    ) -> None:
        root = Path(root)
        meta_candidates = [
            root / "HAM10000_metadata.csv",
            root / "HAM10000_metadata",
            root / "HAM10000_metadata.txt",
        ]
        meta_path = next((p for p in meta_candidates if p.exists()), None)
        if meta_path is None:
            raise FileNotFoundError(
                f"HAM10000 metadata CSV not found under {root}"
            )
        df = pd.read_csv(meta_path)
        # The image files live in either part_1 or part_2 directories.
        df["image_id"] = df["image_id"].astype(str)
        super().__init__(df, image_root=root, transform=transform, buckets=buckets)

        # Cache which folder each image lives in.
        self._image_dirs: List[Path] = [
            p for p in [root / "HAM10000_images_part_1",
                        root / "HAM10000_images_part_2",
                        root / "images",
                        root] if p.exists()
        ]

    def _row_image_path(self, row: pd.Series) -> Path:
        name = f"{row['image_id']}.jpg"
        for d in self._image_dirs:
            candidate = d / name
            if candidate.exists():
                return candidate
        return self._image_dirs[0] / name

    def _row_to_record(self, row: pd.Series) -> Dict[str, Any]:
        canonical = ham10000_canonical_dx(row.get("dx", "nv"))
        return {
            "image_id": str(row["image_id"]),
            "diagnosis_canonical": canonical,
            "diagnosis": binary_label(canonical),
            "concept_labels": _empty_concept_vector(),
            "fitzpatrick": None,
            "fitzpatrick_bucket": None,
        }


# Derm7pt
class Derm7ptDataset(_BaseDermDataset):
    source = "derm7pt"

    def __init__(
        self,
        root: str | os.PathLike,
        split: Optional[str] = None,  # "train" | "valid" | "test"
        transform: Optional[Callable] = None,
        buckets: FitzpatrickBuckets = DEFAULT_BUCKETS,
    ) -> None:
        root = Path(root)
        # Derm7pt ships separate train/valid/test CSVs + a meta CSV.
        meta_path = None
        for name in ("meta.csv", "release_v0/meta/meta.csv",
                     "release_v0/meta.csv"):
            p = root / name
            if p.exists():
                meta_path = p
                break
        if meta_path is None:
            raise FileNotFoundError(f"Derm7pt meta.csv not found under {root}")
        df = pd.read_csv(meta_path)

        # Split files enumerate row indexes into meta.csv
        if split is not None:
            for name in (f"release_v0/meta/{split}_indexes.csv",
                         f"meta/{split}_indexes.csv",
                         f"{split}_indexes.csv"):
                p = root / name
                if p.exists():
                    idx = pd.read_csv(p)["indexes"].tolist()
                    df = df.iloc[idx].reset_index(drop=True)
                    break

        super().__init__(df, image_root=root, transform=transform, buckets=buckets)

    def _row_image_path(self, row: pd.Series) -> Path:
        # Derm7pt stores a relative path in the `derm` (dermoscopy) column.
        rel = row.get("derm") or row.get("image")
        if not rel or (isinstance(rel, float) and np.isnan(rel)):
            return self.image_root
        # The release_v0 archive stores images at either:
        #   <root>/release_v0/images/<rel>
        #   <root>/images/<rel>
        # depending on where the zip was extracted.
        for base in (
            self.image_root / "release_v0" / "images",
            self.image_root / "images",
            self.image_root,
        ):
            candidate = base / str(rel)
            if candidate.exists():
                return candidate
        return self.image_root / "images" / str(rel)

    def _row_to_record(self, row: pd.Series) -> Dict[str, Any]:
        canonical = derm7pt_canonical_dx(row.get("diagnosis", "nevus"))
        concepts = _empty_concept_vector()
        for i, concept in enumerate(CONCEPTS):
            if concept not in DERM7PT_SUPERVISED_CONCEPTS:
                continue
            col = DERM7PT_COLUMN_NAMES[concept]
            raw = row.get(col)
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                continue
            val = derm7pt_concept_binary(concept, raw)
            if val is not None:
                concepts[i] = float(val)
        return {
            "image_id": str(row.get("case_num", row.name)),
            "diagnosis_canonical": canonical,
            "diagnosis": binary_label(canonical),
            "concept_labels": concepts,
            "fitzpatrick": None,
            "fitzpatrick_bucket": None,
        }


# Fitzpatrick17k
class Fitzpatrick17kDataset(_BaseDermDataset):
    source = "fitzpatrick17k"

    # Heuristic mapping from 114 skin conditions to our binary target.
    # Benign/malignant is provided in the CSV as `three_partition_label` in
    # {benign, malignant, non-neoplastic}. We keep the first two.
    def __init__(
        self,
        root: str | os.PathLike,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        buckets: FitzpatrickBuckets = DEFAULT_BUCKETS,
        include_non_neoplastic: bool = False,
        keep_only_existing_images: bool = True,
        subsample: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        root = Path(root)
        csv_path = root / "fitzpatrick17k.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} missing")
        df = pd.read_csv(csv_path)

        if not include_non_neoplastic and "three_partition_label" in df.columns:
            df = df[df["three_partition_label"].isin(["benign", "malignant"])]

        # Keep rows with a valid Fitzpatrick score.
        if "fitzpatrick_scale" in df.columns:
            df = df[df["fitzpatrick_scale"].isin([1, 2, 3, 4, 5, 6])]
        df = df.reset_index(drop=True)
        super().__init__(df, image_root=root, transform=transform, buckets=buckets)

        images_dir = self._resolve_images_dir()
        self._images_dir = images_dir
        if images_dir is None:
            logger.warning(
                "Fitzpatrick17k image directory not found under %s. The Dataset "
                "will still load metadata but __getitem__ will return zero "
                "images. Run download_fitzpatrick17k_images(...) first.", root,
            )
        elif keep_only_existing_images:
            # Drop rows whose image is missing so __getitem__ doesn't hit 404s.
            existing = self._existing_image_mask(self.df, images_dir)
            n_missing = int((~existing).sum())
            if n_missing:
                logger.info(
                    "Fitzpatrick17k: dropping %d / %d rows whose image "
                    "is missing on disk.",
                    n_missing, len(self.df),
                )
            self.df = self.df[existing].reset_index(drop=True)

        if subsample is not None and subsample < len(self.df):
            self.df = self.df.sample(
                n=subsample, random_state=random_state,
            ).reset_index(drop=True)

    def _resolve_images_dir(self) -> Optional[Path]:
        candidates = [
            self.image_root / "images",
            self.image_root,
        ]
        for d in candidates:
            if d.exists() and any(d.glob("*.jpg")) or (d.exists() and any(d.glob("*.jpeg"))):
                return d
        return None

    @staticmethod
    def _existing_image_mask(df: pd.DataFrame, images_dir: Path) -> "pd.Series":
        # Pre-index filenames for O(1) lookup.
        present = {p.name for p in images_dir.iterdir() if p.is_file()}
        names = df["md5hash"].astype(str) + ".jpg"
        return names.isin(present)

    def _row_image_path(self, row: pd.Series) -> Path:
        # md5hash.jpg under images/ or root
        name = f"{row['md5hash']}.jpg" if "md5hash" in row else str(row.get("image_name"))
        base = self._images_dir or (self.image_root / "images")
        return base / name

    def _row_to_record(self, row: pd.Series) -> Dict[str, Any]:
        partition = str(row.get("three_partition_label", "benign")).strip().lower()
        canonical = "melanoma" if partition == "malignant" else "benign"
        fitz = int(row["fitzpatrick_scale"]) if "fitzpatrick_scale" in row else None
        return {
            "image_id": str(row.get("md5hash", row.name)),
            "diagnosis_canonical": canonical,
            "diagnosis": binary_label(canonical),
            "concept_labels": _empty_concept_vector(),
            "fitzpatrick": fitz,
            "fitzpatrick_bucket": self.buckets.bucket_name(fitz),
        }


# DDI
class DDIDataset(_BaseDermDataset):
    source = "ddi"

    def __init__(
        self,
        root: str | os.PathLike,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        buckets: FitzpatrickBuckets = DEFAULT_BUCKETS,
    ) -> None:
        root = Path(root)
        # DDI ships a single CSV; the exact name varies by release.
        meta_candidates = [
            root / "ddi_metadata.csv",
            root / "metadata.csv",
            root / "DDI_metadata.csv",
        ]
        meta_path = next((p for p in meta_candidates if p.exists()), None)
        if meta_path is None:
            raise FileNotFoundError(f"DDI metadata not found under {root}")
        df = pd.read_csv(meta_path)
        super().__init__(df, image_root=root, transform=transform, buckets=buckets)

    def _row_image_path(self, row: pd.Series) -> Path:
        name = str(row.get("DDI_file") or row.get("image") or row.get("filename"))
        # Stanford AIMI distributes DDI as a flat ZIP of PNGs + ddi_metadata.csv,
        # so images live directly under <root>, not <root>/images.
        for base in (self.image_root, self.image_root / "images"):
            candidate = base / name
            if candidate.exists():
                return candidate
        return self.image_root / name

    def _row_to_record(self, row: pd.Series) -> Dict[str, Any]:
        # DDI stores `malignant` as True/False (can arrive as str "True"/"False"
        # after CSV round-tripping).
        mal_raw = row.get("malignant", False)
        if isinstance(mal_raw, str):
            malignant = mal_raw.strip().lower() in {"true", "1", "yes", "t"}
        else:
            malignant = bool(mal_raw)
        canonical = ddi_canonical_dx(malignant)
        # ``skin_tone`` is one of {12, 34, 56} (I-II / III-IV / V-VI).
        fitz_raw = row.get("skin_tone", row.get("fitzpatrick"))
        try:
            fitz_int = int(fitz_raw) if fitz_raw is not None and not (
                isinstance(fitz_raw, float) and np.isnan(fitz_raw)
            ) else None
        except (TypeError, ValueError):
            fitz_int = None
        return {
            "image_id": str(row.get("DDI_file", row.name)),
            "diagnosis_canonical": canonical,
            "diagnosis": binary_label(canonical),
            "concept_labels": _empty_concept_vector(),
            "fitzpatrick": fitz_int,
            "fitzpatrick_bucket": self.buckets.bucket_name(fitz_int),
        }


# Simple registry
DATASET_REGISTRY = {
    "ham10000": HAM10000Dataset,
    "derm7pt": Derm7ptDataset,
    "fitzpatrick17k": Fitzpatrick17kDataset,
    "ddi": DDIDataset,
}


def build_dataset(name: str, **kwargs) -> Dataset:
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Choose from {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[key](**kwargs)


def default_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["image"] = torch.stack([b["image"] for b in batch]) \
        if isinstance(batch[0]["image"], torch.Tensor) else [b["image"] for b in batch]
    out["image_id"] = [b["image_id"] for b in batch]
    out["diagnosis"] = torch.tensor([b["diagnosis"] for b in batch], dtype=torch.long)
    out["diagnosis_canonical"] = [b["diagnosis_canonical"] for b in batch]
    out["concept_labels"] = torch.stack([b["concept_labels"] for b in batch])
    out["fitzpatrick"] = [b["fitzpatrick"] for b in batch]
    out["fitzpatrick_bucket"] = [b["fitzpatrick_bucket"] for b in batch]
    out["source"] = [b["source"] for b in batch]
    return out
