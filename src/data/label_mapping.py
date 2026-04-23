from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Canonical concept vocabulary
CONCEPTS: List[str] = [
    "pigment_network",   # 1 - Derm7pt supervised
    "dots_globules",     # 2 - Derm7pt supervised
    "blue_white_veil",   # 3 - Derm7pt supervised
    "streaks",           # 4 - Derm7pt supervised
    "regression",        # 5 - Derm7pt supervised
    "vascular",          # 6 - Derm7pt supervised
    "asymmetry",         # 7 - VLM-only (no GT)
    "border",            # 8 - VLM-only
    "color_var",         # 9 - VLM-only
]

DERM7PT_SUPERVISED_CONCEPTS: List[str] = CONCEPTS[:6]
VLM_ONLY_CONCEPTS: List[str] = CONCEPTS[6:]


# Derm7pt raw-label -> canonical binary presence mapping
DERM7PT_CONCEPT_BINARIZERS: Dict[str, Dict[str, int]] = {
    "pigment_network": {
        # typical -> 0, atypical -> 1, absent -> 0
        "typical": 0, "atypical": 1, "absent": 0,
    },
    "dots_globules": {
        "absent": 0, "regular": 0, "irregular": 1,
    },
    "blue_white_veil": {
        "absent": 0, "present": 1,
    },
    "streaks": {
        "absent": 0, "regular": 0, "irregular": 1,
    },
    "regression": {
        "absent": 0, "present": 1,
    },
    "vascular": {
        "absent": 0, "within regression": 0,
        "arborizing": 1, "comma": 1, "hairpin": 1, "wreath": 1,
        "dotted": 1, "linear irregular": 1,
    },
}

# The exact column names Derm7pt uses in the meta CSV
DERM7PT_COLUMN_NAMES: Dict[str, str] = {
    "pigment_network": "pigment_network",
    "dots_globules": "dots_and_globules",
    "blue_white_veil": "blue_whitish_veil",
    "streaks": "streaks",
    "regression": "regression_structures",
    "vascular": "vascular_structures",
}


# Canonical diagnosis labels
DIAGNOSIS_CLASSES: List[str] = ["benign", "melanoma"]


HAM10000_DX_MAP: Dict[str, str] = {
    "mel": "melanoma",
    "nv": "benign",
    "bcc": "other_malignant",
    "akiec": "other_malignant",
    "bkl": "benign",
    "df": "benign",
    "vasc": "benign",
}

DERM7PT_DX_MAP: Dict[str, str] = {
    # Canonical Derm7pt diagnosis strings. 
    "melanoma": "melanoma",
    "nevus": "benign",
    "blue nevus": "benign",
    "clark nevus": "benign",
    "combined nevus": "benign",
    "congenital nevus": "benign",
    "dermal nevus": "benign",
    "recurrent nevus": "benign",
    "reed or spitz nevus": "benign",
    "seborrheic keratosis": "benign",
    "lentigo": "benign",
    "dermatofibroma": "benign",
    "vascular lesion": "benign",
    "melanosis": "benign",
    "miscellaneous": "benign",
    "basal cell carcinoma": "other_malignant",
    "melanoma metastasis": "melanoma",
}

DDI_DX_MAP: Dict[str, str] = {
    # DDI uses a Boolean `malignant` column (True/False) plus a free-text dx.
    "malignant": "melanoma",     # DDI fairness benchmark focuses on malignancy
    "benign": "benign",
}


def binary_label(canonical_dx: str) -> int:
    """Return 1 for melanoma / 0 for benign; other_malignant is treated as 1."""
    return 0 if canonical_dx == "benign" else 1


# Fitzpatrick bucket mapping
@dataclass
class FitzpatrickBuckets:
    light: Tuple[int, ...] = (1, 2)
    medium: Tuple[int, ...] = (3, 4)
    dark: Tuple[int, ...] = (5, 6)
    # Compound codes used by DDI's ``skin_tone`` column.
    compound_light: Tuple[int, ...] = (12,)
    compound_medium: Tuple[int, ...] = (34,)
    compound_dark: Tuple[int, ...] = (56,)

    def bucket_name(self, fitzpatrick: Optional[int]) -> Optional[str]:
        if fitzpatrick is None:
            return None
        try:
            val = int(fitzpatrick)
        except (TypeError, ValueError):
            return None
        if val in self.light or val in self.compound_light:
            return "light"
        if val in self.medium or val in self.compound_medium:
            return "medium"
        if val in self.dark or val in self.compound_dark:
            return "dark"
        return None

    def all_buckets(self) -> List[str]:
        return ["light", "medium", "dark"]


DEFAULT_BUCKETS = FitzpatrickBuckets()


# Helpers
def derm7pt_concept_binary(concept: str, raw_value: str) -> Optional[int]:
    mapping = DERM7PT_CONCEPT_BINARIZERS.get(concept)
    if mapping is None:
        return None
    key = str(raw_value).strip().lower()
    return mapping.get(key)


def ham10000_canonical_dx(dx: str) -> str:
    return HAM10000_DX_MAP.get(str(dx).strip().lower(), "benign")


def derm7pt_canonical_dx(dx: str) -> str:
    key = str(dx).strip().lower()
    # Strip any parenthetical qualifier, e.g. "melanoma (in situ)" -> "melanoma"
    key = re.sub(r"\s*\(.*?\)", "", key).strip()
    if key in DERM7PT_DX_MAP:
        return DERM7PT_DX_MAP[key]
    if key.startswith("melanoma"):
        return "melanoma"
    if "nevus" in key or "nevi" in key:
        return "benign"
    return DERM7PT_DX_MAP.get(key, "benign")


def ddi_canonical_dx(malignant_flag: bool) -> str:
    return "melanoma" if bool(malignant_flag) else "benign"
