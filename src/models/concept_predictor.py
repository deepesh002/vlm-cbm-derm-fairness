from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ..utils import get_device

logger = logging.getLogger(__name__)


# Concept prompt templates
@dataclass(frozen=True)
class ConceptPrompt:
    concept_id: str
    clinical_name: str
    positive_prompt: str
    negative_prompt: str
    derm7pt_supervised: bool


CONCEPT_PROMPTS: List[ConceptPrompt] = [
    ConceptPrompt(
        "pigment_network", "Atypical pigment network",
        "A dermoscopic photo showing an atypical pigment network with irregular lines",
        "A dermoscopic photo showing a typical regular pigment network or no network",
        True,
    ),
    ConceptPrompt(
        "dots_globules", "Irregular dots and globules",
        "A dermoscopic photo showing irregular dots or globules within the lesion",
        "A dermoscopic photo without irregular dots or globules",
        True,
    ),
    ConceptPrompt(
        "blue_white_veil", "Blue-white veil",
        "A dermoscopic photo showing a blue-white veil overlying the lesion",
        "A dermoscopic photo without blue-white veil",
        True,
    ),
    ConceptPrompt(
        "streaks", "Irregular streaks / pseudopods",
        "A dermoscopic photo showing irregular streaks or pseudopods at the periphery",
        "A dermoscopic photo without streaks or pseudopods",
        True,
    ),
    ConceptPrompt(
        "regression", "Regression structures",
        "A dermoscopic photo showing white scar-like regression structures",
        "A dermoscopic photo without regression structures",
        True,
    ),
    ConceptPrompt(
        "vascular", "Atypical vascular pattern",
        "A dermoscopic photo showing atypical vascular structures or dotted vessels",
        "A dermoscopic photo with normal or no visible vascular pattern",
        True,
    ),
    ConceptPrompt(
        "asymmetry", "Asymmetry",
        "A dermoscopic photo of an asymmetric lesion in shape or color distribution",
        "A dermoscopic photo of a symmetric round lesion",
        False,
    ),
    ConceptPrompt(
        "border", "Border irregularity",
        "A dermoscopic photo showing irregular jagged borders with abrupt cutoff",
        "A dermoscopic photo with smooth regular borders",
        False,
    ),
    ConceptPrompt(
        "color_var", "Color variegation",
        "A dermoscopic photo showing multiple colors including brown black red blue white",
        "A dermoscopic photo with uniform single color",
        False,
    ),
]

CONCEPT_IDS: List[str] = [c.concept_id for c in CONCEPT_PROMPTS]


# Backbone loading
def _load_open_clip(model_hub_id: str, device: torch.device):
    import open_clip
    logger.info("Loading OpenCLIP backbone: %s", model_hub_id)
    model, preprocess = open_clip.create_model_from_pretrained(model_hub_id)
    tokenizer = open_clip.get_tokenizer(model_hub_id)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess, tokenizer


class BiomedCLIPConceptPredictor:

    def __init__(
        self,
        model_hub_id: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        device: Optional[str | torch.device] = None,
        prompts: Sequence[ConceptPrompt] = tuple(CONCEPT_PROMPTS),
        temperature: Optional[float] = None,
        fallback_hub_id: str = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    ) -> None:
        self.device = torch.device(device) if device is not None else get_device()
        self.prompts = list(prompts)
        try:
            self.model, self.preprocess, self.tokenizer = _load_open_clip(
                model_hub_id, self.device
            )
            self.model_hub_id = model_hub_id
        except Exception as exc:
            logger.warning(
                "Could not load BiomedCLIP (%s). Falling back to %s.",
                exc, fallback_hub_id,
            )
            self.model, self.preprocess, self.tokenizer = _load_open_clip(
                fallback_hub_id, self.device
            )
            self.model_hub_id = fallback_hub_id

        # `logit_scale` is a learned temperature exp(logit_scale). Honour the
        # pretrained value unless the caller overrides.
        if temperature is not None:
            self.logit_scale = torch.tensor(float(temperature), device=self.device)
        else:
            self.logit_scale = self.model.logit_scale.detach().exp()

        # Pre-compute text embeddings (2 per concept: positive + negative).
        self._text_features = self._encode_prompts()

    # Text side
    @torch.no_grad()
    def _encode_prompts(self) -> torch.Tensor:
        texts: List[str] = []
        for p in self.prompts:
            texts.append(p.positive_prompt)
            texts.append(p.negative_prompt)
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = F.normalize(feats, dim=-1)
        return feats  # (2C, D)

    # Image side
    @torch.no_grad()
    def encode_images(self, images) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            batch = images.to(self.device)
        else:
            batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        feats = self.model.encode_image(batch)
        feats = F.normalize(feats, dim=-1)
        return feats

    # Full prediction
    @torch.no_grad()
    def predict(self, images) -> np.ndarray:
        img_feats = self.encode_images(images)                # (N, D)
        logits = self.logit_scale * img_feats @ self._text_features.t()  # (N, 2C)
        logits = logits.view(img_feats.size(0), len(self.prompts), 2)    # (N, C, 2)
        probs = F.softmax(logits, dim=-1)[..., 0]                        # pick pos
        return probs.detach().cpu().numpy().astype(np.float32)

    # DataLoader integration
    @torch.no_grad()
    def predict_dataset(
        self,
        dataloader,
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        all_probs, ids, diag, gt_concepts, fitz, fitz_b = [], [], [], [], [], []
        it = dataloader
        if show_progress:
            it = tqdm(dataloader, desc="Concept extraction", total=len(dataloader))
        for batch in it:
            probs = self.predict(batch["image"])
            all_probs.append(probs)
            ids.extend(batch["image_id"])
            diag.extend(batch["diagnosis"].cpu().numpy().tolist())
            gt_concepts.append(batch["concept_labels"].cpu().numpy())
            fitz.extend(batch["fitzpatrick"])
            fitz_b.extend(batch["fitzpatrick_bucket"])
        return {
            "image_ids": np.asarray(ids, dtype=object),
            "concepts": np.concatenate(all_probs, axis=0),
            "diagnosis": np.asarray(diag, dtype=np.int64),
            "concept_labels": np.concatenate(gt_concepts, axis=0),
            "fitzpatrick": np.asarray(fitz, dtype=object),
            "fitzpatrick_bucket": np.asarray(fitz_b, dtype=object),
        }


# Persistence helpers
def save_concept_bundle(path: str | Path, bundle: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **bundle)


def load_concept_bundle(path: str | Path) -> Dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}
