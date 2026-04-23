from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

from ..utils import get_device

logger = logging.getLogger(__name__)


# Model factory
def build_baseline(
    arch: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    arch = arch.lower()
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, num_classes)
        target_layer_name = "features.8"
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layer_name = "layer4"
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layer_name = "layer4"
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    model.arch = arch
    model.gradcam_target_layer = target_layer_name
    return model


# Training / evaluation helpers
@dataclass
class TrainState:
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_auroc: float = 0.0
    best_val_auroc: float = 0.0


def _step(batch, model, device):
    non_blocking = getattr(device, "type", "cpu") == "cuda"
    imgs = batch["image"].to(device, non_blocking=non_blocking)
    labels = batch["diagnosis"].to(device, non_blocking=non_blocking)
    logits = model(imgs)
    return logits, labels


def train_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 20,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    device: Optional[str | torch.device] = None,
    early_stopping_patience: int = 5,
    checkpoint_path: Optional[str | Path] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    device = torch.device(device) if device is not None else get_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history: List[Dict[str, float]] = []
    best_auroc, best_state, patience = 0.0, None, 0

    for epoch in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}"):
            logits, labels = _step(batch, model, device)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * labels.size(0)
            n += labels.size(0)
        scheduler.step()
        train_loss = total / max(n, 1)

        # Validation
        val_loss, val_auroc = float("nan"), float("nan")
        if val_loader is not None:
            val_loss, val_auroc = evaluate_baseline(model, val_loader, device=device,
                                                    criterion=criterion)
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
                if checkpoint_path is not None:
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"state_dict": best_state,
                                "arch": getattr(model, "arch", "unknown")},
                               checkpoint_path)
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_auroc": val_auroc})

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_auroc": best_auroc}


@torch.no_grad()
def evaluate_baseline(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[str | torch.device] = None,
    criterion: Optional[nn.Module] = None,
) -> Tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    device = torch.device(device) if device is not None else get_device()
    model.eval()
    losses, probs, labels = [], [], []
    for batch in loader:
        logits, y = _step(batch, model, device)
        if criterion is not None:
            losses.append(criterion(logits, y).item() * y.size(0))
        probs.append(F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy())
        labels.append(y.cpu().numpy())
    proba = np.concatenate(probs)
    y_true = np.concatenate(labels)
    try:
        auroc = roc_auc_score(y_true, proba)
    except ValueError:
        auroc = float("nan")
    val_loss = sum(losses) / len(y_true) if losses else float("nan")
    return val_loss, auroc


@torch.no_grad()
def predict_baseline(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[str | torch.device] = None,
) -> Dict[str, np.ndarray]:
    device = torch.device(device) if device is not None else get_device()
    model.to(device).eval()
    ids, diag, probs, fitz, fitz_b = [], [], [], [], []
    for batch in loader:
        logits, y = _step(batch, model, device)
        probs.append(F.softmax(logits, dim=-1).cpu().numpy())
        ids.extend(batch["image_id"])
        diag.extend(batch["diagnosis"].cpu().numpy().tolist())
        fitz.extend(batch["fitzpatrick"])
        fitz_b.extend(batch["fitzpatrick_bucket"])
    return {
        "image_ids": np.asarray(ids, dtype=object),
        "proba": np.concatenate(probs, axis=0),
        "diagnosis": np.asarray(diag, dtype=np.int64),
        "fitzpatrick": np.asarray(fitz, dtype=object),
        "fitzpatrick_bucket": np.asarray(fitz_b, dtype=object),
    }


# Grad-CAM wrapper
class GradCAMExplainer:

    def __init__(self, model: nn.Module, target_layer_name: Optional[str] = None,
                 device: Optional[str | torch.device] = None):
        self.model = model
        self.device = torch.device(device) if device is not None else get_device()
        self.target_layer_name = (
            target_layer_name or getattr(model, "gradcam_target_layer", None)
        )
        if self.target_layer_name is None:
            raise ValueError("No target layer specified for Grad-CAM.")
        self.target_layer = self._resolve(model, self.target_layer_name)

    @staticmethod
    def _resolve(model: nn.Module, dotted: str) -> nn.Module:
        layer = model
        for part in dotted.split("."):
            layer = getattr(layer, part) if not part.isdigit() else layer[int(part)]
        return layer

    def explain(self, images: torch.Tensor, target_class: int = 1) -> np.ndarray:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        cam = GradCAM(model=self.model, target_layers=[self.target_layer])
        targets = [ClassifierOutputTarget(target_class) for _ in range(images.size(0))]
        grayscale_cam = cam(input_tensor=images.to(self.device), targets=targets)
        return grayscale_cam  # (N, H, W)
