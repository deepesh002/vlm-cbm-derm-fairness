from __future__ import annotations

import logging
import os
import platform
import random
from typing import Any, Dict, Optional

import numpy as np

try:  # torch may be absent in a pure-sandbox environment
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

logger = logging.getLogger(__name__)


# Device selection
def get_device(prefer: Optional[str] = None):
    if torch is None:
        return "cpu"

    if prefer:
        return torch.device(prefer)

    # Apple Silicon
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        # Allow CPU fallback for ops MPS does not yet implement.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def describe_device(device=None) -> str:
    if torch is None:
        return "cpu (torch not installed)"
    device = device or get_device()
    device = torch.device(device) if not isinstance(device, torch.device) else device
    if device.type == "cuda":
        idx = device.index or 0
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    if device.type == "mps":
        return f"mps (Apple {platform.machine()})"
    return f"cpu ({platform.processor() or platform.machine()})"


# DataLoader defaults
def dataloader_kwargs(device=None, num_workers: Optional[int] = None) -> Dict[str, Any]:
    device = device or get_device()
    if torch is not None and not isinstance(device, torch.device):
        device = torch.device(device)
    is_cuda = getattr(device, "type", "cpu") == "cuda"

    if num_workers is None:
        if platform.system() == "Darwin":
            num_workers = 0      # fork-based workers can hang on macOS
        else:
            num_workers = min(4, (os.cpu_count() or 2) - 1)

    return {
        "num_workers": num_workers,
        "pin_memory": is_cuda,
        "persistent_workers": bool(num_workers) and is_cuda,
    }


# Seeding
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if getattr(torch.backends, "mps", None) is not None \
                and torch.backends.mps.is_available():
            try:
                torch.mps.manual_seed(seed)
            except Exception:  # pragma: no cover
                pass


def to_device(obj, device=None):
    if torch is None:
        return obj
    device = device or get_device()
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=(getattr(device, "type", None) == "cuda"))
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj
