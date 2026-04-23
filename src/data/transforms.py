from __future__ import annotations

from typing import Tuple

import torch
from torchvision import transforms as T

# BiomedCLIP / OpenCLIP default stats
CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

# ImageNet stats (for ResNet / EfficientNet baselines)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def biomedclip_eval_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def baseline_train_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(int(image_size * 1.15),
                 interpolation=T.InterpolationMode.BICUBIC),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def baseline_eval_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(int(image_size * 1.15),
                 interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(tensor: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    return tensor * std_t + mean_t
