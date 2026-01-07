"""ResNet18-based binary classifier for tamarisk identification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(frozen=True)
class TamariskPrediction:
    """Inference output for a single image."""

    probability: float
    label: str


def default_transforms(image_size: int = 224) -> transforms.Compose:
    """Build inference transforms aligned with the ResNet18 ImageNet recipe."""

    weights = ResNet18_Weights.DEFAULT
    try:
        if image_size == 224:
            return weights.transforms()
    except Exception:
        pass

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    meta = getattr(weights, "meta", {})
    if isinstance(meta, dict):
        mean = tuple(meta.get("mean", mean))
        std = tuple(meta.get("std", std))

    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class TamariskClassifier(nn.Module):
    """Binary classifier that predicts tamarisk presence in an image."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        return logits.squeeze(1)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> list[TamariskPrediction]:
        """Return probabilities and class labels for a batch of images."""

        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        predictions = []
        for prob in probs.detach().cpu().tolist():
            label = "tamarisk" if prob >= threshold else "non_tamarisk"
            predictions.append(TamariskPrediction(probability=prob, label=label))
        return predictions


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix) :]: value for key, value in state_dict.items()}


def _extract_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_classifier(
    weights_path: Optional[str | Path] = None,
    device: Optional[str | torch.device] = None,
    pretrained: bool = True,
) -> TamariskClassifier:
    """Instantiate a classifier and optionally load fine-tuned weights."""

    model = TamariskClassifier(pretrained=pretrained)
    if weights_path is not None:
        resolved = Path(weights_path).expanduser().resolve()
        checkpoint = torch.load(resolved, map_location=device)
        state_dict = _extract_state_dict(checkpoint)
        state_dict = _strip_prefix(state_dict, "module.")
        model.load_state_dict(state_dict, strict=False)
    return model


def batch_images(images: Iterable[torch.Tensor]) -> torch.Tensor:
    """Stack image tensors into a batch."""

    return torch.stack(list(images), dim=0)
