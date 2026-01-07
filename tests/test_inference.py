from __future__ import annotations

import torch
from torchvision.transforms.functional import to_pil_image

from kineo_tamarisk.model.classifier import TamariskClassifier, default_transforms


def test_inference_runs() -> None:
    model = TamariskClassifier(pretrained=False)
    model.eval()

    image = to_pil_image(torch.rand(3, 224, 224))
    transform = default_transforms(image_size=224)
    batch = transform(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(batch)

    assert logits.shape == (1,)
