# kineo_tamarisk

Open-source Python library for image-based salt cedar (tamarisk) identification using a ResNet18
backbone. Designed for lightweight inference and fine-tuning workflows in research and field
monitoring.

## Features

- ResNet18-based binary classifier built on PyTorch + torchvision
- Simple CLI for batch inference
- Training entry point for fine-tuning on your own imagery
- Clean, pip-installable package layout

## Installation

```bash
pip3 install kineo_tamarisk
```

For development (Hatch):

```bash
hatch env create
hatch run pytest
```

## Quick Start

### Python

```python
from pathlib import Path
from PIL import Image
import torch

from kineo_tamarisk.model.classifier import load_classifier, default_transforms

model = load_classifier(weights_path=None, pretrained=True)
model.eval()

transform = default_transforms()
image = Image.open("example.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

with torch.inference_mode():
    logits = model(input_tensor)
    prob = torch.sigmoid(logits)[0].item()

print("tamarisk" if prob >= 0.5 else "non_tamarisk", prob)
```

### CLI

```bash
kineo-tamarisk-infer --weights path/to/checkpoint.pt image1.jpg image2.jpg
```

For JSON output:

```bash
kineo-tamarisk-infer --json image1.jpg
```

## Training

The training script expects an ImageFolder-style directory layout. Use `train/` and optional `val/`:

```
my_dataset/
  train/
    tamarisk/
    non_tamarisk/
  val/
    tamarisk/
    non_tamarisk/
```

Run fine-tuning:

```bash
python scripts/train.py --data-dir my_dataset --output checkpoints/tamarisk.pt
```

If your dataset includes a `tamarisk` class name, the training script treats it as the positive
label for evaluation.

## Testing

```bash
hatch run pytest
```

## License

Apache-2.0. See `LICENSE`.
