"""Command-line interface for tamarisk inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

from kineo_tamarisk.model.classifier import (
    TamariskClassifier,
    batch_images,
    default_transforms,
    load_classifier,
)


def _iter_images(paths: Iterable[str], image_size: int) -> tuple[list[str], torch.Tensor]:
    transform = default_transforms(image_size=image_size)
    tensors = []
    resolved_paths = []
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        with Image.open(resolved) as img:
            tensor = transform(img.convert("RGB"))
        tensors.append(tensor)
        resolved_paths.append(str(resolved))
    return resolved_paths, batch_images(tensors)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run tamarisk inference on one or more images.",
    )
    parser.add_argument("images", nargs="+", help="Paths to image files.")
    parser.add_argument("--weights", help="Path to fine-tuned weights.")
    parser.add_argument("--device", default="cpu", help="Device for inference (cpu/cuda).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold.")
    parser.add_argument("--image-size", type=int, default=224, help="Resize/crop size.")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained backbone.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output instead of plain text.",
    )
    return parser


def run_inference(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    model: TamariskClassifier = load_classifier(
        weights_path=args.weights,
        device=device,
        pretrained=not args.no_pretrained,
    )
    model.to(device)
    model.eval()

    paths, batch = _iter_images(args.images, image_size=args.image_size)
    batch = batch.to(device)
    predictions = model.predict(batch, threshold=args.threshold)

    if args.json:
        payload = [
            {
                "path": path,
                "probability": prediction.probability,
                "label": prediction.label,
            }
            for path, prediction in zip(paths, predictions)
        ]
        print(json.dumps(payload, indent=2))
    else:
        for path, prediction in zip(paths, predictions):
            print(
                f"{path}\tprobability={prediction.probability:.4f}\tlabel={prediction.label}"
            )
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(run_inference(args))


if __name__ == "__main__":
    main()
