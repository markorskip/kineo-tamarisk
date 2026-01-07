"""Thin training entry point for tamarisk classifier fine-tuning."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import TextIO

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights

from kineo_tamarisk.model.classifier import TamariskClassifier


def _imagenet_mean_std() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    weights = ResNet18_Weights.DEFAULT
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    meta = getattr(weights, "meta", {})
    if isinstance(meta, dict):
        mean = tuple(meta.get("mean", mean))
        std = tuple(meta.get("std", std))
    return mean, std


def build_train_transforms(image_size: int) -> transforms.Compose:
    mean, std = _imagenet_mean_std()
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_eval_transforms(image_size: int) -> transforms.Compose:
    mean, std = _imagenet_mean_std()
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _init_csv_logger(log_dir: Path) -> tuple[Path, csv.DictWriter, TextIO]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_metrics.csv"
    is_new = not log_path.exists()
    log_file = log_path.open("a", newline="")
    writer = csv.DictWriter(log_file, fieldnames=["epoch", "train_loss", "val_acc"])
    if is_new:
        writer.writeheader()
    return log_path, writer, log_file


def _init_tensorboard_logger(log_dir: Path, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        print("TensorBoard not available; install 'tensorboard' to enable logging.")
        return None
    return SummaryWriter(log_dir=str(log_dir / "tensorboard"))


def build_loaders(data_dir: Path, image_size: int, batch_size: int, num_workers: int):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_root = train_dir if train_dir.exists() else data_dir
    train_dataset = datasets.ImageFolder(train_root, transform=build_train_transforms(image_size))

    val_loader = None
    if val_dir.exists():
        val_dataset = datasets.ImageFolder(val_dir, transform=build_eval_transforms(image_size))
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return train_loader, val_loader, train_dataset.class_to_idx


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    positive_index: int | None,
) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if positive_index is not None:
            labels = (labels == positive_index).float()
        else:
            labels = labels.float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, positive_index: int | None
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if positive_index is not None:
                labels = (labels == positive_index).long()
            logits = model(images)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune tamarisk classifier.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset root folder.")
    parser.add_argument("--output", type=Path, default=Path("tamarisk.pt"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)

    train_loader, val_loader, class_to_idx = build_loaders(
        args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if len(class_to_idx) != 2:
        raise ValueError("Expected exactly 2 classes for binary classification.")

    positive_index = class_to_idx.get("tamarisk")
    model = TamariskClassifier(pretrained=not args.no_pretrained)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log_path, csv_writer, csv_file = _init_csv_logger(args.log_dir)
    tb_writer = _init_tensorboard_logger(args.log_dir, enabled=not args.no_tensorboard)

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, positive_index
            )
            val_acc = None
            if val_loader is not None:
                val_acc = evaluate(model, val_loader, device, positive_index)
                print(
                    f"Epoch {epoch}/{args.epochs} - loss: {train_loss:.4f} - val_acc: {val_acc:.4f}"
                )
            else:
                print(f"Epoch {epoch}/{args.epochs} - loss: {train_loss:.4f}")

            csv_writer.writerow(
                {"epoch": epoch, "train_loss": f"{train_loss:.6f}", "val_acc": val_acc}
            )
            csv_file.flush()
            if tb_writer is not None:
                tb_writer.add_scalar("loss/train", train_loss, epoch)
                if val_acc is not None:
                    tb_writer.add_scalar("metrics/val_acc", val_acc, epoch)
    finally:
        if tb_writer is not None:
            tb_writer.close()
        csv_file.close()

    print(f"Saved metrics to {log_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
        },
        args.output,
    )
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
