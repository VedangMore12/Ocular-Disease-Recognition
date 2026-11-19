"""Faster isolated training using a reusable preprocessing cache.

This module intentionally writes only below model/cached_run/ by default.
It does not modify model/model_weights.pth or the existing training outputs.
"""

import argparse
import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

from model.train_model import (
    DISEASE_CLASSES,
    _get_split_indices,
    _get_training_class_weights,
    _initialize_data_worker,
)
from utils.preprocess import enhance_fundus_image


NORMALIZE = transforms.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
)


def _read_dataset(csv_path):
    data = pd.read_csv(csv_path)
    required = {"filename", "target"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    base_dir = Path(csv_path).resolve().parent / "images"
    data["full_path"] = data["filename"].map(
        lambda value: str(base_dir / Path(str(value)).name)
    )
    data = data[data["full_path"].map(os.path.exists)].reset_index(drop=True)
    if data.empty:
        raise RuntimeError(f"No valid images found under {base_dir}")
    return data


def prepare_cache(data, cache_dir):
    """Create enhanced 224px PNGs once and return their paths."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = []
    created = 0

    for index, row in data.iterrows():
        source = Path(row["full_path"])
        cache_path = cache_dir / f"{index:05d}.png"
        metadata_path = cache_path.with_suffix(".json")
        source_stat = source.stat()
        expected = {
            "source": str(source),
            "size": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
        }

        valid_cache = False
        if cache_path.exists() and metadata_path.exists():
            try:
                valid_cache = json.loads(metadata_path.read_text()) == expected
            except (OSError, json.JSONDecodeError):
                valid_cache = False

        if not valid_cache:
            try:
                with Image.open(source) as image:
                    enhanced = enhance_fundus_image(image)
                    enhanced = enhanced.resize((224, 224), Image.Resampling.BILINEAR)
                    enhanced.save(cache_path, format="PNG")
                metadata_path.write_text(json.dumps(expected), encoding="utf-8")
                created += 1
            except (OSError, UnidentifiedImageError) as error:
                raise RuntimeError(f"Unable to cache image {source}: {error}") from error
        cache_paths.append(str(cache_path))

    print(f"Preprocessing cache ready: {len(cache_paths)} images ({created} created)")
    return cache_paths


class CachedOcularDataset(Dataset):
    def __init__(self, data, cache_paths, transform):
        self.data = data.reset_index(drop=True)
        self.cache_paths = cache_paths
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            with Image.open(self.cache_paths[index]) as image:
                image = image.convert("RGB")
                if self.transform:
                    image = self.transform(image)
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(
                f"Unable to load cached image {self.cache_paths[index]}"
            ) from error

        target = torch.tensor(
            ast.literal_eval(self.data.iloc[index]["target"]),
            dtype=torch.float32,
        )
        return image, target


def _make_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for layer in (model.layer3, model.layer4, model.fc):
        for parameter in layer.parameters():
            parameter.requires_grad = True
    return model


def _evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    predictions = []
    labels = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            outputs = model(images)
            total_loss += criterion(outputs, targets.argmax(dim=1)).item()
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            labels.extend(targets.argmax(dim=1).cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=range(num_classes),
        average="macro",
        zero_division=0,
    )
    return {
        "val_loss": total_loss / len(loader),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "report": classification_report(
            labels,
            predictions,
            labels=range(num_classes),
            target_names=DISEASE_CLASSES if num_classes == len(DISEASE_CLASSES) else None,
            zero_division=0,
        ),
    }


def train_cached(
    csv_path,
    output_dir,
    epochs=10,
    batch_size=32,
    num_workers=2,
    learning_rate=1e-4,
    selection_metric="macro_f1",
    device_name="auto",
):
    torch.manual_seed(42)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Install a CUDA-enabled "
            "PyTorch build and use a machine with an NVIDIA GPU."
        )
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
        if device_name != "auto" else "cpu"
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"

    data = _read_dataset(csv_path)
    cache_paths = prepare_cache(data, cache_dir)
    train_indices, val_indices = _get_split_indices(data)
    num_classes = len(ast.literal_eval(data.iloc[0]["target"]))

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        NORMALIZE,
    ])
    validation_transform = transforms.Compose([
        transforms.ToTensor(),
        NORMALIZE,
    ])
    train_dataset = CachedOcularDataset(data, cache_paths, train_transform)
    validation_dataset = CachedOcularDataset(data, cache_paths, validation_transform)
    loader_options = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _initialize_data_worker if num_workers > 0 else None,
    }
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )

    model = _make_model(num_classes).to(device)
    weights = _get_training_class_weights(data, train_indices, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        [
            {"params": model.layer3.parameters(), "lr": learning_rate * 0.1},
            {"params": model.layer4.parameters(), "lr": learning_rate},
            {"params": model.fc.parameters(), "lr": learning_rate * 5},
        ],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_value = float("-inf")
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda").argmax(dim=1)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        scheduler.step()

        metrics = _evaluate(model, validation_loader, criterion, device, num_classes)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = train_loss / len(train_loader)
        history.append(metrics)
        print(
            f"Epoch [{epoch + 1}/{epochs}] - "
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Val Loss: {metrics['val_loss']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.2%} | "
            f"Balanced Accuracy: {metrics['balanced_accuracy']:.2%} | "
            f"Macro F1: {metrics['macro_f1']:.2%}"
        )

        value = -metrics["val_loss"] if selection_metric == "val_loss" else metrics[selection_metric]
        if value > best_value:
            best_value = value
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            torch.save(best_state, output_dir / "model_weights.pth")
            print(f"Model improved. Saved to {output_dir / 'model_weights.pth'}")

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output_dir / "best_validation_report.txt").write_text(
        history[max(range(len(history)), key=lambda index: (
            -history[index]["val_loss"] if selection_metric == "val_loss"
            else history[index][selection_metric]
        ))]["report"],
        encoding="utf-8",
    )
    print(f"Device: {device}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with cached preprocessing.")
    parser.add_argument("--csv-path", default="full_df.csv")
    parser.add_argument("--output-dir", default="model/cached_run")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--selection-metric",
        choices=["val_loss", "accuracy", "balanced_accuracy", "macro_f1"],
        default="macro_f1",
    )
    parser.add_argument(
    "--device",
    dest="device_name",
    choices=["auto", "cuda", "cpu"],
    default="auto",
    )
    args = parser.parse_args()
    train_cached(**vars(args))
