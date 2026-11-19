"""Recreate the original-style ResNet18 baseline in an isolated output folder.

This baseline intentionally does not use CLAHE/vessel enhancement, weighted
loss, or layer3/layer4 fine-tuning. It is kept separate from the current
training outputs for comparison.
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
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

from model.train_model import DISEASE_CLASSES, _initialize_data_worker


NORMALIZE = transforms.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
)


class BaselineDataset(Dataset):
    def __init__(self, csv_path, transform):
        data = pd.read_csv(csv_path)
        if "filename" not in data.columns or "target" not in data.columns:
            raise ValueError("CSV must contain 'filename' and 'target' columns")

        image_dir = Path(csv_path).resolve().parent / "images"
        data["full_path"] = data["filename"].map(
            lambda value: str(image_dir / Path(str(value)).name)
        )
        self.data = data[data["full_path"].map(os.path.exists)].reset_index(drop=True)
        if self.data.empty:
            raise RuntimeError(f"No valid images found under {image_dir}")
        self.transform = transform
        print(f"Loaded {len(self.data)} baseline images from {image_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data.iloc[index]["full_path"]
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"Unable to read image {path}") from error

        target = torch.tensor(
            ast.literal_eval(self.data.iloc[index]["target"]),
            dtype=torch.float32,
        )
        return self.transform(image), target


def _split_indices(data):
    if "ID" not in data.columns:
        raise ValueError("The baseline requires patient IDs for a grouped split.")
    return next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            data, groups=data["ID"]
        )
    )


def _make_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return model


def _evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    loss_total = 0.0
    predictions, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            outputs = model(images)
            loss_total += criterion(outputs, targets).item()
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
        "val_loss": loss_total / len(loader),
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


def train_baseline(
    csv_path,
    output_dir,
    epochs=15,
    batch_size=32,
    num_workers=2,
    learning_rate=1e-4,
    patience=4,
    device_name="auto",
):
    torch.manual_seed(42)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Install a CUDA-enabled "
            "PyTorch build on a machine with an NVIDIA GPU."
        )
    device_value = (
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    device = torch.device(device_value)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        NORMALIZE,
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        NORMALIZE,
    ])
    train_dataset = BaselineDataset(csv_path, train_transform)
    validation_dataset = BaselineDataset(csv_path, validation_transform)
    train_indices, validation_indices = _split_indices(train_dataset.data)
    num_classes = len(ast.literal_eval(train_dataset.data.iloc[0]["target"]))

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
        Subset(validation_dataset, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )

    model = _make_model(num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_state = None
    history = []
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

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

        if metrics["val_loss"] < best_loss:
            best_loss = metrics["val_loss"]
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            torch.save(best_state, output_dir / "model_weights.pth")
            stale_epochs = 0
            print(f"Model improved. Saved to {output_dir / 'model_weights.pth'}")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Baseline training did not produce a checkpoint.")
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output_dir / "best_validation_report.txt").write_text(
        min(history, key=lambda item: item["val_loss"])["report"],
        encoding="utf-8",
    )
    print(f"Device: {device}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an isolated ResNet18 baseline.")
    parser.add_argument("--csv-path", default="full_df.csv")
    parser.add_argument("--output-dir", default="model/baseline_run")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument(
        "--device",
        dest="device_name",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    args = parser.parse_args()
    train_baseline(**vars(args))
