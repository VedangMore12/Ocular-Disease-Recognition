import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image, UnidentifiedImageError
import os
import pandas as pd
import ast
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

from utils.preprocess import enhance_fundus_image


DISEASE_CLASSES = [
    "Normal",
    "Diabetes",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration",
    "Hypertension",
    "Pathological Myopia",
    "Other Diseases",
]


class OcularDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found at: {csv_file}")

        self.data = pd.read_csv(csv_file)
        self.transform = transform

        if "filename" not in self.data.columns:
            raise ValueError("CSV must contain a 'filename' column")
        if "target" not in self.data.columns:
            raise ValueError("CSV must contain a 'target' column for labels")

        self.base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "images")
        )
        self.data["full_path"] = self.data["filename"].apply(
            lambda x: os.path.join(self.base_dir, os.path.basename(str(x)))
        )
        self.data = self.data[self.data["full_path"].apply(os.path.exists)].reset_index(drop=True)

        if len(self.data) == 0:
            raise RuntimeError("No valid image paths found. Check your 'images' folder or CSV file.")

        print(f"Loaded {len(self.data)} valid images from {self.base_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["full_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            return self.__getitem__((idx + 1) % len(self.data))
        image = enhance_fundus_image(image)

        target_str = self.data.iloc[idx]["target"]
        if isinstance(target_str, str):
            target = torch.tensor(ast.literal_eval(target_str), dtype=torch.float32)
        else:
            target = torch.tensor(target_str, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target


def _get_split_indices(data):
    if len(data) < 2:
        raise ValueError("At least two valid images are required for a train/validation split.")

    groups = data["ID"] if "ID" in data.columns else data.index
    if pd.Series(groups).nunique() < 2:
        raise ValueError("At least two distinct patient IDs are required for a grouped split.")

    return next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            data, groups=groups
        )
    )


def _get_class_names(num_classes):
    if num_classes == len(DISEASE_CLASSES):
        return DISEASE_CLASSES
    return [f"Class {index}" for index in range(num_classes)]


def _get_validation_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def _get_training_class_weights(data, train_indices, num_classes):
    class_ids = [
        int(np.argmax(ast.literal_eval(data.iloc[index]["target"])))
        for index in train_indices
    ]
    class_counts = np.bincount(class_ids, minlength=num_classes)
    if np.any(class_counts == 0):
        absent_classes = np.flatnonzero(class_counts == 0).tolist()
        raise ValueError(
            f"Training split has no examples for classes: {absent_classes}"
        )
    weights = len(class_ids) / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def _initialize_data_worker(_worker_id):
    del _worker_id
    torch.set_num_threads(1)
    cv2.setNumThreads(0)


def train_model(
    csv_path="full_df.csv",
    model_path="model/model_weights.pth",
    epochs=15,
    batch_size=8,
    lr=1e-4,
    initial_weights_path=None,
    num_workers=0,
    train_layer3=True,
    patience=4,
):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    dataset = OcularDataset(csv_path, train_transform)
    train_indices, val_indices = _get_split_indices(dataset.data)
    val_dataset = OcularDataset(csv_path, _get_validation_transform())
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(val_dataset, val_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_options = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _initialize_data_worker if num_workers > 0 else None,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )

    num_classes = len(ast.literal_eval(dataset.data.iloc[0]["target"]))
    pretrained_weights = (
        None
        if initial_weights_path is not None
        else models.ResNet18_Weights.IMAGENET1K_V1
    )
    model = models.resnet18(weights=pretrained_weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if initial_weights_path is not None:
        if not os.path.isfile(initial_weights_path):
            raise FileNotFoundError(
                f"Initial model weights not found at: {initial_weights_path}"
            )
        state_dict = torch.load(
            initial_weights_path,
            map_location=torch.device("cpu"),
        )
        model.load_state_dict(state_dict, strict=True)

    for param in model.parameters():
        param.requires_grad = False
    trainable_layers = [model.layer4, model.fc]
    if train_layer3:
        trainable_layers.insert(0, model.layer3)
    for layer in trainable_layers:
        for param in layer.parameters():
            param.requires_grad = True

    model = model.to(device)
    class_weights = _get_training_class_weights(
        dataset.data, train_indices, num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    parameter_groups = []
    if train_layer3:
        parameter_groups.append(
            {"params": model.layer3.parameters(), "lr": lr * 0.1}
        )
    parameter_groups.extend([
        {"params": model.layer4.parameters(), "lr": lr},
        {"params": model.fc.parameters(), "lr": lr * 5},
    ])
    optimizer = optim.AdamW(parameter_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    best_val_loss = float("inf")
    best_model_state = None
    trigger_times = 0

    print("Starting training...")
    if "ID" not in dataset.data.columns:
        print("Warning: CSV has no 'ID' column; split is image-level, not patient-level.")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            targets = labels.argmax(dim=1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=device.type == "cuda")
                labels = labels.to(device, non_blocking=device.type == "cuda")
                targets = labels.argmax(dim=1)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.argmax(dim=1).cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            all_labels,
            all_preds,
            labels=range(num_classes),
            average="macro",
            zero_division=0,
        )
        print(
            f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f} "
            f"| Val Loss: {avg_val_loss:.4f} "
            f"| Accuracy: {val_accuracy:.2%} "
            f"| Balanced Accuracy: {val_balanced_accuracy:.2%} "
            f"| Macro Precision: {val_precision:.2%} "
            f"| Macro Recall: {val_recall:.2%} "
            f"| Macro F1: {val_f1:.2%}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            model_dir = os.path.dirname(model_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            torch.save(best_model_state, model_path)
            print(f"Model improved. Saved to {model_path}")
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

    if best_model_state is None:
        raise RuntimeError("Training did not produce a best model checkpoint.")

    model.load_state_dict(best_model_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            outputs = model(imgs.to(device))
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.argmax(dim=1).cpu().numpy())

    print("\nFinal Validation Performance:")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(all_labels, all_preds):.2%}")
    print(
        classification_report(
            np.array(all_labels),
            np.array(all_preds),
            labels=range(num_classes),
            target_names=DISEASE_CLASSES if num_classes == len(DISEASE_CLASSES) else None,
            zero_division=0,
        )
    )


def evaluate_model(model, csv_path="full_df.csv", batch_size=16, device=None):
    dataset = OcularDataset(csv_path, _get_validation_transform())
    _, val_indices = _get_split_indices(dataset.data)
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
    )

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            outputs = model(imgs.to(device))
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.argmax(dim=1).cpu().numpy())

    num_classes = len(ast.literal_eval(dataset.data.iloc[0]["target"]))
    class_names = _get_class_names(num_classes)
    matrix = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    true_positives = np.diag(matrix).astype(float)
    predicted_support = matrix.sum(axis=0).astype(float)
    actual_support = matrix.sum(axis=1).astype(float)
    class_precision = np.divide(
        true_positives,
        predicted_support,
        out=np.zeros(num_classes, dtype=float),
        where=predicted_support != 0,
    )
    class_recall = np.divide(
        true_positives,
        actual_support,
        out=np.zeros(num_classes, dtype=float),
        where=actual_support != 0,
    )
    class_f1 = np.divide(
        2 * class_precision * class_recall,
        class_precision + class_recall,
        out=np.zeros(num_classes, dtype=float),
        where=(class_precision + class_recall) != 0,
    )
    macro_precision = float(class_precision.mean())
    macro_recall = float(class_recall.mean())
    macro_f1 = float(class_f1.mean())
    weighted_f1 = float(
        np.average(class_f1, weights=actual_support)
    )
    per_class = {
        name: {
            "precision": float(class_precision[index]),
            "recall": float(class_recall[index]),
            "f1-score": float(class_f1[index]),
            "support": int(actual_support[index]),
        }
        for index, name in enumerate(class_names)
    }
    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(all_labels, all_preds)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "class_names": class_names,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "validation_samples": len(all_labels),
    }


def load_trained_model(model_path=None, num_classes=8):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "model_weights.pth")


    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained weights not found at: {model_path}")

    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def predict_image(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor.unsqueeze(0))
        probs = torch.sigmoid(outputs)
        preds = int(torch.argmax(probs, dim=1).item())
        confidence = torch.max(probs).item()
        class_name = DISEASE_CLASSES[preds] if preds < len(DISEASE_CLASSES) else "Unknown"
        return preds, confidence, class_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ocular disease classifier.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use 5 epochs and train only ResNet layer4 and the classifier head.",
    )
    args = parser.parse_args()
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_model(
        csv_path=os.path.join(project_dir, "full_df.csv"),
        model_path=os.path.join(project_dir, "model", "model_weights.pth"),
        epochs=args.epochs if args.epochs is not None else (5 if args.fast else 15),
        batch_size=args.batch_size,
        lr=1e-4,
        num_workers=args.num_workers,
        train_layer3=not args.fast,
        patience=2 if args.fast else 4,
    )
