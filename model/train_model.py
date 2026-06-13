import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image, UnidentifiedImageError
import os
import pandas as pd
import ast
from sklearn.metrics import classification_report
import numpy as np


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

        self.base_dir = r"C:\Users\vedan\Desktop\GIT\ocular_disease_tkinter\images"
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

        target_str = self.data.iloc[idx]["target"]
        if isinstance(target_str, str):
            target = torch.tensor(ast.literal_eval(target_str), dtype=torch.float32)
        else:
            target = torch.tensor(target_str, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target


def train_model(
    csv_path="full_df.csv",
    model_path="model/model_weights.pth",
    epochs=5,              # reduced for speed
    batch_size=8,          # smaller for CPU efficiency
    lr=1e-4,
):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    dataset = OcularDataset(csv_path, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    num_classes = len(ast.literal_eval(dataset.data.iloc[0]["target"]))
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_val_loss = float("inf")
    patience, trigger_times = 2, 0

    print("Starting training...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
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
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).int()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Model improved. Saved to {model_path}")
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

    print("\nFinal Validation Performance:")
    print(classification_report(np.array(all_labels), np.array(all_preds), zero_division=0))


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
    disease_classes = [
        "Normal", "Diabetes", "Glaucoma", "Cataract",
        "Age-related Macular Degeneration", "Hypertension",
        "Pathological Myopia", "Other Diseases",
    ]
    with torch.no_grad():
        outputs = model(img_tensor.unsqueeze(0))
        probs = torch.sigmoid(outputs)
        preds = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()
        class_name = disease_classes[preds] if preds < len(disease_classes) else "Unknown"
        return preds, confidence, class_name


if __name__ == "__main__":
    train_model(
        csv_path=r"C:\Users\vedan\Desktop\GIT\ocular_disease_tkinter\full_df.csv",
        model_path="model/model_weights.pth",
        epochs=5,
        batch_size=8,
        lr=1e-4,
    )
