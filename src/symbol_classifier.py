"""
Lightweight CNN Symbol Classifier
Classifies 32x32 grayscale patches into symbol types or background.

Architecture: 3-layer CNN with ~100K parameters
- Designed for 32x32 input (21x21 padded to 32x32)
- 9 classes: 8 symbol types + background
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─── Model ──────────────────────────────────────────────────────────────

class SymbolClassifierCNN(nn.Module):
    """
    Lightweight CNN for symbol classification.
    Input: 32x32 grayscale (1 channel)
    Output: 9 classes (8 symbols + background)
    """
    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Dataset ────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "filled_circle", "open_circle",
    "filled_square", "open_square",
    "filled_triangle", "open_triangle",
    "filled_diamond", "open_diamond",
    "background",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


class PatchDataset(Dataset):
    """Dataset of extracted patches for classification."""

    def __init__(self, patches_dir: str, metadata_path: str, transform=None,
                 max_bg_ratio: float = 3.0):
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.patches_dir = patches_dir
        self.transform = transform

        # Balance dataset: limit background samples
        symbol_count = sum(1 for m in self.metadata if m["class"] != "background")
        max_bg = int(symbol_count * max_bg_ratio)

        balanced = []
        bg_count = 0
        for m in self.metadata:
            if m["class"] == "background":
                if bg_count < max_bg:
                    balanced.append(m)
                    bg_count += 1
            else:
                balanced.append(m)

        self.metadata = balanced

        # Compute class weights for balanced sampling
        class_counts = {}
        for m in self.metadata:
            cls = m["class"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        self.class_counts = class_counts
        total = len(self.metadata)
        self.class_weights = {
            cls: total / (len(class_counts) * cnt)
            for cls, cnt in class_counts.items()
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        img_path = os.path.join(self.patches_dir, meta["file"])
        img = Image.open(img_path).convert('L')  # grayscale

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        label = CLASS_TO_ID[meta["class"]]
        return img, label

    def get_sample_weights(self):
        """Get per-sample weights for WeightedRandomSampler."""
        weights = []
        for m in self.metadata:
            weights.append(self.class_weights[m["class"]])
        return weights


# ─── Training ───────────────────────────────────────────────────────────

def train_classifier(
    train_dir: str,
    val_dir: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "auto",
):
    """Train the symbol classifier."""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Training on: {device}")
    os.makedirs(output_dir, exist_ok=True)

    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load datasets
    train_dataset = PatchDataset(
        os.path.join(train_dir, "patches"),
        os.path.join(train_dir, "patch_metadata.json"),
        transform=train_transform,
        max_bg_ratio=2.0,
    )
    val_dataset = PatchDataset(
        os.path.join(val_dir, "patches"),
        os.path.join(val_dir, "patch_metadata.json"),
        transform=val_transform,
        max_bg_ratio=2.0,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Train class distribution: {train_dataset.class_counts}")

    # Weighted sampler for balanced training
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Model
    model = SymbolClassifierCNN(num_classes=len(CLASS_NAMES)).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_acc = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        n_train = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            n_train += imgs.size(0)

        train_loss /= n_train
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= total
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"  -> New best model saved (acc={val_acc:.4f})")

    # Final evaluation with best model
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth"),
                                      weights_only=True))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    report = classification_report(all_labels, all_preds,
                                    target_names=CLASS_NAMES, digits=4)
    print(f"\n{'='*60}")
    print("FINAL CLASSIFICATION REPORT (Best Model)")
    print(f"{'='*60}")
    print(report)

    # Save report
    with open(os.path.join(output_dir, "classification_report.txt"), 'w') as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss")

    ax2.plot(history["val_acc"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close(fig)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved to: {output_dir}")

    return model, history


# ─── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="data/synthetic/train")
    parser.add_argument("--val-dir", default="data/synthetic/val")
    parser.add_argument("--output", default="models/classifier_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_classifier(
        args.train_dir, args.val_dir, args.output,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )
