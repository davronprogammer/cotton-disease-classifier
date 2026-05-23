"""
Train an EfficientNetB0 classifier for cotton disease recognition.

Expected input layout:
    data/processed/
        train/
            bacterial_blight/
            curl_virus/
            fusarium_wilt/
            healthy/
        val/
            bacterial_blight/
            curl_virus/
            fusarium_wilt/
            healthy/

Generated outputs:
    models/best_model.pth
    models/training_curves.png

Python 3.10+, PyTorch 2.0+
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Central configuration
#
# All paths, hyperparameters, and class metadata live here so training behavior
# is easy to audit and adjust without hunting through the script body.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "processed" / "val"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
CURVES_PATH = MODELS_DIR / "training_curves.png"

EXPECTED_CLASSES = ["bacterial_blight", "curl_virus", "fusarium_wilt", "healthy"]
NUM_CLASSES = len(EXPECTED_CLASSES)
IMAGE_SIZE = 224
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
FREEZE_EPOCHS = 5
RANDOM_SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------------
# Reproducibility
#
# A fixed seed makes shuffling and model initialization repeatable enough for
# practical experiments while still allowing CUDA to choose fast kernels.
# -----------------------------------------------------------------------------
def set_seed(seed: int = RANDOM_SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Data transforms
#
# Training uses lightweight torchvision augmentation. Validation keeps images
# deterministic and only resizes plus normalizes for ImageNet-pretrained weights.
# -----------------------------------------------------------------------------
def build_transforms() -> dict[str, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return {"train": train_transform, "val": val_transform}


# -----------------------------------------------------------------------------
# Dataset and DataLoader construction
#
# ImageFolder infers labels from class folder names. The class names are checked
# against the project contract and later saved into the model checkpoint.
# -----------------------------------------------------------------------------
def build_dataloaders() -> tuple[DataLoader, DataLoader, list[str]]:
    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
    if not VAL_DIR.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {VAL_DIR}")

    data_transforms = build_transforms()
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms["train"])
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=data_transforms["val"])

    if len(train_dataset) == 0:
        raise ValueError(f"No training images found in: {TRAIN_DIR}")
    if len(val_dataset) == 0:
        raise ValueError(f"No validation images found in: {VAL_DIR}")

    class_names = train_dataset.classes
    if class_names != EXPECTED_CLASSES:
        raise ValueError(
            "Unexpected class folder order/names. "
            f"Expected {EXPECTED_CLASSES}, got {class_names}."
        )
    if val_dataset.classes != class_names:
        raise ValueError(
            "Train and validation class folders do not match. "
            f"Train={class_names}, val={val_dataset.classes}."
        )

    num_workers = min(4, os.cpu_count() or 0)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")
    print(f"Class names: {class_names}")

    return train_loader, val_loader, class_names


# -----------------------------------------------------------------------------
# Model creation
#
# EfficientNetB0 is loaded with ImageNet weights. The original classifier is
# replaced by Dropout(0.3) followed by Linear(1280 -> 4), matching the task.
# -----------------------------------------------------------------------------
def build_model(device: torch.device) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1280, out_features=NUM_CLASSES),
    )
    return model.to(device)


# -----------------------------------------------------------------------------
# Freeze/unfreeze control
#
# The feature extractor stays frozen for the first five epochs so the new
# classifier can stabilize. Starting at epoch six, all parameters are trainable.
# -----------------------------------------------------------------------------
def set_base_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


# -----------------------------------------------------------------------------
# Epoch execution
#
# One helper handles both training and validation. Gradients are enabled only in
# training mode, keeping validation faster and memory-efficient.
# -----------------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    progress_label = "train" if is_training else "val"

    progress_bar = tqdm(dataloader, desc=progress_label, leave=False)
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        current_loss = running_loss / total_samples
        current_acc = running_correct / total_samples
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


# -----------------------------------------------------------------------------
# Checkpointing
#
# The best model is selected by validation accuracy. Metadata needed for later
# inference is stored with the weights, especially the class_names list.
# -----------------------------------------------------------------------------
def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_acc: float,
    class_names: list[str],
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "class_names": class_names,
        "model_name": "efficientnet_b0",
        "image_size": IMAGE_SIZE,
        "normalize_mean": IMAGENET_MEAN,
        "normalize_std": IMAGENET_STD,
    }
    torch.save(checkpoint, BEST_MODEL_PATH)


# -----------------------------------------------------------------------------
# Training curves
#
# Loss and accuracy histories are plotted after training and saved into models/
# so the run can be inspected without parsing console logs.
# -----------------------------------------------------------------------------
def save_training_curves(history: dict[str, list[float]]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CURVES_PATH, dpi=200, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Main training workflow
#
# The script wires together data, model, optimizer, scheduler, staged freezing,
# metric tracking, best-checkpoint saving, and final curve export.
# -----------------------------------------------------------------------------
def main() -> None:
    set_seed()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = build_dataloaders()
    model = build_model(device)

    set_base_trainable(model, trainable=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_acc = -1.0

    for epoch in range(1, EPOCHS + 1):
        if epoch == 1:
            print(f"Freezing EfficientNet base layers for epochs 1-{FREEZE_EPOCHS}.")
        if epoch == FREEZE_EPOCHS + 1:
            set_base_trainable(model, trainable=True)
            print("Unfroze all EfficientNet layers.")

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"[Epoch {epoch}/{EPOCHS}] "
            f"{train_loss:.4f} | {val_loss:.4f} | "
            f"{train_acc:.4f} | {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, class_names)
            print(f"Saved new best model: {BEST_MODEL_PATH} (val_acc={val_acc:.4f})")

    save_training_curves(history)
    print(f"Training complete. Best val_acc: {best_val_acc:.4f}")
    print(f"Training curves saved to: {CURVES_PATH}")


if __name__ == "__main__":
    main()
