"""
Evaluate the trained EfficientNetB0 cotton disease classifier on the test set.

Expected inputs:
    models/best_model.pth
    data/processed/test/
        bacterial_blight/
        curl_virus/
        fusarium_wilt/
        healthy/

Generated outputs:
    models/confusion_matrix.png
    models/misclassified/

Python 3.10+
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# -----------------------------------------------------------------------------
# Central configuration
#
# Keep paths and evaluation settings together so the script remains easy to run
# after training without needing command-line arguments.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_PATH = MODELS_DIR / "best_model.pth"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
MISCLASSIFIED_DIR = MODELS_DIR / "misclassified"

EXPECTED_CLASSES = ["bacterial_blight", "curl_virus", "fusarium_wilt", "healthy"]
NUM_CLASSES = len(EXPECTED_CLASSES)
BATCH_SIZE = 32
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------------
# Test transforms
#
# Evaluation uses the same deterministic transform as validation: resize to
# 224x224, convert to tensor, and normalize with ImageNet statistics.
# -----------------------------------------------------------------------------
def build_test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# -----------------------------------------------------------------------------
# Checkpoint loading
#
# The checkpoint dict created by src/3_train.py stores class_names along with
# model weights. Loading those names prevents label-order mismatches at eval.
# -----------------------------------------------------------------------------
def load_checkpoint(device: torch.device) -> dict:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if "class_names" not in checkpoint:
        raise KeyError("Checkpoint is missing required key: 'class_names'")

    return checkpoint


# -----------------------------------------------------------------------------
# Model reconstruction
#
# Rebuild EfficientNetB0 with the same classifier head used during training,
# then load the saved best weights.
# -----------------------------------------------------------------------------
def build_model(checkpoint: dict, device: torch.device) -> nn.Module:
    class_names = checkpoint["class_names"]
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1280, out_features=len(class_names)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Dataset and DataLoader construction
#
# ImageFolder gives each sample a file path through dataset.samples. Those paths
# are used later to copy a few misclassified images for visual inspection.
# -----------------------------------------------------------------------------
def build_test_loader(class_names: list[str]) -> tuple[DataLoader, datasets.ImageFolder]:
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=build_test_transform())
    if len(test_dataset) == 0:
        raise ValueError(f"No test images found in: {TEST_DIR}")

    if test_dataset.classes != class_names:
        raise ValueError(
            "Test class folders do not match checkpoint class_names. "
            f"Checkpoint={class_names}, test={test_dataset.classes}."
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Test images: {len(test_dataset)}")
    print(f"Class names: {class_names}")
    return test_loader, test_dataset


# -----------------------------------------------------------------------------
# Inference
#
# Run the full test set with torch.no_grad() so PyTorch avoids gradient storage
# and keeps evaluation faster and lighter on memory.
# -----------------------------------------------------------------------------
def run_inference(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    all_true_labels: list[int] = []
    all_pred_labels: list[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu()

            all_true_labels.extend(labels.tolist())
            all_pred_labels.extend(predictions.tolist())

    return all_true_labels, all_pred_labels


# -----------------------------------------------------------------------------
# Metrics and reporting
#
# Print overall accuracy plus the sklearn classification report containing
# per-class precision, recall, and F1-score.
# -----------------------------------------------------------------------------
def print_metrics(
    true_labels: list[int],
    pred_labels: list[int],
    class_names: list[str],
) -> None:
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print(f"\nOverall accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(report)


# -----------------------------------------------------------------------------
# Confusion matrix plotting
#
# A labeled seaborn heatmap is saved to models/confusion_matrix.png for quick
# inspection of which classes are being confused.
# -----------------------------------------------------------------------------
def save_confusion_matrix(
    true_labels: list[int],
    pred_labels: list[int],
    class_names: list[str],
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(true_labels, pred_labels, labels=range(len(class_names)))

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")


# -----------------------------------------------------------------------------
# Misclassification export
#
# Save up to five incorrectly classified source images into models/misclassified
# with true and predicted labels embedded in each filename.
# -----------------------------------------------------------------------------
def save_misclassified_examples(
    test_dataset: datasets.ImageFolder,
    true_labels: list[int],
    pred_labels: list[int],
    class_names: list[str],
    max_images: int = 5,
) -> None:
    if MISCLASSIFIED_DIR.exists():
        shutil.rmtree(MISCLASSIFIED_DIR)
    MISCLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for index, (true_label, pred_label) in enumerate(zip(true_labels, pred_labels)):
        if true_label == pred_label:
            continue

        source_path = Path(test_dataset.samples[index][0])
        true_name = class_names[true_label]
        pred_name = class_names[pred_label]
        destination_name = (
            f"{saved_count + 1:02d}_true-{true_name}_pred-{pred_name}"
            f"_{source_path.name}"
        )
        shutil.copy2(source_path, MISCLASSIFIED_DIR / destination_name)
        saved_count += 1

        if saved_count >= max_images:
            break

    print(f"Saved {saved_count} misclassified image(s) to: {MISCLASSIFIED_DIR}")


# -----------------------------------------------------------------------------
# Main evaluation workflow
#
# Load the checkpoint, reconstruct the model, evaluate the complete test set,
# print metrics, save the confusion matrix, and export example mistakes.
# -----------------------------------------------------------------------------
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = load_checkpoint(device)
    class_names = checkpoint["class_names"]

    if class_names != EXPECTED_CLASSES:
        print(
            "[WARNING] Checkpoint class_names differ from expected project "
            f"classes. Using checkpoint order: {class_names}"
        )

    test_loader, test_dataset = build_test_loader(class_names)
    model = build_model(checkpoint, device)

    true_labels, pred_labels = run_inference(model, test_loader, device)
    print_metrics(true_labels, pred_labels, class_names)
    save_confusion_matrix(true_labels, pred_labels, class_names)
    save_misclassified_examples(test_dataset, true_labels, pred_labels, class_names)


if __name__ == "__main__":
    main()
