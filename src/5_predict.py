"""
Predict the cotton disease class for a single image.

Usage:
    python src/5_predict.py data/raw/curl_virus/sample.jpg

Expected input:
    models/best_model.pth

Python 3.10+
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
from torchvision import models, transforms


# -----------------------------------------------------------------------------
# Central configuration
#
# Paths and preprocessing constants match the training/evaluation scripts so
# single-image inference uses the same model contract.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pth"
IMAGE_SIZE = 224
LOW_CONFIDENCE_THRESHOLD = 0.60
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------------
# User input validation
#
# The script expects exactly one image path. It checks existence and common image
# extensions before attempting to open the file with Pillow.
# -----------------------------------------------------------------------------
def parse_image_path() -> Path:
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python src/5_predict.py path/to/image.jpg"
        )

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported image format. Supported formats: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )

    return image_path


# -----------------------------------------------------------------------------
# Image preprocessing
#
# This is the same deterministic transform used for validation and test data:
# resize to 224x224, convert to tensor, and normalize with ImageNet statistics.
# -----------------------------------------------------------------------------
def load_and_preprocess_image(image_path: Path) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Could not read image file: {image_path}") from exc

    return transform(image).unsqueeze(0)


# -----------------------------------------------------------------------------
# Checkpoint loading
#
# The class order is loaded from the checkpoint dict so printed probabilities
# match the model output indices exactly.
# -----------------------------------------------------------------------------
def load_checkpoint(device: torch.device) -> dict:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if "class_names" not in checkpoint:
        raise KeyError("Checkpoint is missing required key: 'class_names'")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing required key: 'model_state_dict'")

    return checkpoint


# -----------------------------------------------------------------------------
# Model reconstruction
#
# Rebuild EfficientNetB0 with the same classifier used for training:
# Dropout(0.3) followed by Linear(1280 -> number_of_classes).
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
# Prediction formatting
#
# Folder-style class names are converted into title case only for the headline
# prediction. The probability table keeps raw checkpoint class names.
# -----------------------------------------------------------------------------
def format_prediction_name(class_name: str) -> str:
    return class_name.replace("_", " ").title()


def print_prediction(
    image_path: Path,
    class_names: list[str],
    probabilities: torch.Tensor,
) -> None:
    best_index = int(torch.argmax(probabilities).item())
    best_class = class_names[best_index]
    confidence = float(probabilities[best_index].item())
    label_width = max(len(class_name) for class_name in class_names)

    print("-" * 40)
    print(f"Image      : {image_path}")
    print(f"Prediction : {format_prediction_name(best_class)}")
    print(f"Confidence : {confidence * 100:.1f}%")
    print("-" * 40)
    print("All class probabilities:")
    for class_name, probability in zip(class_names, probabilities.tolist()):
        print(f"  {class_name:<{label_width}} : {probability * 100:.1f}%")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        print("Low confidence — manual check recommended")


# -----------------------------------------------------------------------------
# Main inference workflow
#
# Validate the input image, load the trained checkpoint, preprocess the image,
# run one forward pass with torch.no_grad(), and print probabilities.
# -----------------------------------------------------------------------------
def main() -> None:
    try:
        image_path = parse_image_path()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = load_checkpoint(device)
        class_names = checkpoint["class_names"]
        model = build_model(checkpoint, device)

        image_tensor = load_and_preprocess_image(image_path).to(device)
        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

        print_prediction(image_path, class_names, probabilities)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
