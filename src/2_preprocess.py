"""
Preprocess cotton disease images into resized train/validation/test splits.

Expected input layout:
    data/raw/
        bacterial_blight/
        curl_virus/
        fusarium_wilt/
        healthy/

Generated output layout:
    data/processed/
        train/<class_name>/
        val/<class_name>/
        test/<class_name>/

Python 3.10+
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------------
# Central configuration
#
# Keep all project paths and preprocessing constants in one place so the script
# can be adjusted safely if the dataset layout or target image size changes.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CLASSES = ("bacterial_blight", "curl_virus", "fusarium_wilt", "healthy")
SPLITS = ("train", "val", "test")
IMAGE_SIZE = (224, 224)
RANDOM_STATE = 42
JPEG_QUALITY = 95
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}


# -----------------------------------------------------------------------------
# Augmentation
#
# These transforms are applied only to train images. Each train input produces
# one resized original and one augmented copy, so the train split doubles after
# successful processing.
# -----------------------------------------------------------------------------
def build_train_augmentation() -> A.Compose:
    """
    Define the train-only augmentation pipeline.

    Albumentations expects RGB images for color transforms. The script reads
    with OpenCV in BGR, converts to RGB before augmentation, and converts back
    to BGR before saving with cv2.imwrite.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=15, p=0.4),
            A.GaussianBlur(p=0.2),
        ]
    )


# -----------------------------------------------------------------------------
# Output directory management
#
# The processed directory is regenerated on every run. This keeps repeated runs
# deterministic and prevents old images from inflating the final split counts.
# -----------------------------------------------------------------------------
def reset_processed_directory() -> None:
    """
    Recreate data/processed from scratch.

    Preprocessing pipelines are commonly rerun while experimenting. Removing
    the old processed directory avoids stale files being counted together with
    newly generated images from a later run.
    """
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    for split in SPLITS:
        for class_name in CLASSES:
            (PROCESSED_DIR / split / class_name).mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Input discovery and validation
#
# Images are tested with cv2.imread before splitting. This makes the split ratios
# depend only on readable images and lets the pipeline continue if a few files
# are corrupt or unsupported.
# -----------------------------------------------------------------------------
def find_readable_images(class_dir: Path) -> list[Path]:
    """
    Return valid image paths from one raw class directory.

    Corrupt or unreadable files are skipped with a warning. Validation happens
    before splitting so the requested train/val/test ratios are computed only
    from images that can actually be processed.
    """
    image_paths = sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    readable_paths: list[Path] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARNING] Skipping unreadable image: {image_path}")
            continue
        readable_paths.append(image_path)

    return readable_paths


# -----------------------------------------------------------------------------
# Stratified class-by-class splitting
#
# Since every class is split independently with the same ratios, the final
# dataset remains stratified across train, validation, and test directories.
# -----------------------------------------------------------------------------
def split_class_images(image_paths: list[Path], class_name: str) -> dict[str, list[Path]]:
    """
    Split one class into 70% train, 15% validation, and 15% test.

    Splitting class-by-class preserves class balance across the full dataset.
    The second split divides the temporary 30% holdout equally into validation
    and test sets.
    """
    if len(image_paths) < 3:
        print(
            f"[WARNING] Class '{class_name}' has only {len(image_paths)} readable "
            "image(s). At least 3 are needed for train/val/test; assigning all "
            "available images to train."
        )
        return {"train": image_paths, "val": [], "test": []}

    train_paths, holdout_paths = train_test_split(
        image_paths,
        test_size=0.30,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    if len(holdout_paths) < 2:
        print(
            f"[WARNING] Class '{class_name}' holdout split produced fewer than "
            "2 images. Validation/test cannot both be populated; assigning "
            "holdout images to validation."
        )
        return {"train": train_paths, "val": holdout_paths, "test": []}

    val_paths, test_paths = train_test_split(
        holdout_paths,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    return {"train": train_paths, "val": val_paths, "test": test_paths}


# -----------------------------------------------------------------------------
# Image writing helpers
#
# All outputs are normalized to JPEG files after resizing to 224x224. Unique
# output names protect against accidental overwrites from duplicate raw stems.
# -----------------------------------------------------------------------------
def make_unique_output_path(destination_dir: Path, source_path: Path) -> Path:
    """
    Build a non-colliding .jpg output path.

    Raw datasets occasionally contain files with the same stem and different
    extensions, such as leaf.jpg and leaf.jpeg. This helper prevents accidental
    overwrites after normalizing all output images to .jpg.
    """
    candidate = destination_dir / f"{source_path.stem}.jpg"
    counter = 1

    while candidate.exists():
        candidate = destination_dir / f"{source_path.stem}_{counter}.jpg"
        counter += 1

    return candidate


def read_resize_image(image_path: Path) -> tuple[bool, object]:
    """
    Read an image with OpenCV and resize it to 224x224.

    A success flag is returned instead of raising so a single bad file cannot
    stop the entire preprocessing run.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return False, None

    resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return True, resized


def save_jpeg(image, output_path: Path) -> bool:
    """
    Save an OpenCV BGR image as a high-quality JPEG.

    OpenCV returns False on write failures, so the caller can keep final counts
    aligned with files that were actually written.
    """
    return cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


# -----------------------------------------------------------------------------
# Per-image processing
#
# The original image is resized and saved for every split. Train images also get
# one augmented sibling named with the '_aug.jpg' suffix.
# -----------------------------------------------------------------------------
def process_image(
    image_path: Path,
    output_dir: Path,
    *,
    augment: bool,
    augmentation: A.Compose,
) -> int:
    """
    Resize and save one image, optionally adding one augmented train copy.

    The original resized image is always attempted first. When augment=True,
    one augmented version is saved next to it with the suffix '_aug.jpg',
    doubling the number of train files for successfully processed images.
    """
    success, resized_image = read_resize_image(image_path)
    if not success:
        print(f"[WARNING] Skipping unreadable image during processing: {image_path}")
        return 0

    output_path = make_unique_output_path(output_dir, image_path)
    if not save_jpeg(resized_image, output_path):
        print(f"[WARNING] Failed to save image: {output_path}")
        return 0

    written_count = 1

    if augment:
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        augmented_rgb = augmentation(image=rgb_image)["image"]
        augmented_bgr = cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2BGR)

        augmented_path = output_path.with_name(f"{output_path.stem}_aug.jpg")
        if save_jpeg(augmented_bgr, augmented_path):
            written_count += 1
        else:
            print(f"[WARNING] Failed to save augmented image: {augmented_path}")

    return written_count


# -----------------------------------------------------------------------------
# Reporting and script entrypoint
#
# Counts are based on files successfully written, not merely files discovered.
# -----------------------------------------------------------------------------
def print_final_counts(counts: dict[str, dict[str, int]]) -> None:
    """
    Print the number of written files for every split/class pair.
    """
    print("\nFinal processed image counts:")
    for split in SPLITS:
        print(f"\n{split}/")
        for class_name in CLASSES:
            print(f"  {class_name}: {counts[split][class_name]}")


def main() -> None:
    """
    Run the full preprocessing workflow.

    Steps:
    1. Validate the expected raw class folders.
    2. Remove stale processed output and recreate split/class directories.
    3. Read and validate raw images class-by-class.
    4. Split each class into train/val/test.
    5. Resize every image to 224x224 and save it into data/processed.
    6. Apply one albumentations transform per train image to double train size.
    7. Print final counts for quick verification.
    """
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DIR}")

    missing_classes = [class_name for class_name in CLASSES if not (RAW_DIR / class_name).is_dir()]
    if missing_classes:
        raise FileNotFoundError(
            "Missing raw class folder(s): " + ", ".join(missing_classes)
        )

    reset_processed_directory()
    train_augmentation = build_train_augmentation()
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for class_name in CLASSES:
        class_dir = RAW_DIR / class_name
        readable_images = find_readable_images(class_dir)

        print(f"\nClass '{class_name}': {len(readable_images)} readable image(s)")
        split_paths = split_class_images(readable_images, class_name)

        for split, image_paths in split_paths.items():
            output_dir = PROCESSED_DIR / split / class_name
            use_augmentation = split == "train"

            for image_path in image_paths:
                counts[split][class_name] += process_image(
                    image_path,
                    output_dir,
                    augment=use_augmentation,
                    augmentation=train_augmentation,
                )

    print_final_counts(counts)


if __name__ == "__main__":
    main()
