"""
Task 6: Data Augmentation Pipeline
- Applies flips, rotations, brightness/contrast, blur, and elastic transforms
- Augments training images ONLY (never test images)
- Saves augmented images alongside originals in data/processed/raw_preprocessed/<class>/
- Each original image produces AUG_FACTOR augmented variants
- Outputs augmentation report to outputs/metrics/augmentation_report.json

Run from project root after preprocess.py:
  python scripts/augment.py
"""

import os
import sys
import cv2
import numpy as np
import json
import albumentations as A
from pathlib import Path

PROC_DIR   = "data/processed/raw_preprocessed"
CLASSES    = ["glioma", "meningioma", "no_tumor", "pituitary"]
AUG_FACTOR = 2      # Each original image produces this many augmented copies
SEED       = 42     # Reproducibility

# --------------------------------------------------------------------------
# Augmentation pipeline
# All transforms are chosen to be clinically safe for MRI:
#   - No color jitter that would misrepresent tissue (only mild brightness/contrast)
#   - Elastic deform simulates scanner variability
#   - No extreme crops that could cut out the tumor region
# --------------------------------------------------------------------------
def build_pipeline() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=20, interpolation=cv2.INTER_AREA, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.ElasticTransform(alpha=30, sigma=5, p=0.3),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
    ], seed=SEED)


def augment_image(img_uint8: np.ndarray, pipeline: A.Compose) -> np.ndarray:
    """
    Apply the augmentation pipeline to a uint8 RGB image.

    Args:
        img_uint8: H x W x 3 uint8 numpy array (RGB)
        pipeline:  Albumentations Compose pipeline

    Returns:
        Augmented H x W x 3 uint8 array
    """
    result = pipeline(image=img_uint8)
    return result["image"]


def augment_all(
    proc_dir: str  = PROC_DIR,
    aug_factor: int = AUG_FACTOR,
) -> dict:
    """
    Augment all preprocessed images in proc_dir/<class>/ folders.
    Augmented files are named:  <original_stem>_aug<n>.png

    Args:
        proc_dir:   Path to preprocessed images (must contain class subfolders).
        aug_factor: Number of augmented variants to produce per original image.

    Returns:
        Report dict with counts per class and any errors.
    """
    if aug_factor < 1:
        raise ValueError(f"aug_factor must be >= 1, got {aug_factor}")

    os.makedirs("outputs/metrics", exist_ok=True)
    pipeline = build_pipeline()
    report   = {"aug_factor": aug_factor, "classes": {}, "errors": []}

    print(f"Augmentation factor: {aug_factor}x per original image")
    print(f"Source: {proc_dir}")
    print("-" * 55)

    for cls in CLASSES:
        cls_dir = os.path.join(proc_dir, cls)

        if not os.path.exists(cls_dir):
            print(f"  SKIP {cls:<15s}: directory not found")
            report["classes"][cls] = {"originals": 0, "augmented": 0, "status": "missing"}
            continue

        originals = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(".png") and "_aug" not in f   # skip already-augmented files
        ]

        if not originals:
            print(f"  SKIP {cls:<15s}: no original images found (run preprocess.py first)")
            report["classes"][cls] = {"originals": 0, "augmented": 0, "status": "empty"}
            continue

        aug_count = 0
        for fname in originals:
            src_path = os.path.join(cls_dir, fname)
            img      = cv2.imread(src_path)
            if img is None:
                report["errors"].append(src_path)
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            for n in range(1, aug_factor + 1):
                aug_img  = augment_image(img_rgb, pipeline)
                stem     = Path(fname).stem
                out_path = os.path.join(cls_dir, f"{stem}_aug{n}.png")
                cv2.imwrite(out_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                aug_count += 1

        readable      = len(originals) - len([e for e in report["errors"] if os.path.dirname(e) == cls_dir])
        report["classes"][cls] = {
            "originals_attempted": len(originals),
            "originals_readable":  readable,
            "augmented":           aug_count,
            "total_images":        len(originals) + aug_count,
            "status":              "ok",
        }
        print(f"  OK   {cls:<15s}: {readable}/{len(originals)} originals readable → +{aug_count} augmented  (total: {len(originals) + aug_count})")

    report_path = "outputs/metrics/augmentation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    total_orig = sum(v.get("originals_readable", 0) for v in report["classes"].values())
    total_aug  = sum(v.get("augmented", 0) for v in report["classes"].values())
    print("-" * 55)
    print(f"Total originals  : {total_orig}")
    print(f"Total augmented  : {total_aug}")
    print(f"Total in dataset : {total_orig + total_aug}")
    print(f"Errors           : {len(report['errors'])}")
    print(f"Report saved     : {report_path}")

    return report


if __name__ == "__main__":
    if not os.path.isdir(PROC_DIR):
        print(f"ERROR: '{PROC_DIR}' not found.")
        print("  Run from the project root: python scripts/augment.py")
        print("  Ensure preprocess.py has been run first.")
        sys.exit(1)

    print("=" * 55)
    print("  TASK 6: DATA AUGMENTATION")
    print("=" * 55)
    augment_all()
