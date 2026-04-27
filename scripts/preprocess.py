"""
Task 5: Preprocess MRI Images
- Resize all images to 224x224 (standard for CNN/ResNet input)
- Convert all images to RGB (3-channel) for transfer learning
- Normalize pixel values to [0, 1] float32
- Save processed images to data/processed/raw_preprocessed/<class>/
- Outputs a preprocessing report and channel statistics

Run from project root after download_dataset.py:
  python scripts/preprocess.py
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import json

RAW_DIR     = "data/raw"
PROC_DIR    = "data/processed/raw_preprocessed"
CLASSES     = ["glioma", "meningioma", "no_tumor", "pituitary"]
TARGET_SIZE = (224, 224)    # H x W — standard ResNet/EfficientNet input


def preprocess_image(path: str) -> np.ndarray | None:
    """
    Load an MRI image, convert BGR->RGB, resize to TARGET_SIZE, normalize to [0,1].

    Returns:
        float32 numpy array of shape (224, 224, 3), or None if image is unreadable.
    """
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def preprocess_all(raw_dir: str = RAW_DIR, proc_dir: str = PROC_DIR) -> dict:
    """
    Preprocess all images from raw_dir into proc_dir.

    Args:
        raw_dir:  Source directory containing class subfolders.
        proc_dir: Destination directory for preprocessed images.

    Returns:
        Report dict with processed/skipped counts per class and error paths.
    """
    os.makedirs("outputs/metrics", exist_ok=True)
    report = {"target_size": list(TARGET_SIZE), "classes": {}, "errors": []}

    print(f"Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} px (RGB)")
    print(f"Source     : {raw_dir}")
    print(f"Destination: {proc_dir}")
    print("-" * 50)

    for cls in CLASSES:
        src_dir  = os.path.join(raw_dir, cls)
        dest_dir = os.path.join(proc_dir, cls)

        # Check source BEFORE creating destination to avoid orphan dirs
        if not os.path.exists(src_dir):
            print(f"  SKIP {cls:<15s}: source not found at {src_dir}")
            report["classes"][cls] = {"processed": 0, "skipped": 0, "status": "source_missing"}
            continue

        files = [
            f for f in os.listdir(src_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not files:
            print(f"  SKIP {cls:<15s}: source directory is empty")
            report["classes"][cls] = {"processed": 0, "skipped": 0, "status": "source_empty"}
            continue

        os.makedirs(dest_dir, exist_ok=True)
        processed = 0

        for fname in files:
            src_path  = os.path.join(src_dir, fname)
            dest_path = os.path.join(dest_dir, Path(fname).stem + ".png")

            img = preprocess_image(src_path)
            if img is None:
                report["errors"].append(src_path)
                continue

            # Convert back to uint8 BGR for cv2.imwrite
            save_img = (img * 255).astype(np.uint8)
            cv2.imwrite(dest_path, cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))
            processed += 1

        skipped = len(files) - processed
        report["classes"][cls] = {"processed": processed, "skipped": skipped, "status": "ok"}
        print(f"  OK   {cls:<15s}: {processed}/{len(files)} processed  ({skipped} skipped)")

    report_path = "outputs/metrics/preprocessing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    total_processed = sum(v.get("processed", 0) for v in report["classes"].values())
    print("-" * 50)
    print(f"Total processed: {total_processed} images")
    print(f"Errors         : {len(report['errors'])}")
    print(f"Report saved   : {report_path}")

    return report


def compute_dataset_stats(proc_dir: str = PROC_DIR) -> dict | None:
    """
    Compute per-channel mean and std across all preprocessed images.
    Uses Welford-style accumulation: averages per-image channel means,
    then derives approximate dataset std. Values are used for DataLoader
    normalization in Task 9.

    Returns:
        Dict with 'mean' and 'std' lists (3 values each), or None if no images.
    """
    pixel_sum    = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    count        = 0

    for cls in CLASSES:
        cls_dir = os.path.join(proc_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(".png"):
                continue
            img = cv2.imread(os.path.join(cls_dir, fname))
            if img is None:                             # Guard: skip corrupt files
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            per_img_mean   = img.reshape(-1, 3).mean(axis=0)
            per_img_sq     = (img.reshape(-1, 3) ** 2).mean(axis=0)
            pixel_sum    += per_img_mean
            pixel_sq_sum += per_img_sq
            count        += 1

    if count == 0:
        print("No processed images found. Run preprocess_all() first.")
        return None

    mean = pixel_sum    / count
    std  = np.sqrt(np.maximum(pixel_sq_sum / count - mean ** 2, 0))   # clip negatives from float precision

    print("\nDataset channel statistics (use these in DataLoader normalization):")
    print(f"  Mean : R={mean[0]:.4f}, G={mean[1]:.4f}, B={mean[2]:.4f}")
    print(f"  Std  : R={std[0]:.4f},  G={std[1]:.4f},  B={std[2]:.4f}")

    stats       = {"mean": mean.tolist(), "std": std.tolist(), "n_images": count}
    stats_path  = "outputs/metrics/channel_stats.json"
    os.makedirs("outputs/metrics", exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved to {stats_path}  (based on {count} images)")

    return stats


if __name__ == "__main__":
    if not os.path.isdir(RAW_DIR):
        print(f"ERROR: '{RAW_DIR}' not found.")
        print("  Run from the project root: python scripts/preprocess.py")
        sys.exit(1)

    print("=" * 50)
    print("  TASK 5: PREPROCESSING")
    print("=" * 50)
    preprocess_all()

    print("\n" + "=" * 50)
    print("  CHANNEL STATISTICS")
    print("=" * 50)
    compute_dataset_stats()
