"""
Task 7: Split Dataset into Train / Test Sets
- 1,400 images per class for training
- 400 images per class for testing (split into 4 batches of 100 each)
- Originals only (no augmented images) are used for splitting;
  augmented images follow their original into train only
- Stratified, reproducible split (seeded)
- Zero data leakage: an original and its augmented copies never span train/test
- Outputs split manifest to outputs/metrics/split_manifest.json

Run from project root after augment.py:
  python scripts/split_dataset.py
"""

import os
import sys
import shutil
import json
import random
from pathlib import Path

PROC_DIR    = "data/processed/raw_preprocessed"
TRAIN_DIR   = "data/processed/train"
TEST_DIR    = "data/processed/test"
CLASSES     = ["glioma", "meningioma", "no_tumor", "pituitary"]
TRAIN_COUNT = 1400   # originals per class
TEST_COUNT  = 400    # originals per class  (100 per batch × 4 batches)
N_BATCHES   = 4
SEED        = 42

# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

def _validate_counts(originals: list, cls: str, train_count: int, test_count: int) -> None:
    """Raise if the class doesn't have enough original images for the requested split."""
    required = train_count + test_count
    if len(originals) < required:
        raise ValueError(
            f"Class '{cls}' has only {len(originals)} original images, "
            f"but {required} are required ({train_count} train + {test_count} test). "
            f"Collect more images or lower train_count/test_count."
        )


def _build_sibling_map(cls_dir: str) -> dict[str, list[str]]:
    """
    Build a map of {original_stem -> [aug_file, ...]} in one os.listdir call.
    Avoids re-reading the directory for every original file.
    """
    sibling_map: dict[str, list[str]] = {}
    for fname in os.listdir(cls_dir):
        if "_aug" in fname and fname.endswith(".png"):
            # e.g. "img_001_aug2.png" -> stem is "img_001"
            stem = fname[: fname.index("_aug")]
            sibling_map.setdefault(stem, []).append(fname)
    return sibling_map


# --------------------------------------------------------------------------
# Main split logic
# --------------------------------------------------------------------------

def split(
    proc_dir:    str = PROC_DIR,
    train_dir:   str = TRAIN_DIR,
    test_dir:    str = TEST_DIR,
    train_count: int = TRAIN_COUNT,
    test_count:  int = TEST_COUNT,
    n_batches:   int = N_BATCHES,
    seed:        int = SEED,
    dry_run:     bool = False,
) -> dict:
    """
    Split preprocessed images into train/test sets.

    Splitting logic:
      1. Collect only *original* images (files without '_aug' in name).
      2. Shuffle with fixed seed for reproducibility.
      3. First train_count → training set (originals + their augmented copies).
      4. Next test_count  → test set (originals only, split across n_batches).
         Augmented copies of test originals are NOT copied to test — prevents leakage.

    Args:
        proc_dir:    Source directory with class subfolders.
        train_dir:   Destination for training images.
        test_dir:    Destination for test batches.
        train_count: Original images per class for training.
        test_count:  Original images per class for testing.
        n_batches:   Number of test batches.
        seed:        Random seed for reproducibility.
        dry_run:     If True, log actions without copying files.

    Returns:
        Manifest dict with file lists per class per split.
    """
    if test_count % n_batches != 0:
        raise ValueError(f"test_count ({test_count}) must be divisible by n_batches ({n_batches})")

    batch_size = test_count // n_batches
    rng        = random.Random(seed)
    manifest   = {"seed": seed, "train_count": train_count, "test_count": test_count,
                  "n_batches": n_batches, "batch_size": batch_size, "classes": {}}

    print(f"Train : {train_count}/class | Test: {test_count}/class ({n_batches} batches × {batch_size})")
    print(f"Source: {proc_dir}")
    print(f"Dry run: {dry_run}")
    print("-" * 60)

    for cls in CLASSES:
        cls_dir = os.path.join(proc_dir, cls)

        if not os.path.exists(cls_dir):
            print(f"  SKIP {cls:<15s}: directory not found")
            manifest["classes"][cls] = {"status": "missing"}
            continue

        # Only original files (no _aug suffix)
        originals = sorted([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(".png") and "_aug" not in f
        ])

        if not originals:
            print(f"  SKIP {cls:<15s}: no original images found")
            manifest["classes"][cls] = {"status": "empty"}
            continue

        _validate_counts(originals, cls, train_count, test_count)

        # Reproducible shuffle
        rng.shuffle(originals)

        train_files = originals[:train_count]
        test_files  = originals[train_count: train_count + test_count]

        cls_manifest = {
            "status":       "ok",
            "train":        [],
            "test_batches": {f"batch{i+1}": [] for i in range(n_batches)},
        }

        # Precompute sibling map once per class (avoids 1,400 os.listdir calls)
        sibling_map = _build_sibling_map(cls_dir)

        # --- TRAIN: copy originals + augmented siblings ---
        dest_train_cls = os.path.join(train_dir, cls)
        if not dry_run:
            os.makedirs(dest_train_cls, exist_ok=True)

        for fname in train_files:
            src = os.path.join(cls_dir, fname)
            if not dry_run:
                shutil.copy2(src, os.path.join(dest_train_cls, fname))
            cls_manifest["train"].append(fname)

            # Copy augmented siblings into training set
            stem = Path(fname).stem
            for aug_fname in sibling_map.get(stem, []):
                aug_src = os.path.join(cls_dir, aug_fname)
                if not dry_run:
                    shutil.copy2(aug_src, os.path.join(dest_train_cls, aug_fname))
                cls_manifest["train"].append(aug_fname)

        # --- TEST: originals only, split across batches ---
        batches = [test_files[i * batch_size:(i + 1) * batch_size] for i in range(n_batches)]
        for b_idx, batch_files in enumerate(batches):
            batch_name    = f"batch{b_idx + 1}"
            dest_batch_cls = os.path.join(test_dir, batch_name, cls)
            if not dry_run:
                os.makedirs(dest_batch_cls, exist_ok=True)
            for fname in batch_files:
                src = os.path.join(cls_dir, fname)
                if not dry_run:
                    shutil.copy2(src, os.path.join(dest_batch_cls, fname))
                cls_manifest["test_batches"][batch_name].append(fname)

        train_total = len(cls_manifest["train"])
        test_total  = sum(len(v) for v in cls_manifest["test_batches"].values())
        manifest["classes"][cls] = cls_manifest
        print(f"  OK   {cls:<15s}: {train_count} train originals (+aug = {train_total}) | {test_total} test originals across {n_batches} batches")

    print("-" * 60)

    # --- Leakage check ---
    leaks = _check_leakage(manifest)
    if leaks:
        print(f"  WARNING: {len(leaks)} leakage(s) detected — same filename in train and test!")
        for l in leaks[:5]:
            print(f"    {l}")
    else:
        print("  Leakage check: PASS (no overlap between train and test originals)")

    manifest["leakage_detected"] = bool(leaks)

    if not dry_run:
        os.makedirs("outputs/metrics", exist_ok=True)
        report_path = "outputs/metrics/split_manifest.json"
        with open(report_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Manifest saved: {report_path}")

    return manifest


def _check_leakage(manifest: dict) -> list[str]:
    """Return list of filenames that appear in both train and any test batch."""
    leaks = []
    for cls, data in manifest["classes"].items():
        if data.get("status") != "ok":
            continue
        # Only compare original filenames (strip _aug variants)
        train_originals = set(
            f for f in data["train"] if "_aug" not in f
        )
        for batch_name, batch_files in data["test_batches"].items():
            overlap = train_originals & set(batch_files)
            for f in overlap:
                leaks.append(f"{cls}/{batch_name}/{f}")
    return leaks


# --------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.isdir(PROC_DIR):
        print(f"ERROR: '{PROC_DIR}' not found.")
        print("  Run from the project root: python scripts/split_dataset.py")
        print("  Ensure preprocess.py and augment.py have been run first.")
        sys.exit(1)

    print("=" * 60)
    print("  TASK 7: DATASET SPLIT")
    print("=" * 60)
    split()
