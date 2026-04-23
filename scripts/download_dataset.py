"""
Task 3: Download Kaggle MRI Brain Tumor Dataset
Dataset: masoudnickparvar/brain-tumor-mri-dataset
Classes: glioma, meningioma, no_tumor, pituitary
Total: ~7,000+ images

Run this script from the project root in Google Colab or locally.

SETUP (one-time):
  1. Go to https://www.kaggle.com/settings -> API -> Create New Token
  2. Download kaggle.json
  3. Place it at ~/.kaggle/kaggle.json  (or upload to Colab)
  4. Run: chmod 600 ~/.kaggle/kaggle.json

COLAB USAGE:
  from google.colab import files
  files.upload()  # upload kaggle.json
  !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
  !python scripts/download_dataset.py
"""

import os
import shutil
import sys

DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
RAW_DIR = "data/raw"
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Alias map: handles Kaggle folder naming variations
# e.g. "notumor" -> "no_tumor", "no tumor" -> "no_tumor"
CLASS_ALIASES = {
    "glioma":      ["glioma"],
    "meningioma":  ["meningioma"],
    "no_tumor":    ["no_tumor", "notumor", "no tumor", "no-tumor", "healthy", "normal"],
    "pituitary":   ["pituitary"],
}


def _check_kaggle():
    """Verify kaggle CLI is available and credentials exist."""
    if shutil.which("kaggle") is None:
        print("ERROR: kaggle CLI not found. Run: pip install kaggle")
        sys.exit(1)
    cred_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(cred_path):
        print("ERROR: Kaggle credentials not found at ~/.kaggle/kaggle.json")
        print("  1. Go to https://www.kaggle.com/settings -> API -> Create New Token")
        print("  2. Move the file: mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json")
        print("  3. chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)
    print("Kaggle credentials: OK")


def _match_class(folder_name: str) -> str | None:
    """Map a folder name to one of the 4 canonical class names."""
    folder_lower = folder_name.lower().strip()
    if not folder_lower:
        return None
    for cls, aliases in CLASS_ALIASES.items():
        for alias in aliases:
            if alias in folder_lower:   # alias must be contained IN folder name, not vice versa
                return cls
    return None


def _organize():
    """
    Flatten all images from nested Kaggle structure into data/raw/<class>/.
    Uses a deduplication counter to avoid filename collisions across sources.

    NOTE: This function is designed to run ONCE immediately after a fresh Kaggle
    download.  Calling it a second time on already-organized data will produce
    duplicate copies of every image (because copy2 preserves sources).  The
    download() function always re-extracts fresh, so this is safe in normal use.
    """
    counters = {cls: 0 for cls in CLASSES}

    for root, dirs, files in os.walk(RAW_DIR):
        # Skip already-organized class dirs to avoid infinite loop
        rel = os.path.relpath(root, RAW_DIR)
        depth = len(rel.split(os.sep))
        if depth == 1 and rel in CLASSES:
            continue

        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            src = os.path.join(root, fname)
            parent = os.path.basename(root)
            matched = _match_class(parent)

            if matched is None:
                continue

            dest_dir = os.path.join(RAW_DIR, matched)
            os.makedirs(dest_dir, exist_ok=True)

            # Deduplicate: prefix with counter if filename already exists
            base, ext = os.path.splitext(fname)
            dest = os.path.join(dest_dir, fname)
            if os.path.exists(dest) and dest != src:
                counters[matched] += 1
                dest = os.path.join(dest_dir, f"{base}_{counters[matched]}{ext}")

            if src != dest:
                shutil.copy2(src, dest)

    # Summary
    print("\nDataset summary after organization:")
    total = 0
    for cls in CLASSES:
        path = os.path.join(RAW_DIR, cls)
        count = 0
        if os.path.exists(path):
            count = len([
                f for f in os.listdir(path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
        total += count
        status = "OK" if count > 0 else "EMPTY — check folder names"
        print(f"  {cls:<15s}: {count:>5} images  [{status}]")
    print(f"  {'TOTAL':<15s}: {total:>5} images")

    if total == 0:
        print("\nWARNING: No images were organized. The Kaggle dataset folder structure")
        print("  may differ from expected. Check data/raw/ contents manually.")


def download():
    _check_kaggle()
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Downloading: {DATASET}")
    ret = os.system(f"kaggle datasets download -d {DATASET} -p {RAW_DIR} --unzip")
    if ret != 0:
        print("ERROR: kaggle download failed. Check your credentials and internet connection.")
        sys.exit(1)
    print("Download complete. Organizing files...")
    _organize()


if __name__ == "__main__":
    # Guard: must be run from project root
    if not os.path.isdir("data/raw"):
        print("ERROR: Run this script from the project root directory.")
        print("  cd 'brain tumor analysis' && python scripts/download_dataset.py")
        sys.exit(1)
    download()
