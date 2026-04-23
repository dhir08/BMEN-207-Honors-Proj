"""
Task 4: Explore and Audit the Dataset
- Class distribution (count per class)
- Image size distribution (min, max, mean)
- Format check (jpg/png/corrupt)
- Channel check (RGB vs grayscale)
- Saves summary plots to outputs/plots/

Run from project root:
  python scripts/audit_dataset.py
"""

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe in Colab and headless environments
import matplotlib.pyplot as plt
import pandas as pd

RAW_DIR  = "data/raw"
CLASSES  = ["glioma", "meningioma", "no_tumor", "pituitary"]
PLOT_DIR = "outputs/plots"
COLORS   = {
    "glioma":     "#E74C3C",
    "meningioma": "#3498DB",
    "no_tumor":   "#2ECC71",
    "pituitary":  "#F39C12",
}


def audit(raw_dir: str = RAW_DIR) -> pd.DataFrame | None:
    """
    Audit all images in raw_dir/<class>/ folders.

    Args:
        raw_dir: Path to raw data directory (default: 'data/raw')

    Returns:
        DataFrame of image metadata, or None if no images found.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    records = []
    corrupt = []

    for cls in CLASSES:
        cls_dir = os.path.join(raw_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"  WARNING: {cls_dir} not found. Skipping.")
            continue

        files = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not files:
            print(f"  WARNING: {cls_dir} exists but is empty.")
            continue

        for fname in files:
            path = os.path.join(cls_dir, fname)
            img  = cv2.imread(path)
            if img is None:
                corrupt.append(path)
                continue
            h, w = img.shape[:2]
            channels = 1 if len(img.shape) == 2 else img.shape[2]
            records.append({
                "class":    cls,
                "filename": fname,
                "height":   h,
                "width":    w,
                "channels": channels,
                "format":   os.path.splitext(fname)[1].lower(),
                "path":     path,
            })

    df = pd.DataFrame(records)

    # --- No-data guard ---
    if df.empty:
        print("\nNo images found. Please run scripts/download_dataset.py first.")
        if corrupt:
            print(f"Corrupt/unreadable files detected ({len(corrupt)}):")
            for p in corrupt[:5]:
                print(f"  {p}")
        return None

    # --- Console summary ---
    print("=" * 55)
    print("  DATASET AUDIT SUMMARY")
    print("=" * 55)
    print(f"  Total images   : {len(df)}")
    print(f"  Corrupt files  : {len(corrupt)}")
    print(f"\n  Class distribution:")
    for cls in CLASSES:
        count = len(df[df["class"] == cls])
        bar   = "█" * (count // 50)
        print(f"    {cls:<15s}: {count:>5}  {bar}")

    print(f"\n  Image sizes (H x W):")
    print(f"    Height — min: {df['height'].min()}, max: {df['height'].max()}, mean: {df['height'].mean():.1f}")
    print(f"    Width  — min: {df['width'].min()},  max: {df['width'].max()},  mean: {df['width'].mean():.1f}")
    print(f"\n  Channels : {dict(df['channels'].value_counts())}")
    print(f"  Formats  : {dict(df['format'].value_counts())}")
    print(f"\n  Uniform size: {'YES' if df['height'].nunique() == 1 and df['width'].nunique() == 1 else 'NO — resizing required in preprocessing'}")
    print("=" * 55)

    # --- Plot 1: Class distribution bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Brain Tumor MRI Dataset — Audit", fontsize=14, fontweight="bold")

    counts     = df.groupby("class").size().reindex(CLASSES).fillna(0).astype(int)
    bar_colors = [COLORS[c] for c in counts.index]
    bars = axes[0].bar(counts.index, counts.values, color=bar_colors, edgecolor="white", linewidth=0.8)
    axes[0].set_title("Class Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Tumor Class")
    axes[0].set_ylabel("Image Count")
    axes[0].set_ylim(0, counts.max() * 1.15)
    for bar, v in zip(bars, counts.values):
        if v > 0:
            axes[0].text(bar.get_x() + bar.get_width() / 2, v + counts.max() * 0.02,
                         str(v), ha="center", va="bottom", fontweight="bold", fontsize=10)

    # --- Plot 2: Image size scatter ---
    for cls in CLASSES:
        sub = df[df["class"] == cls]
        if sub.empty:
            continue
        axes[1].scatter(sub["width"], sub["height"],
                        alpha=0.25, s=6, label=cls, color=COLORS[cls])
    axes[1].set_title("Image Size Distribution (W × H)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Width (px)")
    axes[1].set_ylabel("Height (px)")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    out_png = os.path.join(PLOT_DIR, "dataset_audit.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot  -> {out_png}")

    # --- Save CSV ---
    out_csv = os.path.join(PLOT_DIR, "dataset_audit.csv")
    df.drop(columns=["path"]).to_csv(out_csv, index=False)
    print(f"  CSV   -> {out_csv}")

    if corrupt:
        print(f"\n  WARNING: {len(corrupt)} corrupt/unreadable files:")
        for p in corrupt[:10]:
            print(f"    {p}")

    return df


if __name__ == "__main__":
    if not os.path.isdir(RAW_DIR):
        print(f"ERROR: '{RAW_DIR}' directory not found.")
        print("  Run from the project root: python scripts/audit_dataset.py")
        sys.exit(1)
    audit()
