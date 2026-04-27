"""
Task 17: External Validation — The Cancer Imaging Archive (TCIA)
================================================================
Evaluates the trained model on external MRI samples from TCIA to test
generalizability beyond the Kaggle training distribution.

This script handles the full external-validation pipeline:
  1. Accepts a folder of TCIA-sourced MRI images (DICOM or PNG/JPG)
  2. Applies the SAME preprocessing pipeline as preprocess.py
     (resize → 224×224 RGB, normalize with ImageNet stats)
  3. Runs inference with the trained best_model.pt
  4. Reports per-class accuracy and compares against in-distribution results
  5. Saves results to outputs/metrics/external_validation.json

Expected TCIA folder structure (either flat or class-organised):
    external_data/
        glioma/       *.dcm or *.png
        meningioma/   *.dcm or *.png
        no_tumor/     *.dcm or *.png    (may not exist in TCIA subset)
        pituitary/    *.dcm or *.png

  — OR — flat folder with labelled filenames:
    external_data/
        glioma_001.png
        meningioma_007.dcm
        ...

Notes on TCIA access:
    Brain MRI data for glioma / meningioma classification is available from
    TCIA collections such as:
      • TCGA-GBM  (glioblastoma MRI)
      • TCGA-LGG  (lower-grade glioma)
      • RIDER Brain MRI
    Download via the TCIA Data Retrieval Service (NBIA Data Retriever) or
    directly at https://www.cancerimagingarchive.net/

Install DICOM support (optional — only needed for .dcm input):
    pip install pydicom

Usage:
    python scripts/validate_external.py --data path/to/tcia_images/
    python scripts/validate_external.py --data path/to/tcia/ --model models/best_model.pt
"""

import os
import sys
import json
import argparse
from datetime import datetime

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pydicom
    import numpy as np
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_config import (
    BEST_MODEL_PATH, CLASSES, NUM_CLASSES, IDX_TO_CLASS, CLASS_TO_IDX,
    IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, METRICS_DIR,
)
from scripts.model import build_model, load_checkpoint

EXTERNAL_RESULTS_PATH = os.path.join(METRICS_DIR, "external_validation.json")
IN_DIST_RESULTS_PATH  = os.path.join(METRICS_DIR, "evaluation_results.json")


# ===========================================================================
# IMAGE LOADING — supports .png, .jpg, .jpeg, .bmp, .tif, .dcm
# ===========================================================================

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def load_image_as_pil(path: str) -> "Image.Image | None":
    """
    Load any supported image format and return a PIL Image in RGB mode.
    Returns None if loading fails.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".dcm":
        if not DICOM_AVAILABLE:
            print(f"    WARNING: pydicom not installed — cannot load {path}")
            return None
        try:
            ds    = pydicom.dcmread(path)
            arr   = ds.pixel_array.astype("float32")
            # Normalize to [0, 255] uint8
            arr   = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            img   = Image.fromarray(arr.astype("uint8"))
            # Ensure RGB (DICOM may be grayscale)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:
            print(f"    WARNING: DICOM load failed for {path}: {e}")
            return None

    if ext in SUPPORTED_EXTENSIONS:
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            print(f"    WARNING: PIL load failed for {path}: {e}")
            return None

    return None  # unsupported extension


def preprocess_pil(img: "Image.Image") -> "torch.Tensor":
    """Apply the same preprocessing as the training pipeline and return (1, 3, 224, 224)."""
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(img).unsqueeze(0)


# ===========================================================================
# DISCOVER IMAGES
# ===========================================================================

def discover_images(data_dir: str) -> list[tuple[str, int | None]]:
    """
    Walk data_dir and return (image_path, label_or_None) pairs.

    Label is inferred from the parent folder name if it matches a known class.
    If the image is in a flat folder or unlabelled, label = None.
    """
    found: list[tuple[str, int | None]] = []

    for root, dirs, files in os.walk(data_dir):
        folder_name = os.path.basename(root).lower()
        label = CLASS_TO_IDX.get(folder_name, None)

        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS | {".dcm"}:
                full_path = os.path.join(root, fname)

                # Try to infer label from filename prefix if not from folder
                inferred_label = label
                if inferred_label is None:
                    for cls_name, cls_idx in CLASS_TO_IDX.items():
                        if fname.lower().startswith(cls_name):
                            inferred_label = cls_idx
                            break

                found.append((full_path, inferred_label))

    return found


# ===========================================================================
# INFERENCE
# ===========================================================================

def run_inference(
    model:    "nn.Module",
    device:   "torch.device",
    img_list: list[tuple[str, "int | None"]],
) -> list[dict]:
    """
    Run inference on all discovered images.

    Returns a list of result dicts with keys:
        path, true_label, pred_label, confidence, correct (or None if unlabelled)
    """
    results = []
    model.eval()

    for idx, (path, true_label) in enumerate(img_list):
        pil_img = load_image_as_pil(path)
        if pil_img is None:
            continue

        tensor = preprocess_pil(pil_img).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)
            pred   = logits.argmax(dim=1).item()
            conf   = probs[0, pred].item()

        correct = (pred == true_label) if true_label is not None else None

        results.append({
            "path":        path,
            "true_label":  true_label,
            "true_class":  IDX_TO_CLASS.get(true_label, "unknown") if true_label is not None else "unknown",
            "pred_label":  pred,
            "pred_class":  IDX_TO_CLASS[pred],
            "confidence":  round(conf, 6),
            "correct":     correct,
        })

        if (idx + 1) % 10 == 0:
            print(f"    Processed {idx+1}/{len(img_list)} images...", end="\r")

    print()
    return results


# ===========================================================================
# AGGREGATE METRICS
# ===========================================================================

def compute_accuracy_from_results(results: list[dict]) -> dict:
    """Compute overall and per-class accuracy from inference results (labelled images only)."""
    labelled   = [r for r in results if r["correct"] is not None]
    if not labelled:
        return {"overall": None, "per_class": {}}

    correct = sum(1 for r in labelled if r["correct"])
    overall = correct / len(labelled)

    per_class: dict[str, dict] = {}
    for cls in CLASSES:
        cls_results = [r for r in labelled if r["true_class"] == cls]
        if not cls_results:
            continue
        cls_correct = sum(1 for r in cls_results if r["correct"])
        per_class[cls] = {
            "accuracy": round(cls_correct / len(cls_results), 6),
            "n_correct": cls_correct,
            "n_total":   len(cls_results),
        }

    return {"overall": round(overall, 6), "per_class": per_class}


# ===========================================================================
# MAIN
# ===========================================================================

def main(args: argparse.Namespace) -> None:
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not found. Run in Google Colab.")
        sys.exit(1)

    if not os.path.isdir(args.data):
        print(f"ERROR: Data directory not found: {args.data}")
        print("  Download TCIA images (e.g. TCGA-GBM or TCGA-LGG) and provide the path.")
        sys.exit(1)

    model_path = args.model or BEST_MODEL_PATH
    if not os.path.isfile(model_path):
        print(f"ERROR: Model checkpoint not found: {model_path}")
        print("  Run train.py first to produce best_model.pt")
        sys.exit(1)

    print("=" * 62)
    print("  TASK 17: EXTERNAL VALIDATION — CANCER IMAGING ARCHIVE")
    print("=" * 62)
    print(f"  Data   : {args.data}")
    print(f"  Model  : {model_path}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    model = build_model(pretrained=False)
    load_checkpoint(model, optimizer=None, path=model_path)
    model = model.to(device)
    model.eval()

    # Discover images
    print(f"\n  Scanning {args.data} for images...")
    img_list = discover_images(args.data)
    labelled_count   = sum(1 for _, lbl in img_list if lbl is not None)
    unlabelled_count = len(img_list) - labelled_count
    print(f"  Found {len(img_list)} images  ({labelled_count} labelled, {unlabelled_count} unlabelled)")

    if not img_list:
        print("  No images found. Check that the data directory contains supported files.")
        sys.exit(1)

    # Run inference
    print("\n  Running inference...")
    results = run_inference(model, device, img_list)
    print(f"  Processed {len(results)} images successfully.")

    # Compute metrics
    ext_metrics = compute_accuracy_from_results(results)

    # Load in-distribution results for comparison
    in_dist_overall = None
    if os.path.isfile(IN_DIST_RESULTS_PATH):
        with open(IN_DIST_RESULTS_PATH) as f:
            in_dist_data = json.load(f)
        in_dist_overall = in_dist_data.get("overall", {}).get("accuracy")

    # Print summary
    print("\n  External Validation Results")
    print("  " + "-" * 44)
    if ext_metrics["overall"] is not None:
        print(f"  External accuracy : {ext_metrics['overall']:.4f}")
        if in_dist_overall is not None:
            drop = in_dist_overall - ext_metrics["overall"]
            print(f"  In-dist accuracy  : {in_dist_overall:.4f}   (Δ = {drop:+.4f})")
        print()
        for cls, m in ext_metrics["per_class"].items():
            print(f"  {cls:<15s}: {m['accuracy']:.4f}  ({m['n_correct']}/{m['n_total']})")
    else:
        print("  No labelled images found — cannot compute accuracy.")
        print("  Prediction distribution:")
        from collections import Counter
        pred_dist = Counter(r["pred_class"] for r in results)
        for cls, cnt in sorted(pred_dist.items()):
            print(f"    {cls:<15s}: {cnt} predictions")

    # Save results
    os.makedirs(METRICS_DIR, exist_ok=True)
    output = {
        "task":              "Task 17: External Validation (TCIA)",
        "evaluated_at":      datetime.now().isoformat(),
        "data_dir":          args.data,
        "model_path":        model_path,
        "total_images":      len(results),
        "labelled_images":   labelled_count,
        "external_metrics":  ext_metrics,
        "in_dist_accuracy":  in_dist_overall,
        "generalization_gap": (
            round(in_dist_overall - ext_metrics["overall"], 6)
            if in_dist_overall and ext_metrics["overall"] is not None
            else None
        ),
        "per_image_results": results,
    }
    with open(EXTERNAL_RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {EXTERNAL_RESULTS_PATH}")
    print("\nDone.")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate trained model on external TCIA MRI images"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        metavar="DIR",
        help="Path to directory containing TCIA MRI images (class subfolders or flat)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="PATH",
        help=f"Path to model checkpoint (default: {BEST_MODEL_PATH})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
