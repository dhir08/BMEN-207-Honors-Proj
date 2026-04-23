"""
Task 18: Visualize Results — Sample Predictions with Grad-CAM Overlays
=======================================================================
Produces publication-quality visualization grids showing:
  - Original MRI image
  - Grad-CAM heatmap overlay
  - Model prediction + confidence
  - True class label

For each of the 4 brain tumor classes, generates:
  1. A 2×N figure (original | Grad-CAM) for N sample images
  2. A combined 4-class summary figure (all classes on one page)
  3. A training curve plot (loss + accuracy over epochs, both phases)

Outputs:
  outputs/plots/class_<name>_predictions.png    — per-class prediction grid
  outputs/plots/all_classes_summary.png         — combined 4-class overview
  outputs/plots/training_curves.png             — loss/accuracy curves

Usage (run after gradcam.py and evaluate.py):
    python scripts/visualize.py
    python scripts/visualize.py --model models/best_model.pt --samples 4
"""

import os
import sys
import json
import argparse
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_config import (
    BEST_MODEL_PATH, TEST_DIR, CLASSES, IDX_TO_CLASS, CLASS_TO_IDX,
    IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, PLOTS_DIR, METRICS_DIR,
)
from scripts.model import build_model, load_checkpoint

TRAINING_LOG_PATH = os.path.join(METRICS_DIR, "training_log.json")
SUMMARY_PATH      = os.path.join(PLOTS_DIR, "all_classes_summary.png")
TRAINING_CURVES   = os.path.join(PLOTS_DIR, "training_curves.png")

DEFAULT_SAMPLES = 4


# ===========================================================================
# IMAGE + GRAD-CAM UTILITIES
# ===========================================================================

def load_for_visualization(path: str) -> tuple:
    """
    Load image for both model input and visual overlay.

    Returns:
        tensor:  (1, 3, 224, 224) normalized model input
        rgb_arr: (224, 224, 3) float32 [0,1] for overlay rendering
        pil_img: PIL Image (224×224 RGB)
    """
    import torchvision.transforms as T
    import numpy as np

    img = Image.open(path).convert("RGB")

    transform_model = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    transform_vis = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])

    tensor  = transform_model(img).unsqueeze(0)
    raw     = transform_vis(img)
    rgb_arr = raw.permute(1, 2, 0).numpy().astype("float32")

    return tensor, rgb_arr, img.resize((IMG_SIZE, IMG_SIZE))


def predict_with_gradcam(
    model:       "torch.nn.Module",
    tensor:      "torch.Tensor",
    rgb_arr,
    target_cls:  int,
    device:      "torch.device",
) -> tuple:
    """
    Returns (overlay, pred_label, confidence).
    Falls back to raw RGB if Grad-CAM unavailable.
    """
    import numpy as np

    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
        pred   = logits.argmax(dim=1).item()
        conf   = probs[0, pred].item()

    if not GRADCAM_AVAILABLE:
        overlay = (rgb_arr * 255).astype("uint8")
        return overlay, pred, conf

    target_layers = [model.layer4[-1]]
    targets       = [ClassifierOutputTarget(target_cls)]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=targets)

    overlay = show_cam_on_image(rgb_arr, grayscale_cam[0], use_rgb=True)
    return overlay, pred, conf


# ===========================================================================
# PER-CLASS PREDICTION GRID
# ===========================================================================

def plot_class_grid(
    model:      "torch.nn.Module",
    cls:        str,
    img_paths:  list[str],
    device:     "torch.device",
    output_path: str,
) -> None:
    """
    Create a 2-row × N-col grid for a single class:
      Row 0: original MRI images
      Row 1: Grad-CAM overlays with prediction labels
    """
    n   = len(img_paths)
    fig = plt.figure(figsize=(4 * n, 9))
    fig.suptitle(
        f"Class: {cls.upper()}  —  ResNet-50 Brain Tumor Classifier",
        fontsize=14, fontweight="bold", y=1.01,
    )

    gs = gridspec.GridSpec(2, n, figure=fig, hspace=0.35, wspace=0.1)

    for col, img_path in enumerate(img_paths):
        tensor, rgb_arr, pil_img = load_for_visualization(img_path)
        overlay, pred, conf      = predict_with_gradcam(
            model, tensor, rgb_arr, target_cls=CLASS_TO_IDX[cls], device=device
        )

        pred_cls    = IDX_TO_CLASS[pred]
        is_correct  = pred_cls == cls
        label_color = "green" if is_correct else "red"
        label_str   = f"Pred: {pred_cls}\n({conf:.1%})"
        if not is_correct:
            label_str += f"\nTrue: {cls}"

        # Row 0 — original
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(pil_img)
        ax0.set_title(f"Original #{col+1}", fontsize=9)
        ax0.axis("off")

        # Row 1 — Grad-CAM
        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(overlay)
        ax1.set_title(label_str, fontsize=9, color=label_color)
        ax1.axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()


# ===========================================================================
# COMBINED 4-CLASS SUMMARY FIGURE
# ===========================================================================

def plot_all_classes_summary(
    model:      "torch.nn.Module",
    samples:    dict[str, list[str]],
    device:     "torch.device",
    output_path: str,
    n_per_class: int = 2,
) -> None:
    """
    4-row (one per class) × 2*N-col figure alternating original | Grad-CAM.
    """
    n   = n_per_class
    fig = plt.figure(figsize=(4 * n * 2, 5 * len(CLASSES)))
    fig.suptitle(
        "Brain Tumor MRI — ResNet-50 Predictions with Grad-CAM Explanations\n"
        "(green border = correct, red = misclassified)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    gs = gridspec.GridSpec(len(CLASSES), n * 2, figure=fig, hspace=0.5, wspace=0.05)

    for row_idx, cls in enumerate(CLASSES):
        cls_paths = samples.get(cls, [])[:n]
        if not cls_paths:
            continue

        for col_pair, img_path in enumerate(cls_paths):
            tensor, rgb_arr, pil_img = load_for_visualization(img_path)
            overlay, pred, conf      = predict_with_gradcam(
                model, tensor, rgb_arr, target_cls=CLASS_TO_IDX[cls], device=device
            )

            pred_cls    = IDX_TO_CLASS[pred]
            is_correct  = pred_cls == cls
            border_color = "green" if is_correct else "red"

            # Original
            ax_orig = fig.add_subplot(gs[row_idx, col_pair * 2])
            ax_orig.imshow(pil_img)
            ax_orig.set_title(
                f"{cls}\n(original)",
                fontsize=8, color="navy",
            )
            for spine in ax_orig.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)
            ax_orig.axis("off")

            # Grad-CAM
            ax_cam = fig.add_subplot(gs[row_idx, col_pair * 2 + 1])
            ax_cam.imshow(overlay)
            ax_cam.set_title(
                f"Pred: {pred_cls} ({conf:.0%})",
                fontsize=8, color="green" if is_correct else "red",
            )
            for spine in ax_cam.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)
            ax_cam.axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  All-class summary → {output_path}")


# ===========================================================================
# TRAINING CURVES
# ===========================================================================

def plot_training_curves(log_path: str, output_path: str) -> bool:
    """
    Read training_log.json and plot loss + accuracy curves for both phases.
    Returns True if plotted, False if log not found.
    """
    if not os.path.isfile(log_path):
        return False

    with open(log_path) as f:
        log = json.load(f)

    phase1 = [e for e in log if e["phase"] == 1]
    phase2 = [e for e in log if e["phase"] == 2]

    def extract(entries, key):
        return list(range(1, len(entries) + 1)), [e[key] for e in entries]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training Curves — Brain Tumor ResNet-50 (Two-Phase Transfer Learning)",
                 fontsize=12, fontweight="bold")

    # Loss
    ax = axes[0]
    if phase1:
        x, y = extract(phase1, "train_loss")
        ax.plot(x, y, "b--", label="P1 Train Loss", linewidth=1.5)
        x, y = extract(phase1, "val_loss")
        ax.plot(x, y, "b-",  label="P1 Val Loss",   linewidth=1.5)
    if phase2:
        offset = len(phase1)
        x2 = [offset + i for i in range(1, len(phase2) + 1)]
        ax.plot(x2, [e["train_loss"] for e in phase2], "r--", label="P2 Train Loss", linewidth=1.5)
        ax.plot(x2, [e["val_loss"]   for e in phase2], "r-",  label="P2 Val Loss",   linewidth=1.5)
        if phase1:
            ax.axvline(x=len(phase1) + 0.5, color="gray", linestyle=":", linewidth=1.2,
                       label="Phase 1→2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    if phase1:
        x, y = extract(phase1, "train_acc")
        ax.plot(x, y, "b--", label="P1 Train Acc", linewidth=1.5)
        x, y = extract(phase1, "val_acc")
        ax.plot(x, y, "b-",  label="P1 Val Acc",   linewidth=1.5)
    if phase2:
        offset = len(phase1)
        x2 = [offset + i for i in range(1, len(phase2) + 1)]
        ax.plot(x2, [e["train_acc"] for e in phase2], "r--", label="P2 Train Acc", linewidth=1.5)
        ax.plot(x2, [e["val_acc"]   for e in phase2], "r-",  label="P2 Val Acc",   linewidth=1.5)
        if phase1:
            ax.axvline(x=len(phase1) + 0.5, color="gray", linestyle=":", linewidth=1.2,
                       label="Phase 1→2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    return True


# ===========================================================================
# COLLECT SAMPLES
# ===========================================================================

def collect_samples(batch_num: int = 1, n: int = DEFAULT_SAMPLES) -> dict[str, list[str]]:
    samples: dict[str, list[str]] = {}
    batch_dir = os.path.join(TEST_DIR, f"batch{batch_num}")
    for cls in CLASSES:
        cls_dir = os.path.join(batch_dir, cls)
        if not os.path.isdir(cls_dir):
            samples[cls] = []
            continue
        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(".png"))[:n]
        samples[cls] = [os.path.join(cls_dir, f) for f in files]
    return samples


# ===========================================================================
# MAIN
# ===========================================================================

def main(args: argparse.Namespace) -> None:
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not found. Run in Google Colab.")
        sys.exit(1)
    if not MATPLOTLIB_AVAILABLE:
        print("ERROR: matplotlib not found. Install with: pip install matplotlib")
        sys.exit(1)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model or BEST_MODEL_PATH

    if not os.path.isfile(model_path):
        print(f"ERROR: Model checkpoint not found: {model_path}")
        sys.exit(1)

    print("=" * 62)
    print("  TASK 18: VISUALIZE PREDICTIONS WITH GRAD-CAM")
    print("=" * 62)
    print(f"  Model   : {model_path}")
    print(f"  Device  : {device}")
    print(f"  Samples : {args.samples}/class")

    if not GRADCAM_AVAILABLE:
        print("  WARNING: pytorch-grad-cam not installed — overlays will show raw images.")
        print("           Install with: pip install grad-cam\n")

    model = build_model(pretrained=False)
    load_checkpoint(model, optimizer=None, path=model_path)
    model = model.to(device)
    model.eval()

    samples = collect_samples(batch_num=1, n=args.samples)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Per-class grids
    print("\n  Generating per-class prediction grids...")
    for cls in CLASSES:
        cls_paths = samples.get(cls, [])
        if not cls_paths:
            print(f"  WARNING: No images found for class '{cls}'")
            continue
        out = os.path.join(PLOTS_DIR, f"class_{cls}_predictions.png")
        plot_class_grid(model, cls, cls_paths, device, out)
        print(f"    {cls:<15s} → {out}")

    # Combined summary
    print("\n  Generating all-class summary figure...")
    plot_all_classes_summary(model, samples, device, SUMMARY_PATH, n_per_class=2)

    # Training curves
    print("\n  Generating training curves...")
    saved = plot_training_curves(TRAINING_LOG_PATH, TRAINING_CURVES)
    if saved:
        print(f"  Training curves   → {TRAINING_CURVES}")
    else:
        print(f"  (training_log.json not found — run train.py first for this plot)")

    print("\nDone.")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize model predictions and Grad-CAM overlays"
    )
    parser.add_argument("--model", type=str, default=None, metavar="PATH",
                        help=f"Model checkpoint path (default: {BEST_MODEL_PATH})")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, metavar="N",
                        help=f"Images per class (default: {DEFAULT_SAMPLES})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
