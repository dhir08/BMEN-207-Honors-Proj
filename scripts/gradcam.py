"""
Task 16: Grad-CAM Explainability Heatmaps
==========================================
Integrates pytorch-grad-cam to generate class activation maps that highlight
which regions of the MRI scan the ResNet-50 classifier relies on most.

Target layer: model.layer4 (final convolutional block — highest-level spatial features).
Method:       GradCAM (Gradient-weighted Class Activation Mapping, Selvaraju et al. 2017).

For each of the 4 classes this script:
  1. Selects N sample images from test batch 1
  2. Runs forward pass → computes Grad-CAM heatmap for the predicted (or true) class
  3. Overlays the heatmap on the original image
  4. Saves individual PNGs to outputs/gradcam/<class_name>/
  5. Saves a 4-panel summary grid to outputs/gradcam/gradcam_summary_<class>.png

Outputs:
  outputs/gradcam/<class>/         — individual overlay images
  outputs/gradcam/                 — class summary grids (used by visualize.py)

Install in Colab:
    !pip install grad-cam

Usage:
    python scripts/gradcam.py
    python scripts/gradcam.py --model models/best_model.pt --samples 4
"""

import os
import sys
import json
import argparse
from datetime import datetime

# ---------------------------------------------------------------------------
# Guard: PyTorch required
# ---------------------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Guard: pytorch-grad-cam required
# ---------------------------------------------------------------------------
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

# ---------------------------------------------------------------------------
# PIL always available (Pillow in requirements.txt)
# ---------------------------------------------------------------------------
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_config import (
    BEST_MODEL_PATH, TEST_DIR, CLASSES, NUM_CLASSES, IDX_TO_CLASS, CLASS_TO_IDX,
    IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, OUTPUTS_DIR, BATCH_SIZE,
)
from scripts.model import build_model, load_checkpoint

GRADCAM_DIR      = os.path.join(OUTPUTS_DIR, "gradcam")
GRADCAM_LOG_PATH = os.path.join(GRADCAM_DIR, "gradcam_log.json")

# Number of sample images to generate per class
DEFAULT_SAMPLES_PER_CLASS = 4


# ===========================================================================
# IMAGE UTILITIES
# ===========================================================================

def load_image_tensor(path: str) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Load a PNG MRI image and return:
      - tensor:   (1, 3, 224, 224) float32 — model input (normalized)
      - rgb_arr:  (224, 224, 3) float32 in [0, 1] — for overlay rendering
    """
    import torchvision.transforms as T
    import numpy as np

    img   = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(img).unsqueeze(0)          # (1, 3, 224, 224)

    # Un-normalised RGB for overlay (float32 [0,1])
    raw_t   = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])(img)
    rgb_arr = raw_t.permute(1, 2, 0).numpy().astype("float32")

    return tensor, rgb_arr


def denormalize(tensor: "torch.Tensor") -> "torch.Tensor":
    """Undo ImageNet normalization for visualization."""
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.squeeze(0).cpu() * std + mean).clamp(0, 1)


# ===========================================================================
# GRAD-CAM GENERATION
# ===========================================================================

def generate_gradcam_for_image(
    model:      "torch.nn.Module",
    img_tensor: "torch.Tensor",
    rgb_arr:    "object",            # np.ndarray (H, W, 3) float32 [0,1]
    target_class: int | None,
    device:     "torch.device",
) -> tuple["object", int, float]:
    """
    Run GradCAM on a single image.

    Args:
        model:         Trained ResNet-50 (eval mode).
        img_tensor:    (1, 3, 224, 224) normalized tensor.
        rgb_arr:       (224, 224, 3) float32 [0,1] for overlay rendering.
        target_class:  Class index to explain. If None, uses top predicted class.
        device:        CUDA or CPU.

    Returns:
        (overlay_rgb, predicted_class, confidence)
        overlay_rgb:     np.ndarray (224, 224, 3) uint8 — heatmap on image.
        predicted_class: int — argmax class index.
        confidence:      float — softmax probability of predicted class.
    """
    import numpy as np
    import torch

    img_tensor = img_tensor.to(device)

    # Predict
    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.softmax(logits, dim=1)
        pred   = logits.argmax(dim=1).item()
        conf   = probs[0, pred].item()

    tgt_cls = target_class if target_class is not None else pred

    # GradCAM targeting the last ResNet block
    target_layers = [model.layer4[-1]]
    targets       = [ClassifierOutputTarget(tgt_cls)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

    # grayscale_cam shape: (1, H, W) — take first (only) image
    overlay = show_cam_on_image(rgb_arr, grayscale_cam[0], use_rgb=True)

    return overlay, pred, conf


# ===========================================================================
# COLLECT SAMPLE IMAGE PATHS PER CLASS
# ===========================================================================

def collect_samples(batch_num: int = 1, n: int = DEFAULT_SAMPLES_PER_CLASS) -> dict[str, list[str]]:
    """
    Return up to n image paths per class from the specified test batch.

    Args:
        batch_num: Which test batch directory to sample from (1-indexed).
        n:         Number of samples per class.

    Returns:
        Dict mapping class_name → list of PNG paths.
    """
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
# MATPLOTLIB GRID (optional)
# ===========================================================================

def save_class_summary_grid(
    overlays:   list,            # list of np.ndarray (H, W, 3)
    class_name: str,
    output_path: str,
) -> bool:
    """
    Save a 1×N grid of Grad-CAM overlays for a single class.
    Returns True if saved, False if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    n   = len(overlays)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, overlay in zip(axes, overlays):
        ax.imshow(overlay)
        ax.axis("off")

    fig.suptitle(f"Grad-CAM — {class_name} (ResNet-50, layer4)", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    return True


# ===========================================================================
# SAVE OVERLAY AS PNG
# ===========================================================================

def save_overlay_png(overlay, path: str) -> None:
    """Save np.ndarray uint8 overlay as PNG."""
    import numpy as np
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(overlay.astype("uint8")).save(path)


# ===========================================================================
# MAIN
# ===========================================================================

def main(args: argparse.Namespace) -> None:
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not found. Run in Google Colab.")
        sys.exit(1)

    if not GRADCAM_AVAILABLE:
        print("ERROR: pytorch-grad-cam not found.")
        print("  Install with:  pip install grad-cam")
        sys.exit(1)

    import numpy as np

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model or BEST_MODEL_PATH

    if not os.path.isfile(model_path):
        print(f"ERROR: Model checkpoint not found: {model_path}")
        print("  Run train.py first to produce best_model.pt")
        sys.exit(1)

    print("=" * 62)
    print("  TASK 16: GRAD-CAM EXPLAINABILITY HEATMAPS")
    print("=" * 62)
    print(f"  Model  : {model_path}")
    print(f"  Device : {device}")
    print(f"  Samples/class: {args.samples}\n")

    model = build_model(pretrained=False)
    load_checkpoint(model, optimizer=None, path=model_path)
    model = model.to(device)
    model.eval()

    samples    = collect_samples(batch_num=1, n=args.samples)
    log_entries: list[dict] = []

    for cls in CLASSES:
        cls_idx   = CLASS_TO_IDX[cls]
        cls_paths = samples.get(cls, [])

        if not cls_paths:
            print(f"  WARNING: No images found for class '{cls}' in batch 1 — skipping.")
            continue

        print(f"  Generating Grad-CAM for class: {cls} ({len(cls_paths)} images)")
        cls_dir    = os.path.join(GRADCAM_DIR, cls)
        overlays   = []

        for i, img_path in enumerate(cls_paths):
            img_tensor, rgb_arr = load_image_tensor(img_path)
            overlay, pred_cls, conf = generate_gradcam_for_image(
                model, img_tensor, rgb_arr,
                target_class=cls_idx,    # always explain the true class
                device=device,
            )
            overlays.append(overlay)

            # Save individual overlay
            out_name = f"{cls}_{i+1:02d}_pred-{IDX_TO_CLASS[pred_cls]}_conf{conf:.2f}.png"
            out_path = os.path.join(cls_dir, out_name)
            save_overlay_png(overlay, out_path)

            correct_str = "✓" if pred_cls == cls_idx else "✗"
            print(f"    [{i+1}/{len(cls_paths)}] pred={IDX_TO_CLASS[pred_cls]} "
                  f"({conf:.2%}) {correct_str}  → {out_path}")

            log_entries.append({
                "class":          cls,
                "true_label":     cls_idx,
                "pred_label":     pred_cls,
                "confidence":     round(conf, 6),
                "correct":        pred_cls == cls_idx,
                "source_image":   img_path,
                "overlay_saved":  out_path,
            })

        # Save per-class summary grid
        grid_path = os.path.join(GRADCAM_DIR, f"gradcam_summary_{cls}.png")
        saved = save_class_summary_grid(overlays, cls, grid_path)
        if saved:
            print(f"    Summary grid → {grid_path}")

    # --- Save log ---
    os.makedirs(GRADCAM_DIR, exist_ok=True)
    log_output = {
        "task":        "Task 16: Grad-CAM Heatmaps",
        "generated_at": datetime.now().isoformat(),
        "model_path":  model_path,
        "target_layer": "model.layer4 (final ResNet block)",
        "method":      "GradCAM (Selvaraju et al. 2017)",
        "entries":     log_entries,
    }
    with open(GRADCAM_LOG_PATH, "w") as f:
        json.dump(log_output, f, indent=2)
    print(f"\n  Grad-CAM log  → {GRADCAM_LOG_PATH}")
    print(f"  Overlay images in: {GRADCAM_DIR}")
    print("\nDone.")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM explainability heatmaps for brain tumor classes"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="PATH",
        help=f"Path to model checkpoint (default: {BEST_MODEL_PATH})",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        metavar="N",
        help=f"Number of sample images per class (default: {DEFAULT_SAMPLES_PER_CLASS})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
