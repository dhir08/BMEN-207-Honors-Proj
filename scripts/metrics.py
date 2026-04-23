"""
Task 15: Classification Metrics — Confusion Matrix, Precision, Recall, F1
==========================================================================
Reads the raw prediction data written by evaluate.py (Task 14) and computes:
  - Per-class precision, recall, F1-score, and support
  - Macro and weighted averages
  - 4×4 confusion matrix
  - Saves a PNG confusion-matrix heatmap to outputs/plots/
  - Saves a JSON metrics report to outputs/metrics/

Outputs:
  outputs/metrics/classification_report.json    — full metrics dict
  outputs/plots/confusion_matrix.png            — heatmap figure

Usage (run AFTER evaluate.py):
    python scripts/metrics.py
    python scripts/metrics.py --predictions outputs/metrics/all_predictions.json
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_config import CLASSES, NUM_CLASSES, METRICS_DIR, PLOTS_DIR

ALL_PREDICTIONS_PATH   = os.path.join(METRICS_DIR, "all_predictions.json")
REPORT_OUTPUT_PATH     = os.path.join(METRICS_DIR, "classification_report.json")
CONFUSION_MATRIX_PATH  = os.path.join(PLOTS_DIR, "confusion_matrix.png")


# ===========================================================================
# PURE-PYTHON METRIC HELPERS (no sklearn required for core math)
# ===========================================================================

def build_confusion_matrix(
    true_labels: list[int],
    pred_labels: list[int],
    num_classes: int,
) -> list[list[int]]:
    """
    Build an (N×N) confusion matrix from flat prediction lists.
    cm[true][pred] = count.
    """
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(true_labels, pred_labels):
        cm[t][p] += 1
    return cm


def per_class_metrics(cm: list[list[int]], num_classes: int) -> dict:
    """
    Compute precision, recall, F1-score, and support for each class.

    Definitions:
        precision_i = cm[i][i] / sum(cm[j][i] for j in range(N))  # TP / (TP+FP)
        recall_i    = cm[i][i] / sum(cm[i][j] for j in range(N))  # TP / (TP+FN)
        f1_i        = 2 * P * R / (P + R)
    """
    results = {}
    for i in range(num_classes):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(num_classes)) - tp   # predicted i, not actually i
        fn = sum(cm[i][j] for j in range(num_classes)) - tp   # actually i, not predicted i
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        results[CLASSES[i]] = {
            "precision": round(precision, 6),
            "recall":    round(recall,    6),
            "f1_score":  round(f1,        6),
            "support":   support,
        }
    return results


def macro_avg(per_class: dict) -> dict:
    """Unweighted average across all classes."""
    keys = ["precision", "recall", "f1_score"]
    return {k: round(sum(v[k] for v in per_class.values()) / len(per_class), 6) for k in keys}


def weighted_avg(per_class: dict) -> dict:
    """Support-weighted average across all classes."""
    total_support = sum(v["support"] for v in per_class.values())
    keys          = ["precision", "recall", "f1_score"]
    wavg = {}
    for k in keys:
        wavg[k] = round(
            sum(v[k] * v["support"] for v in per_class.values()) / total_support
            if total_support > 0 else 0.0,
            6,
        )
    return wavg


def overall_accuracy(true_labels: list[int], pred_labels: list[int]) -> float:
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    return round(correct / len(true_labels), 6) if true_labels else 0.0


# ===========================================================================
# MATPLOTLIB CONFUSION MATRIX PLOT (optional — skipped if matplotlib absent)
# ===========================================================================

def plot_confusion_matrix(cm: list[list[int]], output_path: str) -> bool:
    """
    Render and save a colour-coded confusion matrix heatmap.

    Returns True if saved successfully, False if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend (Colab/server safe)
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
    except ImportError:
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cm_arr = np.array(cm)
    n      = cm_arr.shape[0]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Normalize for colour intensity, keep raw counts as labels
    cm_norm = cm_arr.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Recall (row-normalised)")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(CLASSES, fontsize=10)

    # Annotate each cell with raw count
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Brain Tumor MRI Classifier\n"
                 "(ResNet-50, 4-class, row-normalised colour)", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


# ===========================================================================
# MAIN
# ===========================================================================

def main(args: argparse.Namespace) -> None:
    predictions_path = args.predictions or ALL_PREDICTIONS_PATH

    if not os.path.isfile(predictions_path):
        print(f"ERROR: Predictions file not found: {predictions_path}")
        print("  Run evaluate.py (Task 14) first to generate all_predictions.json")
        sys.exit(1)

    with open(predictions_path) as f:
        data = json.load(f)

    true_labels: list[int] = data["true_labels"]
    pred_labels: list[int] = data["pred_labels"]

    print("=" * 62)
    print("  TASK 15: CLASSIFICATION METRICS")
    print("=" * 62)
    print(f"  Loaded {len(true_labels)} predictions from: {predictions_path}\n")

    # --- Compute all metrics ---
    cm         = build_confusion_matrix(true_labels, pred_labels, NUM_CLASSES)
    per_class  = per_class_metrics(cm, NUM_CLASSES)
    macro      = macro_avg(per_class)
    weighted   = weighted_avg(per_class)
    acc        = overall_accuracy(true_labels, pred_labels)

    # --- Print report ---
    print(f"  Overall accuracy : {acc:.4f}  ({int(acc * len(true_labels))}/{len(true_labels)})\n")
    print(f"  {'Class':<15s}  {'Precision':>9s}  {'Recall':>7s}  {'F1':>7s}  {'Support':>7s}")
    print("  " + "-" * 50)
    for cls, m in per_class.items():
        print(
            f"  {cls:<15s}  {m['precision']:>9.4f}  {m['recall']:>7.4f}  "
            f"{m['f1_score']:>7.4f}  {m['support']:>7d}"
        )
    print("  " + "-" * 50)
    print(
        f"  {'macro avg':<15s}  {macro['precision']:>9.4f}  {macro['recall']:>7.4f}  "
        f"{macro['f1_score']:>7.4f}"
    )
    print(
        f"  {'weighted avg':<15s}  {weighted['precision']:>9.4f}  {weighted['recall']:>7.4f}  "
        f"{weighted['f1_score']:>7.4f}"
    )

    print("\n  Confusion Matrix (rows=true, cols=predicted):")
    header = "  " + " " * 16 + "  ".join(f"{c[:8]:>8s}" for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8d}" for v in row)
        print(f"  {CLASSES[i]:<15s}  {row_str}")

    # --- Save JSON report ---
    os.makedirs(METRICS_DIR, exist_ok=True)
    report = {
        "task":           "Task 15: Classification Metrics",
        "computed_at":    datetime.now().isoformat(),
        "num_samples":    len(true_labels),
        "overall_accuracy": acc,
        "per_class":      per_class,
        "macro_avg":      macro,
        "weighted_avg":   weighted,
        "confusion_matrix": cm,
        "classes":        CLASSES,
    }
    with open(REPORT_OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Metrics report   → {REPORT_OUTPUT_PATH}")

    # --- Save confusion matrix plot ---
    saved = plot_confusion_matrix(cm, CONFUSION_MATRIX_PATH)
    if saved:
        print(f"  Confusion matrix → {CONFUSION_MATRIX_PATH}")
    else:
        print("  (matplotlib not available — confusion_matrix.png skipped)")

    print("\nDone.")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute classification metrics from saved predictions"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        metavar="PATH",
        help=f"Path to all_predictions.json (default: {ALL_PREDICTIONS_PATH})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
