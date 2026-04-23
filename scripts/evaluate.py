"""
Task 14: Model Evaluation — 4 Test Batches
===========================================
Loads the trained best_model.pt and evaluates it against all four test batches
to verify that accuracy is consistent (not inflated by a single lucky batch).

Outputs:
  outputs/metrics/evaluation_results.json   — per-batch & overall accuracy
  outputs/metrics/all_predictions.json      — raw (true_label, pred_label) pairs
                                              consumed by metrics.py (Task 15)

Usage (Google Colab — run after train.py completes):
    python scripts/evaluate.py
    python scripts/evaluate.py --model models/best_model.pt   # explicit path
    python scripts/evaluate.py --batches 1 2 3 4              # subset of batches
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

# ---------------------------------------------------------------------------
# Guard: PyTorch required
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_config import (
    BEST_MODEL_PATH, TEST_DIR, BATCH_SIZE, NUM_WORKERS,
    NUM_TEST_BATCHES, CLASSES, NUM_CLASSES, IDX_TO_CLASS,
    METRICS_DIR,
)
from scripts.model import build_model, load_checkpoint
from scripts.dataloader import get_test_batch_loader

EVAL_RESULTS_PATH   = os.path.join(METRICS_DIR, "evaluation_results.json")
ALL_PREDICTIONS_PATH = os.path.join(METRICS_DIR, "all_predictions.json")


# ===========================================================================
# BATCH EVALUATION
# ===========================================================================

def evaluate_batch(
    model:      "nn.Module",
    batch_num:  int,
    device:     "torch.device",
    criterion:  "nn.Module",
) -> dict:
    """
    Run inference on a single numbered test batch.

    Returns a dict with:
        batch_num, num_images, loss, accuracy,
        per_class_correct, per_class_total, per_class_accuracy,
        true_labels, pred_labels   (lists of ints for downstream metrics)
    """
    loader = get_test_batch_loader(
        test_dir=TEST_DIR,
        batch_num=batch_num,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0
    all_true:  list[int] = []
    all_pred:  list[int] = []

    per_class_correct = {cls: 0 for cls in CLASSES}
    per_class_total   = {cls: 0 for cls in CLASSES}

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)
            preds  = logits.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            correct      += (preds == labels).sum().item()
            total        += images.size(0)

            for true_lbl, pred_lbl in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                cls_name = IDX_TO_CLASS[true_lbl]
                per_class_correct[cls_name] += int(true_lbl == pred_lbl)
                per_class_total[cls_name]   += 1
                all_true.append(true_lbl)
                all_pred.append(pred_lbl)

    per_class_accuracy = {
        cls: (per_class_correct[cls] / per_class_total[cls]
              if per_class_total[cls] > 0 else 0.0)
        for cls in CLASSES
    }

    return {
        "batch_num":           batch_num,
        "num_images":          total,
        "loss":                round(running_loss / total, 6),
        "accuracy":            round(correct / total, 6),
        "per_class_correct":   per_class_correct,
        "per_class_total":     per_class_total,
        "per_class_accuracy":  {k: round(v, 6) for k, v in per_class_accuracy.items()},
        "true_labels":         all_true,
        "pred_labels":         all_pred,
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main(args: argparse.Namespace) -> None:
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not found. Run in Google Colab.")
        sys.exit(1)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # --- Load model ---
    model_path = args.model or BEST_MODEL_PATH
    if not os.path.isfile(model_path):
        print(f"ERROR: Model checkpoint not found: {model_path}")
        print("  Run train.py first to produce best_model.pt")
        sys.exit(1)

    print(f"\n  Loading model from: {model_path}")
    model = build_model(pretrained=False)   # pretrained=False: we load our own weights
    load_checkpoint(model, optimizer=None, path=model_path)
    model = model.to(device)
    model.eval()
    print(f"  Device: {device}\n")

    # --- Evaluate each batch ---
    batch_nums = args.batches or list(range(1, NUM_TEST_BATCHES + 1))
    batch_results = []
    all_true_global: list[int] = []
    all_pred_global: list[int] = []

    print("=" * 62)
    print("  TASK 14: MODEL EVALUATION — TEST BATCHES")
    print("=" * 62)

    for bn in batch_nums:
        t0 = time.time()
        print(f"  Evaluating test batch {bn}...")
        result = evaluate_batch(model, bn, device, criterion)
        elapsed = time.time() - t0

        # Accumulate for overall metrics
        all_true_global.extend(result.pop("true_labels"))
        all_pred_global.extend(result.pop("pred_labels"))
        result["elapsed_s"] = round(elapsed, 1)

        batch_results.append(result)

        print(
            f"    Batch {bn}: accuracy={result['accuracy']:.4f}  "
            f"loss={result['loss']:.4f}  ({result['num_images']} images, {elapsed:.1f}s)"
        )
        for cls in CLASSES:
            pa = result['per_class_accuracy'][cls]
            print(f"      {cls:<15s}: {pa:.4f}")

    # --- Overall metrics ---
    total_images   = sum(r["num_images"] for r in batch_results)
    total_correct  = sum(r["num_images"] * r["accuracy"] for r in batch_results)
    overall_acc    = total_correct / total_images if total_images > 0 else 0.0
    acc_values     = [r["accuracy"] for r in batch_results]
    acc_std        = float((sum((a - overall_acc) ** 2 for a in acc_values) / len(acc_values)) ** 0.5)

    print(f"\n  Overall accuracy : {overall_acc:.4f}")
    print(f"  Batch std dev    : {acc_std:.4f}  "
          f"({'consistent' if acc_std < 0.02 else 'variable'} across batches)")
    print(f"  Total images     : {total_images}")

    # --- Save results ---
    os.makedirs(METRICS_DIR, exist_ok=True)

    eval_output = {
        "task":           "Task 14: Model Evaluation",
        "evaluated_at":   datetime.now().isoformat(),
        "model_path":     model_path,
        "device":         str(device),
        "batches":        batch_results,
        "overall": {
            "accuracy":       round(overall_acc, 6),
            "accuracy_std":   round(acc_std, 6),
            "total_images":   total_images,
            "consistency":    "consistent" if acc_std < 0.02 else "variable",
        },
    }
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(eval_output, f, indent=2)
    print(f"\n  Evaluation results → {EVAL_RESULTS_PATH}")

    # Save flat predictions for metrics.py
    predictions_output = {
        "task":        "Task 14 → Task 15 handoff",
        "classes":     CLASSES,
        "true_labels": all_true_global,
        "pred_labels": all_pred_global,
    }
    with open(ALL_PREDICTIONS_PATH, "w") as f:
        json.dump(predictions_output, f, indent=2)
    print(f"  All predictions  → {ALL_PREDICTIONS_PATH}")
    print("\nDone.")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained brain tumor model on all test batches"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="PATH",
        help=f"Path to model checkpoint (default: {BEST_MODEL_PATH})",
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Specific batch numbers to evaluate (default: 1 2 3 4)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("=" * 62)
    print("  TASK 14: EVALUATE MODEL ON 4 TEST BATCHES")
    print("=" * 62)
    args = parse_args()
    main(args)
