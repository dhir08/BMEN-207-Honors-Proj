"""
Task 19: Benchmark — Compare Against Kaggle Baseline
=====================================================
Loads our trained model's results and compares them against documented
Kaggle and peer-reviewed literature baselines for the 4-class brain tumor
MRI classification task (glioma / meningioma / no_tumor / pituitary).

Outputs:
  outputs/metrics/benchmark_comparison.json  — full comparison table
  outputs/plots/benchmark_bar.png            — bar chart comparing F1 scores

Literature baselines (sourced during Task 10 architecture selection):
  • ResNet-50 fine-tuned   — 99.68%  (Scientific Reports, 2023)
  • EfficientNet-B3        — 99.23%  (GitHub zacharyvunguyen, 2022)
  • EfficientNet-B0        — 98.97%  (Springer Nature, 2022)
  • VGG-16 (baseline)      — 94.82%  (commonly reported on this Kaggle dataset)
  • Custom CNN (no TL)     — 86.40%  (typical non-transfer learning baseline)

Usage (run after evaluate.py and metrics.py):
    python scripts/benchmark.py
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_config import METRICS_DIR, PLOTS_DIR, CLASSES

BENCHMARK_OUTPUT_PATH = os.path.join(METRICS_DIR, "benchmark_comparison.json")
BENCHMARK_PLOT_PATH   = os.path.join(PLOTS_DIR,   "benchmark_bar.png")
EVAL_RESULTS_PATH     = os.path.join(METRICS_DIR, "evaluation_results.json")
REPORT_PATH           = os.path.join(METRICS_DIR, "classification_report.json")


# ===========================================================================
# LITERATURE BASELINES (Task 10 — model_selection.py sources)
# ===========================================================================

BASELINES = [
    {
        "name":          "ResNet-50 (Fine-tuned)",
        "accuracy_pct":  99.68,
        "weighted_f1":   None,         # paper reports accuracy only
        "source":        "Scientific Reports 2023 — 4-class Kaggle brain tumor",
        "notes":         "Same dataset, same architecture as our model",
    },
    {
        "name":          "EfficientNet-B3",
        "accuracy_pct":  99.23,
        "weighted_f1":   None,
        "source":        "GitHub: zacharyvunguyen (2022)",
        "notes":         "Compound scaling — higher parameter count than B0",
    },
    {
        "name":          "EfficientNet-B0",
        "accuracy_pct":  98.97,
        "weighted_f1":   None,
        "source":        "Springer Nature 2022 — Lightweight Transfer Learning",
        "notes":         "Efficient alternative; 5.3M params vs ResNet-50's 25.5M",
    },
    {
        "name":          "VGG-16 (Transfer Learning)",
        "accuracy_pct":  94.82,
        "weighted_f1":   None,
        "source":        "Kaggle community benchmarks (commonly reported)",
        "notes":         "Older architecture; included as mid-tier baseline",
    },
    {
        "name":          "Custom CNN (No Transfer Learning)",
        "accuracy_pct":  86.40,
        "weighted_f1":   None,
        "source":        "Kaggle community benchmarks (commonly reported)",
        "notes":         "Training from scratch on ~5,600 images",
    },
]


# ===========================================================================
# LOAD OUR RESULTS
# ===========================================================================

def load_our_results() -> dict | None:
    """Load evaluation + metrics results. Returns None if not yet generated."""
    our = {"accuracy_pct": None, "weighted_f1": None, "per_class_f1": {}}

    if os.path.isfile(EVAL_RESULTS_PATH):
        with open(EVAL_RESULTS_PATH) as f:
            eval_data = json.load(f)
        acc = eval_data.get("overall", {}).get("accuracy")
        if acc is not None:
            our["accuracy_pct"] = round(acc * 100, 2)

    if os.path.isfile(REPORT_PATH):
        with open(REPORT_PATH) as f:
            report_data = json.load(f)
        wavg = report_data.get("weighted_avg", {})
        our["weighted_f1"] = wavg.get("f1_score")
        per_class = report_data.get("per_class", {})
        for cls in CLASSES:
            our["per_class_f1"][cls] = per_class.get(cls, {}).get("f1_score")

    return our


# ===========================================================================
# PRINT COMPARISON TABLE
# ===========================================================================

def print_comparison_table(our: dict, baselines: list) -> None:
    print("\n  Benchmark Comparison Table")
    print("  " + "-" * 68)
    print(f"  {'Model':<38s}  {'Accuracy (%)':>12s}  {'Weighted F1':>11s}")
    print("  " + "-" * 68)

    # Our model first
    our_acc_str = f"{our['accuracy_pct']:.2f}" if our["accuracy_pct"] else "pending"
    our_f1_str  = f"{our['weighted_f1']:.4f}" if our["weighted_f1"] else "pending"
    print(f"  {'★ Our Model (ResNet-50, this project)':<38s}  {our_acc_str:>12s}  {our_f1_str:>11s}")

    print()
    for b in baselines:
        acc_str = f"{b['accuracy_pct']:.2f}" if b["accuracy_pct"] else "N/A"
        f1_str  = f"{b['weighted_f1']:.4f}" if b["weighted_f1"] else "N/A"
        print(f"  {b['name']:<38s}  {acc_str:>12s}  {f1_str:>11s}")

    print("  " + "-" * 68)

    if our["accuracy_pct"]:
        best_baseline = max(b["accuracy_pct"] for b in baselines if b["accuracy_pct"])
        delta         = our["accuracy_pct"] - best_baseline
        label         = "above" if delta >= 0 else "below"
        print(f"\n  Our model is {abs(delta):.2f}pp {label} the best literature baseline ({best_baseline:.2f}%)")


# ===========================================================================
# BAR CHART
# ===========================================================================

def plot_benchmark_bar(our: dict, baselines: list, output_path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    names    = ["Our Model"] + [b["name"].replace("(", "\n(") for b in baselines]
    accs     = [our["accuracy_pct"] or 0] + [b["accuracy_pct"] or 0 for b in baselines]
    colors   = ["#2196F3"] + ["#90CAF9"] * len(baselines)

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(names, accs, color=colors, edgecolor="white", height=0.55)

    # Annotate bars
    for bar, acc in zip(bars, accs):
        if acc:
            ax.text(acc + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{acc:.2f}%", va="center", ha="left", fontsize=9)

    ax.set_xlabel("Test Accuracy (%)", fontsize=11)
    ax.set_title(
        "Brain Tumor Classification — Model Accuracy Benchmark\n"
        "(4-class: glioma / meningioma / no_tumor / pituitary)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(80, 101)
    ax.invert_yaxis()                  # highest bar at top
    ax.axvline(x=accs[0], color="#1565C0", linestyle="--", linewidth=1.2, label="Our model")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close()
    return True


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    print("=" * 62)
    print("  TASK 19: BENCHMARK COMPARISON")
    print("=" * 62)

    our = load_our_results()

    if our["accuracy_pct"] is None:
        print("\n  NOTE: evaluation_results.json not found.")
        print("  Run evaluate.py (Task 14) after training to populate our results.")
        print("  Showing baseline-only table for now.\n")

    print_comparison_table(our, BASELINES)

    # Build output dict
    comparison = {
        "task":          "Task 19: Benchmark Comparison",
        "generated_at":  datetime.now().isoformat(),
        "our_model": {
            "name":         "ResNet-50 (This Project)",
            "accuracy_pct": our["accuracy_pct"],
            "weighted_f1":  our["weighted_f1"],
            "per_class_f1": our["per_class_f1"],
        },
        "baselines": BASELINES,
        "analysis": {
            "best_baseline_accuracy": max(
                b["accuracy_pct"] for b in BASELINES if b["accuracy_pct"]
            ),
            "delta_vs_best_baseline": (
                round(our["accuracy_pct"] -
                      max(b["accuracy_pct"] for b in BASELINES if b["accuracy_pct"]), 2)
                if our["accuracy_pct"] else None
            ),
        },
    }

    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(BENCHMARK_OUTPUT_PATH, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Benchmark report → {BENCHMARK_OUTPUT_PATH}")

    saved = plot_benchmark_bar(our, BASELINES, BENCHMARK_PLOT_PATH)
    if saved:
        print(f"  Benchmark chart  → {BENCHMARK_PLOT_PATH}")
    else:
        print("  (matplotlib not available — benchmark_bar.png skipped)")

    print("\nDone.")


if __name__ == "__main__":
    main()
