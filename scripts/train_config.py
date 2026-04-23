"""
Task 12: Training Parameter Configuration
- Centralizes ALL hyperparameters, paths, and training settings in one place
- Single source of truth imported by train.py and any evaluation scripts
- Adjustable constants with rationale documented for each decision

Two-phase transfer learning strategy:
    Phase 1 — Feature Extraction:  backbone frozen, train classifier head (5 epochs, LR=1e-3)
    Phase 2 — Fine-Tuning:         last 3 ResNet blocks + head unfrozen (20 epochs, LR=1e-5)

Class label encoding (consistent across the entire project):
    glioma      → 0
    meningioma  → 1
    no_tumor    → 2
    pituitary   → 3

Run from project root:
    python scripts/train_config.py        # prints full config summary
"""

import os
import json
from datetime import datetime

# ===========================================================================
# PATHS
# ===========================================================================

# Root directories — adjust if running from a different working directory
PROJECT_ROOT   = "."                              # set to your Colab mount path if needed
DATA_DIR       = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR      = os.path.join(DATA_DIR, "processed", "train")
TEST_DIR       = os.path.join(DATA_DIR, "processed", "test")
MODELS_DIR     = os.path.join(PROJECT_ROOT, "models")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")
OUTPUTS_DIR    = os.path.join(PROJECT_ROOT, "outputs")
METRICS_DIR    = os.path.join(OUTPUTS_DIR, "metrics")
PLOTS_DIR      = os.path.join(OUTPUTS_DIR, "plots")

# Final saved model weights (best Phase 2 checkpoint)
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pt")

# ===========================================================================
# DATASET
# ===========================================================================

CLASSES     = ["glioma", "meningioma", "no_tumor", "pituitary"]
NUM_CLASSES = len(CLASSES)

CLASS_TO_IDX: dict[str, int] = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: cls for cls, i in CLASS_TO_IDX.items()}

NUM_TEST_BATCHES = 4          # test set split into 4 batches of 100 images/class each
IMG_SIZE         = 224        # spatial resolution fed to ResNet-50 (must match preprocess.py)

# ===========================================================================
# DATA LOADING
# ===========================================================================

BATCH_SIZE   = 32             # mini-batch size; reduce to 16 if Colab T4 OOM
NUM_WORKERS  = 2              # parallel CPU workers for DataLoader
                              # (set to 0 if Colab raises RuntimeError with workers)
RANDOM_SEED  = 42             # ensures reproducible DataLoader shuffling

# ImageNet normalization statistics (must match dataloader.py build_train_transforms)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ===========================================================================
# MODEL ARCHITECTURE
# ===========================================================================

ARCHITECTURE    = "ResNet-50"
BACKBONE        = "resnet50"                 # torchvision.models identifier
PRETRAINED      = True                       # load ImageNet IMAGENET1K_V1 weights
BACKBONE_OUT_FEATURES = 2048                 # ResNet-50 fc input dimension

# Classifier head: Dropout(p=DROPOUT_RATE) → Linear(2048 → NUM_CLASSES)
DROPOUT_RATE = 0.5
# Rationale: 0.5 dropout is the primary overfitting defence for this ~5,600-image dataset.
# Combined with weight_decay (L2 regularization) it substantially reduces generalization gap.
# Source: Scientific Reports ResNet-50 brain tumor study used comparable regularization.

# ===========================================================================
# OPTIMIZER — Adam
# ===========================================================================

OPTIMIZER = "Adam"
# Rationale: Adam's adaptive learning rates per-parameter are well-suited to transfer
# learning where different layers need different effective step sizes. Consistent with
# the majority of Kaggle and literature baselines on this dataset.

# ===========================================================================
# LOSS FUNCTION
# ===========================================================================

LOSS_FUNCTION = "CrossEntropyLoss"
# Rationale: Standard multi-class classification loss. Combines LogSoftmax + NLLLoss
# in a numerically stable form. No class weighting applied because the training set
# is balanced at 1,400 images/class after split_dataset.py stratification.

# ===========================================================================
# LEARNING RATE SCHEDULER
# ===========================================================================

SCHEDULER          = "ReduceLROnPlateau"
SCHEDULER_MODE     = "max"    # monitor validation accuracy (higher = better)
SCHEDULER_PATIENCE = 3        # epochs to wait before reducing LR
SCHEDULER_FACTOR   = 0.5      # multiply LR by this on plateau (halve it)
# Rationale: Plateau detection prevents wasting GPU time on unproductive LR values.
# patience=3 allows the model to recover from short stagnation periods before cutting LR.
# factor=0.5 is a conservative reduction that avoids overcorrecting.

# ===========================================================================
# PHASE 1 — FEATURE EXTRACTION
# ===========================================================================
# Freeze entire backbone; train only the randomly-initialized classifier head.
# Purpose: Establish a stable linear mapping from pretrained ImageNet features to
# brain tumor classes before disturbing any pretrained backbone weights.

PHASE1_EPOCHS       = 5
PHASE1_LR           = 1e-3
PHASE1_WEIGHT_DECAY = 1e-4
# Rationale: Higher LR (1e-3) is appropriate for the randomly initialized head —
# the pretrained weights are frozen so there is no risk of catastrophic forgetting.
# 5 epochs is sufficient for the head to converge on the frozen backbone features.

PHASE1_FROZEN_MODULES    = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
PHASE1_TRAINABLE_MODULES = ["fc"]

# ===========================================================================
# PHASE 2 — FINE-TUNING
# ===========================================================================
# Unfreeze last 3 ResNet blocks (layer2, layer3, layer4) + head.
# Keep conv1, bn1, layer1 frozen — low-level edge detectors are universal.

PHASE2_EPOCHS       = 20
PHASE2_LR           = 1e-5
PHASE2_WEIGHT_DECAY = 1e-4
# Rationale: Low LR (1e-5) prevents catastrophic forgetting of ImageNet features
# while allowing deeper layers to adapt to MRI-specific textures and shapes.
# 20 epochs with early stopping (patience=5) guards against overfitting on small dataset.

PHASE2_FROZEN_MODULES    = ["conv1", "bn1", "layer1"]
PHASE2_TRAINABLE_MODULES = ["layer2", "layer3", "layer4", "fc"]

# Early stopping (Phase 2 only — Phase 1 always runs its full 5 epochs)
EARLY_STOPPING_PATIENCE = 5   # stop if val accuracy doesn't improve for 5 consecutive epochs
# Rationale: Prevents overfitting and wastes no Colab GPU time once the model has
# converged. 5-epoch patience allows recovery from short plateaus. (Risk #1 mitigation)

# ===========================================================================
# CHECKPOINT STRATEGY (Risk #5: GPU session safety)
# ===========================================================================

SAVE_EVERY_EPOCH    = True    # save checkpoint after each epoch regardless of improvement
# Rationale: Colab sessions can time out unexpectedly. Per-epoch checkpoints ensure
# training can be resumed from the last completed epoch, not from scratch.

KEEP_BEST_ONLY      = False   # if True, overwrite checkpoint only when val_acc improves
# Set False to keep all epoch checkpoints for analysis; True to save disk space.

CHECKPOINT_FILENAME_TEMPLATE = "phase{phase}_ep{epoch:02d}_acc{val_acc:.4f}.pt"
# Example output: phase2_ep07_acc0.9412.pt

# ===========================================================================
# LOGGING / METRICS
# ===========================================================================

TRAINING_LOG_PATH = os.path.join(METRICS_DIR, "training_log.json")
# Stores per-epoch loss, train_acc, val_acc, and LR for both phases.

RESULTS_SUMMARY_PATH = os.path.join(METRICS_DIR, "training_results.json")
# Final summary written at end of training with best epoch, best val_acc, total time.

# ===========================================================================
# REPRODUCIBILITY
# ===========================================================================

TORCH_SEED  = 42   # set via torch.manual_seed() at start of train.py
NUMPY_SEED  = 42   # set via numpy.random.seed() if numpy is used


# ===========================================================================
# HELPER: export full config as dict (for JSON logging)
# ===========================================================================

def get_config_dict() -> dict:
    """Return all training parameters as a serializable dict for logging."""
    return {
        "task":           "Task 12: Training Configuration",
        "generated_at":   datetime.now().isoformat(),

        "paths": {
            "train_dir":      TRAIN_DIR,
            "test_dir":       TEST_DIR,
            "checkpoint_dir": CHECKPOINT_DIR,
            "best_model":     BEST_MODEL_PATH,
            "training_log":   TRAINING_LOG_PATH,
        },

        "dataset": {
            "classes":         CLASSES,
            "num_classes":     NUM_CLASSES,
            "class_to_idx":    CLASS_TO_IDX,
            "num_test_batches": NUM_TEST_BATCHES,
            "img_size":        IMG_SIZE,
        },

        "data_loading": {
            "batch_size":   BATCH_SIZE,
            "num_workers":  NUM_WORKERS,
            "random_seed":  RANDOM_SEED,
        },

        "model": {
            "architecture":    ARCHITECTURE,
            "backbone":        BACKBONE,
            "pretrained":      PRETRAINED,
            "dropout_rate":    DROPOUT_RATE,
            "num_classes":     NUM_CLASSES,
        },

        "optimizer":      OPTIMIZER,
        "loss_function":  LOSS_FUNCTION,

        "scheduler": {
            "type":     SCHEDULER,
            "mode":     SCHEDULER_MODE,
            "patience": SCHEDULER_PATIENCE,
            "factor":   SCHEDULER_FACTOR,
        },

        "phase_1": {
            "epochs":          PHASE1_EPOCHS,
            "lr":              PHASE1_LR,
            "weight_decay":    PHASE1_WEIGHT_DECAY,
            "frozen_modules":  PHASE1_FROZEN_MODULES,
            "trainable":       PHASE1_TRAINABLE_MODULES,
        },

        "phase_2": {
            "epochs":               PHASE2_EPOCHS,
            "lr":                   PHASE2_LR,
            "weight_decay":         PHASE2_WEIGHT_DECAY,
            "frozen_modules":       PHASE2_FROZEN_MODULES,
            "trainable":            PHASE2_TRAINABLE_MODULES,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },

        "checkpointing": {
            "save_every_epoch": SAVE_EVERY_EPOCH,
            "keep_best_only":   KEEP_BEST_ONLY,
            "filename_template": CHECKPOINT_FILENAME_TEMPLATE,
        },

        "reproducibility": {
            "torch_seed": TORCH_SEED,
            "numpy_seed": NUMPY_SEED,
        },
    }


def save_config_json(path: str = os.path.join(METRICS_DIR, "train_config.json")) -> None:
    """Persist the full config dict to a JSON file for experiment reproducibility."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(get_config_dict(), f, indent=2)
    print(f"  Config saved → {path}")


# ===========================================================================
# CLI — print full config summary
# ===========================================================================

if __name__ == "__main__":
    print("=" * 62)
    print("  TASK 12: TRAINING PARAMETER CONFIGURATION")
    print("=" * 62)

    cfg = get_config_dict()

    print("\n  PATHS")
    print(f"    Train dir      : {TRAIN_DIR}")
    print(f"    Test dir       : {TEST_DIR}")
    print(f"    Checkpoints    : {CHECKPOINT_DIR}")
    print(f"    Best model     : {BEST_MODEL_PATH}")
    print(f"    Training log   : {TRAINING_LOG_PATH}")

    print("\n  DATASET")
    print(f"    Classes        : {CLASSES}")
    print(f"    Label mapping  : {CLASS_TO_IDX}")
    print(f"    Image size     : {IMG_SIZE}×{IMG_SIZE}")
    print(f"    Test batches   : {NUM_TEST_BATCHES}")

    print("\n  DATA LOADING")
    print(f"    Batch size     : {BATCH_SIZE}")
    print(f"    Num workers    : {NUM_WORKERS}")
    print(f"    Random seed    : {RANDOM_SEED}")

    print("\n  MODEL")
    print(f"    Architecture   : {ARCHITECTURE}")
    print(f"    Pretrained     : ImageNet (IMAGENET1K_V1)")
    print(f"    Dropout        : {DROPOUT_RATE}")
    print(f"    Classifier     : Dropout({DROPOUT_RATE}) → Linear(2048 → {NUM_CLASSES})")

    print("\n  LOSS / OPTIMIZER / SCHEDULER")
    print(f"    Loss           : {LOSS_FUNCTION}")
    print(f"    Optimizer      : {OPTIMIZER}")
    print(f"    Scheduler      : {SCHEDULER}(mode='{SCHEDULER_MODE}', "
          f"patience={SCHEDULER_PATIENCE}, factor={SCHEDULER_FACTOR})")

    print("\n  PHASE 1 — Feature Extraction")
    print(f"    Epochs         : {PHASE1_EPOCHS}")
    print(f"    Learning rate  : {PHASE1_LR}")
    print(f"    Weight decay   : {PHASE1_WEIGHT_DECAY}")
    print(f"    Frozen         : {', '.join(PHASE1_FROZEN_MODULES)}")
    print(f"    Trainable      : {', '.join(PHASE1_TRAINABLE_MODULES)}")

    print("\n  PHASE 2 — Fine-Tuning")
    print(f"    Epochs         : {PHASE2_EPOCHS}  (+ early stopping patience={EARLY_STOPPING_PATIENCE})")
    print(f"    Learning rate  : {PHASE2_LR}")
    print(f"    Weight decay   : {PHASE2_WEIGHT_DECAY}")
    print(f"    Frozen         : {', '.join(PHASE2_FROZEN_MODULES)}")
    print(f"    Trainable      : {', '.join(PHASE2_TRAINABLE_MODULES)}")

    print("\n  CHECKPOINTING")
    print(f"    Save every epoch : {SAVE_EVERY_EPOCH}")
    print(f"    Filename format  : {CHECKPOINT_FILENAME_TEMPLATE}")

    print("\n  Saving config JSON...")
    save_config_json()

    print("\nDone.")
