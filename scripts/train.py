"""
Task 13: Full Two-Phase Training Loop — Google Colab (GPU)
==========================================================
Trains the ResNet-50 brain tumor classifier in two phases:

  Phase 1 — Feature Extraction (5 epochs, LR=1e-3)
    Backbone frozen; only the classifier head (fc) is trained.
    Establishes a stable linear mapping before touching pretrained weights.

  Phase 2 — Fine-Tuning (up to 20 epochs, LR=1e-5, early stopping patience=5)
    Unfreezes last 3 ResNet blocks (layer2, layer3, layer4) + head.
    Low LR prevents catastrophic forgetting of ImageNet features.

Features:
  - Per-epoch checkpointing (Colab GPU session safety — Risk #5)
  - Best-model tracking by validation accuracy
  - Early stopping in Phase 2 (Risk #1 overfitting mitigation)
  - Full training log saved as JSON after every epoch
  - Reproducible: seeds set for torch and numpy at startup

Usage (Google Colab):
    # Mount Google Drive first, then:
    import sys
    sys.path.insert(0, "/content/drive/MyDrive/<your-project-root>")

    !python scripts/train.py                              # full training from scratch
    !python scripts/train.py --resume models/checkpoints/phase1_ep05_acc0.8750.pt --phase 2
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

# ---------------------------------------------------------------------------
# Guard: PyTorch must be available
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Allow running as: python scripts/train.py  (from project root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_config import (
    # Paths
    TRAIN_DIR, TEST_DIR, CHECKPOINT_DIR, BEST_MODEL_PATH,
    TRAINING_LOG_PATH, RESULTS_SUMMARY_PATH, METRICS_DIR,
    # Dataset
    CLASSES, NUM_CLASSES, NUM_TEST_BATCHES,
    # Data loading
    BATCH_SIZE, NUM_WORKERS, RANDOM_SEED,
    # Optimizer / scheduler
    PHASE1_LR, PHASE1_WEIGHT_DECAY, PHASE1_EPOCHS,
    PHASE2_LR, PHASE2_WEIGHT_DECAY, PHASE2_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    SCHEDULER_MODE, SCHEDULER_PATIENCE, SCHEDULER_FACTOR,
    CHECKPOINT_FILENAME_TEMPLATE,
    TORCH_SEED, NUMPY_SEED,
    get_config_dict,
)

from scripts.model import (
    build_model,
    freeze_backbone,
    unfreeze_for_finetuning,
    get_optimizer,
    get_scheduler,
    save_checkpoint,
    load_checkpoint,
    describe_model,
)

from scripts.dataloader import get_train_loader, get_test_batch_loader


# ===========================================================================
# REPRODUCIBILITY
# ===========================================================================

def set_seeds(torch_seed: int = TORCH_SEED, numpy_seed: int = NUMPY_SEED) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    try:
        import numpy as np
        np.random.seed(numpy_seed)
    except ImportError:
        pass


# ===========================================================================
# TRAINING & VALIDATION UTILITIES
# ===========================================================================

def train_one_epoch(
    model:       "nn.Module",
    loader:      "torch.utils.data.DataLoader",
    optimizer:   "torch.optim.Optimizer",
    criterion:   "nn.Module",
    device:      "torch.device",
    epoch_num:   int,
    total_epochs: int,
    phase:       int,
) -> tuple[float, float]:
    """
    Run one full training epoch.

    Args:
        model:        The ResNet-50 model in train mode.
        loader:       Training DataLoader.
        optimizer:    Adam optimizer for the current phase.
        criterion:    CrossEntropyLoss.
        device:       CPU or CUDA device.
        epoch_num:    1-based current epoch number.
        total_epochs: Total epochs planned for this phase.
        phase:        1 or 2 (for display only).

    Returns:
        (avg_loss, accuracy) over the full training set.
    """
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds        = logits.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += images.size(0)

        # Progress indicator every 20 batches
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(loader):
            print(
                f"    Phase {phase} | Epoch {epoch_num}/{total_epochs} "
                f"| Batch {batch_idx+1}/{len(loader)} "
                f"| Loss: {running_loss/total:.4f} "
                f"| Acc: {correct/total:.4f}",
                end="\r",
            )

    print()  # newline after \r progress
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model:     "nn.Module",
    loader:    "torch.utils.data.DataLoader",
    criterion: "nn.Module",
    device:    "torch.device",
) -> tuple[float, float]:
    """
    Evaluate the model on a validation/test DataLoader.

    Args:
        model:     The ResNet-50 model.
        loader:    DataLoader (test batch 1 used as validation proxy during training).
        criterion: CrossEntropyLoss.
        device:    CPU or CUDA device.

    Returns:
        (avg_loss, accuracy) over the full validation set.
    """
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits       = model(images)
            loss         = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            preds        = logits.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += images.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ===========================================================================
# CHECKPOINT NAMING
# ===========================================================================

def make_checkpoint_path(phase: int, epoch: int, val_acc: float) -> str:
    """Build a checkpoint file path from the template in train_config.py."""
    filename = CHECKPOINT_FILENAME_TEMPLATE.format(
        phase=phase, epoch=epoch, val_acc=val_acc
    )
    return os.path.join(CHECKPOINT_DIR, filename)


# ===========================================================================
# TRAINING LOG
# ===========================================================================

def append_epoch_log(
    log: list,
    phase: int,
    epoch: int,
    train_loss: float,
    train_acc:  float,
    val_loss:   float,
    val_acc:    float,
    lr:         float,
    elapsed:    float,
) -> None:
    """Append one epoch's metrics to the running log list (in-place)."""
    log.append({
        "phase":      phase,
        "epoch":      epoch,
        "train_loss": round(train_loss, 6),
        "train_acc":  round(train_acc,  6),
        "val_loss":   round(val_loss,   6),
        "val_acc":    round(val_acc,    6),
        "lr":         lr,
        "elapsed_s":  round(elapsed,    1),
        "timestamp":  datetime.now().isoformat(),
    })


def save_log(log: list, path: str) -> None:
    """Write the training log to disk as a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)


# ===========================================================================
# PHASE 1 — FEATURE EXTRACTION
# ===========================================================================

def run_phase1(
    model:       "nn.Module",
    train_loader: "torch.utils.data.DataLoader",
    val_loader:  "torch.utils.data.DataLoader",
    criterion:   "nn.Module",
    device:      "torch.device",
    log:         list,
) -> tuple["nn.Module", float]:
    """
    Phase 1 — Feature Extraction.
    Train only the classifier head with backbone frozen for PHASE1_EPOCHS epochs.

    Args:
        model:        ResNet-50 with backbone already frozen via freeze_backbone().
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader (test batch 1).
        criterion:    CrossEntropyLoss.
        device:       CUDA or CPU device.
        log:          Shared training log list (appended to in-place).

    Returns:
        (model, best_phase1_val_acc)
    """
    print("\n" + "=" * 62)
    print("  PHASE 1 — FEATURE EXTRACTION")
    print(f"  Epochs: {PHASE1_EPOCHS}  |  LR: {PHASE1_LR}  |  Frozen: backbone")
    print("=" * 62)

    optimizer = get_optimizer(model, phase=1)
    scheduler = get_scheduler(optimizer, patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR)

    best_val_acc = 0.0
    best_ckpt    = None

    for epoch in range(1, PHASE1_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch_num=epoch, total_epochs=PHASE1_EPOCHS, phase=1,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - epoch_start

        # Get current LR (after potential scheduler step)
        current_lr = optimizer.param_groups[0]["lr"]

        # Step scheduler on validation accuracy
        scheduler.step(val_acc)

        # Print epoch summary
        print(
            f"  [Phase 1 | Epoch {epoch:02d}/{PHASE1_EPOCHS:02d}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  "
            f"LR: {current_lr:.2e}  ({elapsed:.1f}s)"
        )

        # Log metrics
        append_epoch_log(log, 1, epoch, train_loss, train_acc, val_loss, val_acc,
                         current_lr, elapsed)
        save_log(log, TRAINING_LOG_PATH)

        # Save checkpoint every epoch
        ckpt_path = make_checkpoint_path(phase=1, epoch=epoch, val_acc=val_acc)
        save_checkpoint(model, optimizer, epoch, ckpt_path,
                        extra={"phase": 1, "val_acc": val_acc, "train_acc": train_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt    = ckpt_path
            print(f"  ★ New best Phase 1 val acc: {best_val_acc:.4f}")

    print(f"\n  Phase 1 complete. Best val acc: {best_val_acc:.4f}")
    if best_ckpt:
        print(f"  Best checkpoint: {best_ckpt}")

    return model, best_val_acc


# ===========================================================================
# PHASE 2 — FINE-TUNING
# ===========================================================================

def run_phase2(
    model:       "nn.Module",
    train_loader: "torch.utils.data.DataLoader",
    val_loader:  "torch.utils.data.DataLoader",
    criterion:   "nn.Module",
    device:      "torch.device",
    log:         list,
) -> tuple["nn.Module", float, int]:
    """
    Phase 2 — Fine-Tuning.
    Unfreeze last 3 ResNet blocks; train end-to-end with low LR and early stopping.

    Args:
        model:        ResNet-50 with Phase 1 weights loaded.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader (test batch 1).
        criterion:    CrossEntropyLoss.
        device:       CUDA or CPU device.
        log:          Shared training log list (appended to in-place).

    Returns:
        (model, best_val_acc, best_epoch)
    """
    print("\n" + "=" * 62)
    print("  PHASE 2 — FINE-TUNING")
    print(f"  Max epochs: {PHASE2_EPOCHS}  |  LR: {PHASE2_LR}  "
          f"|  Early stop patience: {EARLY_STOPPING_PATIENCE}")
    print("=" * 62)

    unfreeze_for_finetuning(model)
    describe_model(model)

    optimizer = get_optimizer(model, phase=2)
    scheduler = get_scheduler(optimizer, patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR)

    best_val_acc   = 0.0
    best_epoch     = 0
    patience_count = 0
    best_ckpt      = None

    for epoch in range(1, PHASE2_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch_num=epoch, total_epochs=PHASE2_EPOCHS, phase=2,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - epoch_start

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)

        print(
            f"  [Phase 2 | Epoch {epoch:02d}/{PHASE2_EPOCHS:02d}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  "
            f"LR: {current_lr:.2e}  ({elapsed:.1f}s)"
        )

        # Log metrics
        append_epoch_log(log, 2, epoch, train_loss, train_acc, val_loss, val_acc,
                         current_lr, elapsed)
        save_log(log, TRAINING_LOG_PATH)

        # Save checkpoint every epoch
        ckpt_path = make_checkpoint_path(phase=2, epoch=epoch, val_acc=val_acc)
        save_checkpoint(model, optimizer, epoch, ckpt_path,
                        extra={"phase": 2, "val_acc": val_acc, "train_acc": train_acc})

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_epoch     = epoch
            patience_count = 0
            best_ckpt      = ckpt_path
            # Also save as canonical best_model.pt
            save_checkpoint(model, optimizer, epoch, BEST_MODEL_PATH,
                            extra={"phase": 2, "val_acc": best_val_acc})
            print(f"  ★ New best val acc: {best_val_acc:.4f}  → saved to {BEST_MODEL_PATH}")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{EARLY_STOPPING_PATIENCE})")

        # Early stopping check
        if patience_count >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping triggered after {epoch} epochs "
                  f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs).")
            break

    print(f"\n  Phase 2 complete. Best val acc: {best_val_acc:.4f}  (epoch {best_epoch})")
    if best_ckpt:
        print(f"  Best checkpoint: {best_ckpt}")
        print(f"  Canonical best model: {BEST_MODEL_PATH}")

    return model, best_val_acc, best_epoch


# ===========================================================================
# MAIN TRAINING ENTRY POINT
# ===========================================================================

def main(args: argparse.Namespace) -> None:
    """
    Orchestrate full two-phase training.

    Args:
        args: Parsed command-line arguments.
              args.resume — path to checkpoint to resume from (optional)
              args.phase  — 1 or 2 (which phase to start from)
    """
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not found. Run this script in Google Colab:")
        print("  !pip install torch torchvision")
        sys.exit(1)

    # --- Setup ---
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Make output directories
    for d in [CHECKPOINT_DIR, METRICS_DIR]:
        os.makedirs(d, exist_ok=True)

    # Save config for reproducibility
    from scripts.train_config import save_config_json
    save_config_json()

    # --- DataLoaders ---
    print("\nBuilding DataLoaders...")
    train_loader = get_train_loader(
        train_dir=TRAIN_DIR,
        batch_size=BATCH_SIZE,
        seed=RANDOM_SEED,
        num_workers=NUM_WORKERS,
    )
    # Use test batch 1 as the validation set during training
    val_loader = get_test_batch_loader(
        test_dir=TEST_DIR,
        batch_num=1,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    print(f"  Train batches : {len(train_loader)}  ({len(train_loader.dataset):,} images)")
    print(f"  Val batches   : {len(val_loader)}   ({len(val_loader.dataset):,} images)")

    # --- Model ---
    print("\nBuilding model...")
    model     = build_model(pretrained=True)
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Shared epoch log across both phases
    log: list = []

    # --- Resume logic ---
    start_phase = args.phase
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        load_checkpoint(model, optimizer=None, path=args.resume)
        print(f"  Starting from Phase {start_phase}")

    # --- Phase 1 ---
    if start_phase == 1:
        freeze_backbone(model)
        describe_model(model)
        model, p1_best = run_phase1(
            model, train_loader, val_loader, criterion, device, log
        )
    else:
        print("\n  Skipping Phase 1 (--phase 2 specified or resuming).")

    # --- Phase 2 ---
    model, p2_best, p2_best_epoch = run_phase2(
        model, train_loader, val_loader, criterion, device, log
    )

    # --- Final summary ---
    print("\n" + "=" * 62)
    print("  TRAINING COMPLETE")
    print("=" * 62)
    print(f"  Best Phase 2 val acc : {p2_best:.4f}  (epoch {p2_best_epoch})")
    print(f"  Best model saved to  : {BEST_MODEL_PATH}")
    print(f"  Training log         : {TRAINING_LOG_PATH}")

    # Write results summary JSON
    results = {
        "task":              "Task 13: Training Loop",
        "completed_at":      datetime.now().isoformat(),
        "device":            str(device),
        "best_val_acc":      round(p2_best, 6),
        "best_epoch_phase2": p2_best_epoch,
        "total_epochs_run":  len(log),
        "best_model_path":   BEST_MODEL_PATH,
        "training_log_path": TRAINING_LOG_PATH,
        "config":            get_config_dict(),
    }
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(RESULTS_SUMMARY_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results summary      : {RESULTS_SUMMARY_PATH}")
    print("\nDone.")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brain Tumor MRI — Two-Phase ResNet-50 Training"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT_PATH",
        help="Path to a .pt checkpoint to resume from.",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[1, 2],
        help="Phase to start from (1=feature extraction, 2=fine-tuning). Default: 1.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
