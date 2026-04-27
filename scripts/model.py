"""
Task 11: CNN Model Architecture — ResNet-50 Transfer Learning
- Load pretrained ResNet-50 (ImageNet weights) via torchvision
- Replace classifier head: Dropout(0.5) → Linear(2048 → 4)
- freeze_backbone()         → Phase 1: train only the classifier head
- unfreeze_for_finetuning() → Phase 2: unfreeze last 3 ResNet blocks + head
- get_optimizer()           → Adam with weight_decay=1e-4
- get_scheduler()           → ReduceLROnPlateau(mode='max', patience=3, factor=0.5)
- save_checkpoint() / load_checkpoint() → GPU session safety (Risk #5)
- count_trainable_params()  → diagnostic helper

Class label encoding (must match dataloader.py throughout the project):
    glioma      → 0
    meningioma  → 1
    no_tumor    → 2
    pituitary   → 3

IMPORTANT: Requires PyTorch and torchvision.
  Install in Google Colab with:
    !pip install torch torchvision

Usage (in Colab training notebook):
    from scripts.model import build_model, freeze_backbone, unfreeze_for_finetuning
    from scripts.model import get_optimizer, get_scheduler, save_checkpoint, load_checkpoint

    # Phase 1 — feature extraction
    model = build_model(pretrained=True)
    freeze_backbone(model)
    optimizer  = get_optimizer(model, phase=1)
    scheduler  = get_scheduler(optimizer)

    for epoch in range(5):
        # ... training loop ...
        save_checkpoint(model, optimizer, epoch, "models/checkpoints/phase1_latest.pt")

    # Phase 2 — fine-tuning
    unfreeze_for_finetuning(model)
    optimizer = get_optimizer(model, phase=2)
    scheduler = get_scheduler(optimizer)

    for epoch in range(20):
        # ... training loop ...
        save_checkpoint(model, optimizer, epoch, "models/checkpoints/phase2_latest.pt")
"""

import os
import sys

# ---------------------------------------------------------------------------
# Guard: torch/torchvision are Colab-only (not installed in dev VM)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES     = ["glioma", "meningioma", "no_tumor", "pituitary"]
NUM_CLASSES = len(CLASSES)

CLASS_TO_IDX: dict[str, int] = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: cls for cls, i in CLASS_TO_IDX.items()}

# ResNet-50 architecture constants
BACKBONE_OUT_FEATURES = 2048     # ResNet-50 fc input dimension
DROPOUT_RATE          = 0.5      # validated for MRI overfitting prevention (Risk #1)
IMG_SIZE              = 224      # must match preprocess.py + dataloader.py

# Phase 1: freeze backbone, train head only
PHASE1_LR           = 1e-3
PHASE1_WEIGHT_DECAY = 1e-4
PHASE1_EPOCHS       = 5

# Phase 2: unfreeze last 3 ResNet blocks, fine-tune end-to-end
PHASE2_LR           = 1e-5
PHASE2_WEIGHT_DECAY = 1e-4
PHASE2_EPOCHS       = 20

# Scheduler
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR   = 0.5

CHECKPOINT_DIR = "models/checkpoints"


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(
    num_classes: int = NUM_CLASSES,
    dropout:     float = DROPOUT_RATE,
    pretrained:  bool = True,
) -> "nn.Module":
    """
    Build a ResNet-50 model with a custom classifier head for brain tumor classification.

    Architecture:
        ResNet-50 backbone (pretrained on ImageNet)
        └── classifier head (replaces original fc):
                Dropout(p=dropout)
                Linear(2048 → num_classes)

    Pretrained weights provide transferable edge/texture/structure features from
    ImageNet, significantly reducing the labeled-data requirement (addresses Risk #4).

    Args:
        num_classes: Number of output classes (default: 4).
        dropout:     Dropout probability in the classifier head (default: 0.5).
        pretrained:  If True, load ImageNet pretrained weights (default: True).

    Returns:
        ResNet-50 nn.Module with the modified classifier head.

    Raises:
        ImportError: If torch / torchvision are not installed.
    """
    _require_torch("build_model")

    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.resnet50(weights=weights)

    # Replace the default fc (1000-class ImageNet head) with our 4-class head.
    # Dropout(0.5) before the linear layer is the primary overfitting defence
    # alongside weight_decay in the optimizer (Risk #1 mitigation).
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(BACKBONE_OUT_FEATURES, num_classes),
    )

    return model


# ---------------------------------------------------------------------------
# Phase management
# ---------------------------------------------------------------------------

def freeze_backbone(model: "nn.Module") -> None:
    """
    Phase 1 — Feature Extraction: freeze all backbone layers; leave only
    the classifier head (model.fc) trainable.

    Frozen modules: conv1, bn1, layer1, layer2, layer3, layer4
    Trainable modules: fc

    This forces the network to learn a linear mapping from pretrained
    ImageNet features to brain tumor classes before any backbone weights
    are disturbed, giving a stable starting point for Phase 2.

    Args:
        model: ResNet-50 model returned by build_model().

    Raises:
        ImportError: If torch is not installed.
    """
    _require_torch("freeze_backbone")

    _set_requires_grad(model, False)                  # freeze everything
    for param in model.fc.parameters():               # unfreeze head only
        param.requires_grad = True

    n_trainable = count_trainable_params(model)
    print(f"  Phase 1 — backbone frozen. Trainable params: {n_trainable:,}")


def unfreeze_for_finetuning(model: "nn.Module") -> None:
    """
    Phase 2 — Fine-Tuning: unfreeze the last three ResNet blocks (layer2,
    layer3, layer4) plus the classifier head. Keep conv1, bn1, and layer1
    frozen to preserve low-level edge detectors that transfer universally.

    Frozen modules : conv1, bn1, layer1
    Trainable modules: layer2, layer3, layer4, fc

    Using a much smaller learning rate (1e-5) prevents catastrophic
    forgetting of the ImageNet features while allowing the deeper layers
    to adapt to MRI-specific texture and shape patterns.

    Args:
        model: ResNet-50 model (should have completed Phase 1 training).

    Raises:
        ImportError: If torch is not installed.
    """
    _require_torch("unfreeze_for_finetuning")

    # Keep earliest layers frozen (low-level features are universal)
    frozen_modules = [model.conv1, model.bn1, model.layer1]
    for module in frozen_modules:
        for param in module.parameters():
            param.requires_grad = False

    # Unfreeze everything else (layer2, layer3, layer4, fc)
    for name, module in model.named_children():
        if module not in frozen_modules:
            for param in module.parameters():
                param.requires_grad = True

    n_trainable = count_trainable_params(model)
    print(f"  Phase 2 — last 3 blocks unfrozen. Trainable params: {n_trainable:,}")


# ---------------------------------------------------------------------------
# Optimizer & scheduler factories
# ---------------------------------------------------------------------------

def get_optimizer(
    model: "nn.Module",
    phase: int = 1,
) -> "torch.optim.Optimizer":
    """
    Return an Adam optimizer configured for the specified training phase.

    Phase 1 (LR=1e-3): higher learning rate for training the randomly
    initialized classifier head from scratch.

    Phase 2 (LR=1e-5): low learning rate for fine-tuning pretrained
    backbone weights without catastrophic forgetting.

    weight_decay=1e-4 (L2 regularization) is applied in both phases as
    a secondary overfitting defence alongside dropout (Risk #1 mitigation).

    Args:
        model: ResNet-50 model with phase-appropriate layers set trainable.
        phase: 1 for feature extraction, 2 for fine-tuning.

    Returns:
        Configured Adam optimizer operating only on requires_grad=True params.

    Raises:
        ImportError: If torch is not installed.
        ValueError:  If phase is not 1 or 2.
    """
    _require_torch("get_optimizer")

    if phase not in (1, 2):
        raise ValueError(f"phase must be 1 or 2, got {phase}")

    lr           = PHASE1_LR           if phase == 1 else PHASE2_LR
    weight_decay = PHASE1_WEIGHT_DECAY  if phase == 1 else PHASE2_WEIGHT_DECAY

    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)


def get_scheduler(
    optimizer: "torch.optim.Optimizer",
    patience: int = SCHEDULER_PATIENCE,
    factor:   float = SCHEDULER_FACTOR,
) -> "torch.optim.lr_scheduler.ReduceLROnPlateau":
    """
    Return a ReduceLROnPlateau scheduler that monitors validation accuracy.

    Reduces LR by `factor` when accuracy stops improving for `patience`
    epochs. This prevents wasting Colab GPU time on unproductive epochs
    and helps escape local minima during Phase 2 (Risk #5 mitigation).

    Args:
        optimizer: The optimizer returned by get_optimizer().
        patience:  Number of epochs with no improvement before LR reduction.
        factor:    Factor to multiply LR by on plateau (e.g., 0.5 → halve LR).

    Returns:
        ReduceLROnPlateau scheduler (monitor: validation accuracy, mode: max).

    Raises:
        ImportError: If torch is not installed.
    """
    _require_torch("get_scheduler")

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",          # maximize validation accuracy
        patience=patience,
        factor=factor,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers (Risk #5: GPU session safety)
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:     "nn.Module",
    optimizer: "torch.optim.Optimizer",
    epoch:     int,
    path:      str,
    extra:     dict | None = None,
) -> None:
    """
    Save model + optimizer state to a .pt checkpoint file.

    Saving after every epoch ensures that Colab GPU session timeouts
    (Risk #5) don't result in total loss of training progress. The
    checkpoint can be resumed with load_checkpoint().

    Args:
        model:     The ResNet-50 model being trained.
        optimizer: The current optimizer (preserves momentum/state).
        epoch:     Current epoch index (0-based).
        path:      Destination file path (e.g., 'models/checkpoints/phase1_ep03.pt').
        extra:     Optional dict of additional metadata to save
                   (e.g., {'val_acc': 0.94, 'phase': 1}).

    Raises:
        ImportError: If torch is not installed.
        OSError:     If the checkpoint directory cannot be created.
    """
    _require_torch("save_checkpoint")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)

    torch.save(payload, path)
    print(f"  Checkpoint saved → {path}  (epoch {epoch})")


def load_checkpoint(
    model:     "nn.Module",
    optimizer: "torch.optim.Optimizer | None",
    path:      str,
) -> dict:
    """
    Load a saved checkpoint into the model (and optionally optimizer).

    Args:
        model:     The ResNet-50 model to restore weights into.
        optimizer: Optimizer to restore state into, or None to skip.
        path:      Path to the .pt checkpoint file.

    Returns:
        The full checkpoint dict (contains 'epoch' and any extra metadata).

    Raises:
        ImportError:       If torch is not installed.
        FileNotFoundError: If the checkpoint file does not exist.
    """
    _require_torch("load_checkpoint")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint  = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    epoch = checkpoint.get("epoch", "unknown")
    print(f"  Checkpoint loaded ← {path}  (epoch {epoch})")
    return checkpoint


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def count_trainable_params(model: "nn.Module") -> int:
    """
    Return the number of parameters with requires_grad=True.

    Args:
        model: Any nn.Module.

    Returns:
        Integer count of trainable parameters.

    Raises:
        ImportError: If torch is not installed.
    """
    _require_torch("count_trainable_params")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: "nn.Module") -> int:
    """
    Return the total number of parameters (trainable + frozen).

    Args:
        model: Any nn.Module.

    Returns:
        Integer count of all parameters.

    Raises:
        ImportError: If torch is not installed.
    """
    _require_torch("count_total_params")
    return sum(p.numel() for p in model.parameters())


def describe_model(model: "nn.Module") -> None:
    """
    Print a human-readable parameter summary of the model grouped by
    top-level module (conv1, bn1, layer1–4, fc).

    Args:
        model: ResNet-50 model returned by build_model().

    Raises:
        ImportError: If torch is not installed.
    """
    _require_torch("describe_model")

    print(f"\nModel: ResNet-50 (modified for {NUM_CLASSES}-class brain tumor classification)")
    print("-" * 58)
    total, trainable = 0, 0
    for name, module in model.named_children():
        n_total     = sum(p.numel() for p in module.parameters())
        n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        status      = "trainable" if n_trainable > 0 else "FROZEN"
        print(f"  {name:<10s}  {n_total:>10,} params  [{status}]")
        total     += n_total
        trainable += n_trainable
    print("-" * 58)
    print(f"  {'TOTAL':<10s}  {total:>10,} params")
    print(f"  {'TRAINABLE':<10s}  {trainable:>10,} params  ({100*trainable/total:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_torch(fn_name: str) -> None:
    """Raise ImportError with install instructions if torch is missing."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            f"'{fn_name}' requires PyTorch and torchvision.\n"
            "Install them in Google Colab with:\n"
            "  !pip install torch torchvision\n"
            "or locally:\n"
            "  pip install torch torchvision"
        )


def _set_requires_grad(model: "nn.Module", value: bool) -> None:
    """Set requires_grad for all parameters in the model."""
    for param in model.parameters():
        param.requires_grad = value


# ---------------------------------------------------------------------------
# CLI smoke-test (runs without training; prints model summary + phase configs)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 58)
    print("  TASK 11: RESNET-50 TRANSFER LEARNING ARCHITECTURE")
    print("=" * 58)

    if not TORCH_AVAILABLE:
        print("\n  PyTorch not found — run this script in Google Colab.")
        print("  The following configuration will be used:\n")
        print(f"  Architecture  : ResNet-50 (torchvision.models.resnet50)")
        print(f"  Pretrained    : ImageNet (IMAGENET1K_V1)")
        print(f"  Classifier    : Dropout({DROPOUT_RATE}) → Linear(2048 → {NUM_CLASSES})")
        print(f"  Classes       : {CLASSES}")
        print(f"  Label mapping : {CLASS_TO_IDX}")
        print(f"\n  Phase 1  — Epochs: {PHASE1_EPOCHS},  LR: {PHASE1_LR},  frozen: all backbone")
        print(f"  Phase 2  — Epochs: {PHASE2_EPOCHS}, LR: {PHASE2_LR}, frozen: conv1+bn1+layer1")
        print(f"\n  Optimizer : Adam(weight_decay={PHASE1_WEIGHT_DECAY})")
        print(f"  Scheduler : ReduceLROnPlateau(mode='max', patience={SCHEDULER_PATIENCE}, "
              f"factor={SCHEDULER_FACTOR})")
        sys.exit(0)

    # Torch available — build and inspect the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    print("\nBuilding model (pretrained=True)...")
    model = build_model(pretrained=True)
    model.to(device)

    print("\n--- Phase 1 configuration (freeze_backbone) ---")
    freeze_backbone(model)
    describe_model(model)
    opt1  = get_optimizer(model, phase=1)
    sched = get_scheduler(opt1)
    print(f"  Optimizer : {opt1.__class__.__name__}  LR={PHASE1_LR}  "
          f"weight_decay={PHASE1_WEIGHT_DECAY}")
    print(f"  Scheduler : ReduceLROnPlateau  patience={SCHEDULER_PATIENCE}  "
          f"factor={SCHEDULER_FACTOR}")

    print("\n--- Phase 2 configuration (unfreeze_for_finetuning) ---")
    unfreeze_for_finetuning(model)
    describe_model(model)
    opt2 = get_optimizer(model, phase=2)
    print(f"  Optimizer : {opt2.__class__.__name__}  LR={PHASE2_LR}  "
          f"weight_decay={PHASE2_WEIGHT_DECAY}")

    # Forward pass sanity check
    print("Forward pass sanity check (batch=2, 3×224×224)...")
    dummy = torch.zeros(2, 3, IMG_SIZE, IMG_SIZE, device=device)
    with torch.no_grad():
        logits = model(dummy)
    print(f"  Input  shape : {tuple(dummy.shape)}")
    print(f"  Output shape : {tuple(logits.shape)}  (expected [2, {NUM_CLASSES}])")
    assert logits.shape == (2, NUM_CLASSES), (
        f"Output shape mismatch: expected (2, {NUM_CLASSES}), got {tuple(logits.shape)}"
    )
    print("  Shape assertion passed.")

    # Checkpoint round-trip test
    print("\nCheckpoint round-trip test...")
    ckpt_path = os.path.join(CHECKPOINT_DIR, "_smoke_test.pt")
    save_checkpoint(model, opt2, epoch=0, path=ckpt_path, extra={"phase": 2, "val_acc": 0.0})
    _ = load_checkpoint(model, opt2, ckpt_path)
    os.remove(ckpt_path)
    print("  Checkpoint write/read/delete passed.")

    print("\nDone.")
