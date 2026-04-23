"""
Task 9: PyTorch DataLoader Pipeline
- Custom BrainTumorDataset class (torch.utils.data.Dataset)
- Training transforms: resize, normalize, random flip/rotation for extra augmentation
- Test transforms: resize + normalize only (no stochasticity in evaluation)
- get_train_loader()     → DataLoader for the full training set
- get_test_batch_loader() → DataLoader for a single numbered test batch

Class label encoding (consistent throughout the project):
    glioma      → 0
    meningioma  → 1
    no_tumor    → 2
    pituitary   → 3

IMPORTANT: This script requires PyTorch and torchvision.
  Run in Google Colab or any environment with:
    pip install torch torchvision

Usage (in a Colab training notebook):
    from scripts.dataloader import get_train_loader, get_test_batch_loader

    train_loader = get_train_loader("data/processed/train", batch_size=32)
    test_loader  = get_test_batch_loader("data/processed/test", batch_num=1, batch_size=32)

    for images, labels in train_loader:
        # images: (B, 3, 224, 224) float32 tensor, normalized
        # labels: (B,) int64 tensor, values 0-3
        ...
"""

import os
import sys

# PIL is always available (part of requirements.txt Pillow install)
from PIL import Image

# ---------------------------------------------------------------------------
# Guard: torch/torchvision are Colab-only (no disk space in dev VM)
# ---------------------------------------------------------------------------
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

TRAIN_DIR = "data/processed/train"
TEST_DIR  = "data/processed/test"
CLASSES   = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Map class name → integer label (used consistently by model and metrics scripts)
CLASS_TO_IDX: dict[str, int] = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: cls for cls, i in CLASS_TO_IDX.items()}

# ImageNet statistics (used for normalization — appropriate for ResNet/EfficientNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_SIZE = 224   # must match preprocess.py output size


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def build_train_transforms() -> "T.Compose":
    """
    Lightweight augmentation on top of the already-augmented training data.
    Kept mild — heavy augmentation was already applied in augment.py.
    Normalization uses ImageNet stats for compatibility with pretrained ResNet/EfficientNet.
    """
    _require_torch("build_train_transforms")
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),                                          # [0,255] uint8 → [0,1] float32
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_test_transforms() -> "T.Compose":
    """
    Deterministic transforms for test set evaluation.
    No random augmentation — only resize + normalize for fair, reproducible evaluation.
    """
    _require_torch("build_test_transforms")
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class BrainTumorDataset:
    """
    Custom PyTorch Dataset for the brain tumor MRI classification project.

    Expects directory layout:
        root_dir/
            glioma/       *.png
            meningioma/   *.png
            no_tumor/     *.png
            pituitary/    *.png

    Each PNG is loaded as RGB and passed through the provided transform.

    Args:
        root_dir:  Path to the class-subfolder directory.
        transform: torchvision transform to apply to each image.
        classes:   List of class names in label order (default: CLASSES).

    Raises:
        FileNotFoundError: If root_dir does not exist.
        ValueError:        If no images are found under root_dir.
    """

    def __init__(self, root_dir: str, transform=None, classes: list[str] = None):
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

        self.root_dir  = root_dir
        self.transform = transform
        self.classes   = classes or CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Build flat list of (filepath, label) pairs — one os.listdir per class dir
        self.samples: list[tuple[str, int]] = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            label = self.class_to_idx[cls]
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(".png"):
                    self.samples.append((os.path.join(cls_dir, fname), label))

        if not self.samples:
            raise ValueError(
                f"No PNG images found under '{root_dir}'. "
                "Run preprocess.py (and split_dataset.py for test batches) first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Returns:
            image : float32 tensor (3, H, W) after transform, or PIL Image if no transform.
            label : int64 tensor scalar (0-3).
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        path, label = self.samples[idx]

        # Load image — handle corrupt files gracefully
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise IOError(f"Cannot open image '{path}': {e}") from e

        if self.transform is not None:
            img = self.transform(img)

        if TORCH_AVAILABLE:
            return img, torch.tensor(label, dtype=torch.long)
        else:
            # Fallback for environments without torch (testing only)
            return img, label

    def class_counts(self) -> dict[str, int]:
        """Return number of samples per class in this dataset."""
        counts: dict[str, int] = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            counts[IDX_TO_CLASS[label]] += 1
        return counts


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def get_train_loader(
    train_dir:  str = TRAIN_DIR,
    batch_size: int = 32,
    seed:       int = 42,
    num_workers: int = 2,
) -> "DataLoader":
    """
    Build a DataLoader for the full training set (all 4 class subfolders).

    Args:
        train_dir:   Path to the training directory (contains class subfolders).
        batch_size:  Images per mini-batch.
        seed:        Random seed for reproducible shuffling.
        num_workers: Parallel data loading workers (set 0 for Colab safety).

    Returns:
        Shuffled DataLoader with training transforms.

    Raises:
        ImportError:       If torch is not installed.
        FileNotFoundError: If train_dir does not exist.
    """
    _require_torch("get_train_loader")

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataset = BrainTumorDataset(train_dir, transform=build_train_transforms())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )


def get_test_batch_loader(
    test_dir:    str = TEST_DIR,
    batch_num:   int = 1,
    batch_size:  int = 32,
    num_workers: int = 2,
) -> "DataLoader":
    """
    Build a DataLoader for a single numbered test batch.

    The test directory is expected to contain batch1/, batch2/, ... subdirs,
    each with the 4 class subfolders inside.

    Args:
        test_dir:    Root test directory (contains batchN subdirs).
        batch_num:   Which batch to load (1-indexed, e.g., 1 → batch1/).
        batch_size:  Images per mini-batch.
        num_workers: Parallel data loading workers.

    Returns:
        Non-shuffled DataLoader with test transforms.

    Raises:
        ImportError:       If torch is not installed.
        FileNotFoundError: If the batch directory does not exist.
        ValueError:        If batch_num < 1.
    """
    if batch_num < 1:
        raise ValueError(f"batch_num must be >= 1, got {batch_num}")

    _require_torch("get_test_batch_loader")

    batch_dir = os.path.join(test_dir, f"batch{batch_num}")
    dataset   = BrainTumorDataset(batch_dir, transform=build_test_transforms())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,           # deterministic order for reproducible evaluation
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


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


def describe_dataset(root_dir: str, label: str = "") -> None:
    """
    Print a human-readable summary of a dataset directory.
    Useful for sanity-checking before training.
    Runs without torch.
    """
    tag = f" ({label})" if label else ""
    print(f"\nDataset summary{tag}: {root_dir}")
    total = 0
    for cls in CLASSES:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"  {cls:<15s}: [MISSING]")
            continue
        n = len([f for f in os.listdir(cls_dir) if f.endswith(".png")])
        total += n
        print(f"  {cls:<15s}: {n:>6} images  (label={CLASS_TO_IDX[cls]})")
    print(f"  {'TOTAL':<15s}: {total:>6} images")


# ---------------------------------------------------------------------------
# CLI smoke-test (runs without training)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  TASK 9: DATALOADER PIPELINE")
    print("=" * 55)

    if not TORCH_AVAILABLE:
        print("\n⚠  PyTorch not found — run this script in Google Colab.")
        print("   Displaying dataset directory summary instead.\n")
        for label, path in [("Train", TRAIN_DIR), ("Test batch1", os.path.join(TEST_DIR, "batch1"))]:
            describe_dataset(path, label)
        sys.exit(0)

    # Torch available — show loader stats
    describe_dataset(TRAIN_DIR, "Train")
    for b in range(1, 5):
        describe_dataset(os.path.join(TEST_DIR, f"batch{b}"), f"Test batch{b}")

    print("\nBuilding train DataLoader (batch_size=32)...")
    try:
        loader = get_train_loader(batch_size=32, num_workers=0)
        imgs, lbls = next(iter(loader))
        print(f"  Batch shape : {tuple(imgs.shape)}  (B, C, H, W)")
        print(f"  Labels      : {lbls.tolist()[:8]}...")
        print(f"  Total batches: {len(loader)}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\nDone.")
