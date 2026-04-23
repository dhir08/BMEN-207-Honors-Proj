# Brain Tumor MRI Classification
### BMEN 207 Honors Project — Bhargav Ashwin & Dhir Parekh

A deep learning pipeline for **4-class brain tumor MRI classification** using ResNet-50 transfer learning with Grad-CAM explainability.

**Classes:** Glioma · Meningioma · No Tumor · Pituitary

---

## Pipeline Overview

| Task | Script | Description |
|------|--------|-------------|
| 3 | `download_dataset.py` | Kaggle dataset download |
| 4 | `audit_dataset.py` | Class distribution & image audit |
| 5 | `preprocess.py` | Resize, RGB convert, normalize → PNG |
| 6 | `augment.py` | Offline augmentation (Albumentations) |
| 7 | `split_dataset.py` | Stratified 1,400/class train + test split |
| 8 | `verify_batches.py` | Verify 4×100/class test batches |
| 9 | `dataloader.py` | PyTorch DataLoader pipeline |
| 10 | `model_selection.py` | ResNet-50 vs EfficientNet-B0 comparison |
| 11 | `model.py` | ResNet-50 architecture + checkpointing |
| 12 | `train_config.py` | Centralized hyperparameter config |
| 13 | `train.py` | Two-phase training loop (Colab GPU) |
| 14 | `evaluate.py` | Evaluate on all 4 test batches |
| 15 | `metrics.py` | Confusion matrix, precision, recall, F1 |
| 16 | `gradcam.py` | Grad-CAM explainability heatmaps |
| 17 | `validate_external.py` | External validation on TCIA samples |
| 18 | `visualize.py` | Prediction grids + training curves |
| 19 | `benchmark.py` | Compare vs Kaggle/literature baselines |

---

## Quick Start (Google Colab)

```python
# 1. Mount Drive and clone repo
from google.colab import drive
drive.mount('/content/drive')

# 2. Install dependencies
!pip install torch torchvision albumentations grad-cam matplotlib openpyxl

# 3. Run pipeline
!python scripts/preprocess.py
!python scripts/split_dataset.py
!python scripts/train.py
!python scripts/evaluate.py
!python scripts/metrics.py
!python scripts/gradcam.py
!python scripts/visualize.py
!python scripts/benchmark.py
```

## Training Strategy

**Phase 1 — Feature Extraction** (5 epochs, LR=1e-3)
- Backbone frozen; only the 4-class classifier head is trained
- Establishes a stable mapping from ImageNet features to tumor classes

**Phase 2 — Fine-Tuning** (up to 20 epochs, LR=1e-5, early stopping patience=5)
- Unfreezes ResNet layers 2–4 for MRI-specific adaptation
- Low LR prevents catastrophic forgetting of ImageNet features

## Results

> Training on Google Colab in progress. Results tables will be updated upon completion.

| Model | Accuracy | Source |
|-------|----------|--------|
| **Ours (ResNet-50)** | Pending | This project |
| ResNet-50 (Lit.) | 99.68% | Scientific Reports 2023 |
| EfficientNet-B3 | 99.23% | GitHub 2022 |
| EfficientNet-B0 | 98.97% | Springer Nature 2022 |

## Requirements

```
torch
torchvision
albumentations
Pillow
numpy
matplotlib
grad-cam
openpyxl
pydicom (optional, for TCIA DICOM files)
```

Install: `pip install -r requirements.txt`
