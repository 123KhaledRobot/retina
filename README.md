# Diabetic Retinopathy Detection

Automated severity grading of diabetic retinopathy from retinal fundus images using deep learning (PyTorch).

**Dataset:** [Kaggle — Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)

---

## Problem Statement

Diabetic retinopathy is a diabetes complication that damages blood vessels in the retina and is a leading cause of preventable blindness. Early detection is critical — a missed diagnosis (false negative) can result in irreversible vision loss.

This project builds a CNN-based classifier that assigns a severity score to retinal fundus images on the following scale:

| Score | Severity |
|-------|----------|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

---

## Dataset Overview

- High-resolution retinal fundus images (`.jpeg`)
- Left and right eye provided per patient (e.g. `1_left.jpeg`, `1_right.jpeg`)
- Images vary in resolution (2592×1944 to 4928×3264) due to different camera models
- Labels provided in `trainLabels.csv`
- Real-world noise: underexposed/overexposed images, artifacts, blur, and inverted orientations

---

## Lesion Types

Two clinically distinct lesion types are targeted:

**Hemorrhages** — dark red spots caused by ruptures in retinal blood vessels.
**Exudates** — bright yellow/white deposits from fatty leakage through damaged vessel walls.

These two lesion types require different preprocessing strategies (see below).

---

## Preprocessing Pipeline

Each raw image goes through the following steps before being fed to the model:

### Step 1 — Crop to Square
Remove black borders and crop to a square aspect ratio to preserve the circular retinal region without distortion.

### Step 2 — CLAHE (Contrast Limited Adaptive Histogram Equalization)
Applied at full resolution to correct underexposed/overexposed images. CLAHE enhances contrast locally (in tiles) rather than globally, which is critical for retinal images where the optic disc region is significantly brighter than the periphery.

### Step 3 — Color Channel Engineering
Instead of feeding raw RGB to the model, three semantically meaningful channels are constructed:

| Channel Slot | Content | Purpose |
|---|---|---|
| R | Warm (red+orange+yellow filtered) | Enhances bright exudates |
| G | Green channel only | Enhances dark hemorrhages (red appears black under green filter) |
| B | Original grayscale | Preserves overall structural information |

This replaces the standard RGB input with a clinically-informed 3-channel representation that ResNet can directly consume.

### Step 4 — Resize to 224×224
Resize to match ResNet's expected input dimensions.

### Step 5 — Augmentation
Applied on the resized (small) image for computational efficiency:
- Horizontal and vertical flips
- Random rotations

> **Note:** Aggressive cropping is avoided as clinically important features (microaneurysms, vessel patterns) may be located near image edges.

### Step 6 — Normalization
Pixel values are normalized using ImageNet statistics (required for pretrained ResNet):
```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

## Class Imbalance

The dataset is heavily skewed toward class 0 (healthy). Without correction, a naive model learns to always predict "healthy" and achieves deceptively high accuracy.

**Mitigation strategies:**
- **Class-weighted loss:** Rare classes receive higher penalty weights during training.
```python
nn.CrossEntropyLoss(weight=class_weights)
# weight_i = total_samples / (num_classes × count_i)
```
- **Augmentation-based oversampling:** Synthetic images are generated from minority classes via flips and rotations.

---

## Model Architecture

**Chosen Model: ResNet50 with Transfer Learning**

| Model | Reason for decision |
|---|---|
| YOLO | Rejected — designed for object detection (bounding boxes), not image classification |
| VGG16 | Rejected — ~138M parameters, high training cost, no skip connections |
| ResNet50 | Selected — ~25M parameters, skip connections solve vanishing gradient, strong ImageNet pretraining |

### Why ResNet?

Deep networks suffer from the **vanishing gradient problem** — the learning signal weakens as it travels backwards through many layers. ResNet solves this with **skip connections** that provide a direct gradient highway:

```
Input → [Conv Layers] → Output
  ↘_______________________↗
         (skip added)
```

### Transfer Learning Strategy

ResNet50 is pretrained on ImageNet. Early layers learn universal features (edges, textures, gradients) that transfer well to retinal images.

**Fine-tuning approach:**
1. Replace the final classification head: `nn.Linear(2048, 5)`
2. Freeze all backbone layers initially (preserve pretrained weights)
3. Train the head until it converges
4. Optionally unfreeze deeper layers with a very small learning rate for fine-tuning

---

## Evaluation Metrics

Standard accuracy is not appropriate here due to class imbalance and the ordinal nature of the labels. The following metrics are used:

| Metric | Why it matters |
|---|---|
| **Quadratic Weighted Kappa (QWK)** | Official Kaggle competition metric. Penalizes predictions proportionally to how far they are from the true severity — a prediction of 0 for a class 4 case is penalized more than a prediction of 3. |
| **Per-class Recall** | Measures how many actual positive cases are correctly detected. Critical in medical settings — false negatives (missed diagnoses) are more dangerous than false positives. |
| **F1 Score (macro)** | Harmonic mean of Precision and Recall across all classes. Accounts for imbalance by treating each class equally. |
| **Confusion Matrix** | Visualizes systematic misclassification patterns across severity levels. |

> In medical diagnosis, **Recall is prioritized over Precision** — missing a severe retinopathy case risks preventable blindness.

---

## Environment Setup

```bash
# Create conda environment
conda create -n retina python=3.10 -y
conda activate retina

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install opencv-python matplotlib pandas numpy Pillow
```

**Verified:** PyTorch 2.5.1 + CUDA 12.1 on NVIDIA GeForce MX150 (local testing) with training on rented GPU.

---

## Project Structure

```
retina/
├── data/
│   └── sample/              # Small preview set for pipeline testing
├── eda.py                   # Exploratory data analysis
├── dataset.py               # PyTorch Dataset class (in progress)
├── preprocessing.py         # CLAHE + color channel engineering (in progress)
├── model.py                 # ResNet50 with custom head (in progress)
├── train.py                 # Training loop (in progress)
└── README.md
```

---

## Development Approach

> **Principle:** Build and validate the entire pipeline on `sample.zip` (10 images) before scaling to the full 88GB dataset on a rented GPU. Fail fast, fail cheap.

**EDA findings on sample set:**
- Image resolutions range from 2592×1944 to 4928×3264 — all must be resized
- Pixel value ranges vary (one image had max=145 instead of 255) — confirming underexposure in real data
- Left/right eye pairs from the same patient share identical dimensions (same camera per session)

---

## References

- [Kaggle Diabetic Retinopathy Detection Competition](https://www.kaggle.com/c/diabetic-retinopathy-detection)
- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- CLAHE: Zuiderveld, "Contrast Limited Adaptive Histogram Equalization", 1994
