# SoCalGuessr 

A deep learning image classifier that predicts which Southern California city a street-level photo was taken in — trained to outperform humans at a task most locals can't ace.

## Overview

Given a street-view image, SoCalGuessr predicts one of 6 Southern California cities:

`Anaheim` · `Bakersfield` · `Los Angeles` · `Riverside` · `San Diego` · `SLO`

**Human baseline accuracy: 40%** — the model achieves **~91.07%** validation accuracy.

---

## Model Architecture

Built on a pretrained **ResNet18** (ImageNet) with a custom 6-class classifier head replacing the original fully connected layer.

```
ResNet18 Backbone (frozen in Phase 1, unfrozen in Phase 2)
    └── Linear(512 → 256)
    └── ReLU
    └── Dropout(p=0.3)
    └── Linear(256 → 6)
```

| | Phase 1 | Phase 2 |
|---|---|---|
| Backbone | Frozen | Unfrozen |
| Trainable params | 132,870 | ~11.2M |
| Learning rate | 0.001 | 0.0001 |
| Epochs | 10 | 10 |
| Optimizer | Adam | Adam |
| Loss | CrossEntropyLoss | CrossEntropyLoss |

---

## Training Procedure

Training used a **two-stage transfer learning** approach:

**Stage 1 — Head Training:** The ResNet18 backbone is frozen. Only the custom classifier head is trained, allowing the model to learn city-specific patterns on top of general ImageNet features.

**Stage 2 — Full Fine-Tuning:** All layers are unfrozen and trained end-to-end at a reduced learning rate. The sharp loss drop visible in the training curve reflects the increased model capacity unlocked at this stage.

Best model weights are saved via checkpoint whenever validation accuracy improves.

### Training Curve
> Loss drops sharply after epoch 10 (red dashed line) when fine-tuning begins — full backbone capacity drives rapid convergence.

---

## Results

| Metric | Value |
|---|---|
| Human baseline | 40% |
| Best validation accuracy | ~91.07% |
| Total training time | 12.16 minutes |
| Batch size | 64 |

---

## Data

Images are organized by city prefix in filenames (`CityName-*.jpg`). An 80/20 stratified train-validation split ensures balanced class representation across all 6 cities.

**Augmentations applied during training:**
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness, contrast, saturation)
- Resize to 224×224
- ImageNet normalization

---

## Usage

### Train
```bash
python model.py
```
Trains in two stages and saves best weights to `weights.pt`.

### Predict
```python
from predict import predict

results = predict("path/to/image/folder")
# Returns: {'image.jpg': 'San_Diego', ...}
```

---

## Tech Stack

- Python
- PyTorch & torchvision
- ResNet18 (pretrained)
- scikit-learn (train/val split)
- Matplotlib

---

## Author

**Jiahe Jason Qin** — DSC 140B Final Project, UC San Diego
