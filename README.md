# AA-ATS: Anatomically-Aware Adaptive Token Sampler

This repository contains the implementation of our thesis project:

**Anatomically-Aware Adaptive Token Sampler (AA-ATS)**

AA-ATS extends Masked Autoencoders (MAE) by incorporating anatomical information from brain MRI into the masking process used during self-supervised pretraining.

---

# Repository Structure

```text
.
├── data.py
├── anatomy.py
├── models.py
├── run_pretrain.py
├── check_progress.py
├── plot_results.py
├── diagnose_mse.py
│
├── cache/
│   └── preprocessed BraTS data
│
├── results/
│   ├── reconstruction.csv
│   ├── checkpoints/
│   ├── logs/
│   └── figures/
```

---

# What Each File Does

## data.py

Handles loading and preprocessing of BraTS data.

### Responsibilities

- Load BraTS patients
- Create train/validation folds
- Extract axial slices
- Apply augmentation
- Generate tensors for training

### Used By

```text
run_pretrain.py
diagnose_mse.py
```

---

## anatomy.py

Computes anatomical features used by AA-ATS.

### Available Methods

```text
Sobel
Canny
Hybrid
```

### Outputs

Generates four features per patch:

```text
mean
max
std
density
```

These become:

```text
Aᵢ ∈ ℝ⁴
```

for each image patch.

---

## models.py

Contains the main model definitions.

### Patch Embedding

Converts MRI slices into transformer tokens.

```text
224×224 image
↓
196 patches
↓
Transformer tokens
```

### Token Sampler

Implements multiple masking strategies:

```text
random
adaptive
hard_anat
anatomical
```

### AA-ATS

The proposed method that combines:

```text
Patch Embeddings
+
Anatomical Features
```

through a learnable fusion module.

### Transformer Encoder

Uses a ViT-Base backbone.

### Decoder

Used for self-supervised reconstruction.

---

# Running AA-ATS

This runs the proposed thesis model.

```bash
python run_pretrain.py \
  --cache ./cache \
  --mask_mode anatomical \
  --mask_ratio 0.75 \
  --epochs 20
```

### Example Configuration

```text
Model: AA-ATS
Mask Ratio: 75%
Epochs: 20
Fusion: Learnable
Anatomical Map: Hybrid
```

---

# Running Other Masking Strategies

## Random MAE

```bash
python run_pretrain.py \
  --cache ./cache \
  --mask_mode random \
  --mask_ratio 0.75 \
  --epochs 20
```

## Adaptive Sampler

```bash
python run_pretrain.py \
  --cache ./cache \
  --mask_mode adaptive \
  --mask_ratio 0.75 \
  --epochs 20
```

## Rule-Based Anatomical Masking

```bash
python run_pretrain.py \
  --cache ./cache \
  --mask_mode hard_anat \
  --mask_ratio 0.75 \
  --epochs 20
```

---

# Running All Configurations

Run the complete experimental sweep:

```bash
python run_pretrain.py \
  --cache ./cache \
  --sweep \
  --epochs 20
```

This automatically runs:

```text
4 masking strategies
×
3 masking ratios
=
12 experiments
```

---

# Outputs

## reconstruction.csv

Located at:

```text
results/reconstruction.csv
```

Stores:

```text
model
mask_ratio
epochs
final_val_mse
best_val_mse
minutes
```

Each completed experiment appends one row.

---

## checkpoints

Located at:

```text
results/checkpoints/
```

Stores trained model weights:

```text
*.pt
```

---

## logs

Located at:

```text
results/logs/
```

Stores epoch-by-epoch training history:

```text
*.json
```

---

# Checking Progress

If a sweep is still running:

```bash
python check_progress.py \
  --results ./results
```

Displays:

- Completed runs
- Missing runs
- Duplicate runs
- Currently available comparisons

---

# Generating Tables and Figures

```bash
python plot_results.py \
  --results ./results
```

Produces:

```text
table_3_1_reconstruction.csv
figures/
```

---

# Useful Commands

## Train AA-ATS

```bash
python run_pretrain.py \
  --cache ./cache \
  --mask_mode anatomical \
  --mask_ratio 0.75 \
  --epochs 20
```

## Train Random MAE

```bash
python run_pretrain.py \
  --cache ./cache \
  --mask_mode random \
  --mask_ratio 0.75 \
  --epochs 20
```

## Continue Sweep

```bash
python run_pretrain.py \
  --cache ./cache \
  --sweep
```

## Generate Figures

```bash
python plot_results.py \
  --results ./results
```

---

# Modifying AA-ATS

If you want to modify the proposed model:

## Change Anatomical Features

Edit:

```text
anatomy.py
```

## Change Fusion Logic

Edit:

```text
models.py
```

Look for:

```python
# Direct Fusion
# Learnable Fusion
```

## Change Sampling Strategy

Edit:

```python
sample_visible()
```

inside:

```text
models.py
```

## Change Training Parameters

Edit:

```text
run_pretrain.py
```

or pass command-line arguments:

```bash
--epochs
--mask_ratio
--batch_size
--lr
```

---

# Typical Workflow

```text
Prepare BraTS Data
↓
Run Training
↓
Generate Results
↓
Inspect Figures
↓
Modify Model
↓
Train Again
```

---

# Quick Start

```bash
python run_pretrain.py \
  --cache ./cache \
  --mask_mode anatomical \
  --mask_ratio 0.75 \
  --epochs 20

python plot_results.py \
  --results ./results
```

This is the fastest way to train and evaluate AA-ATS.