# FallDetectionNet — Pose-Based Fall Detection for Journal Publication

Spatial-temporal fall detection using YOLOv8-Pose skeleton extraction, dilated
temporal convolutions, and joint attention.  The pipeline is designed for
Q2-journal-quality results with reproducible training, proper cross-validation,
and full MLflow experiment tracking.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Pipeline Scripts](#5-pipeline-scripts)
   - [data_pipeline.py](#51-data_pipelinepy)
   - [02_data_validation.py](#52-02_data_validationpy)
   - [03_train.py](#53-03_trainpy)
6. [Model Architecture](#6-model-architecture)
7. [Training Strategy](#7-training-strategy)
8. [Configuration Reference](#8-configuration-reference)
9. [Outputs and Artifacts](#9-outputs-and-artifacts)
10. [Dataset Statistics](#10-dataset-statistics)
11. [Reproducing Results](#11-reproducing-results)
12. [Known Issues and Fixes](#12-known-issues-and-fixes)

---

## 1. Project Overview

This project implements a two-stage fall detection system:

**Stage A — NTU RGB+D 120 Pretraining**
Multi-class action classification across 10 selected NTU action classes
(sitting down, standing up, fall, headache, chest pain, etc.).  The model
learns general human motion representations before seeing any fall-specific
data.

**Stage B — Binary Fine-tuning**
The pretrained backbone is fine-tuned on URFD and Le2i with two variants:
- **Variant A** — frozen backbone, head-only training (fast, good baseline)
- **Variant B** — full fine-tuning with differential learning rates and
  cosine annealing with warmup (best final performance)

**Key design decisions for journal quality:**
- Sequence-level 5-fold cross-validation (never split windows randomly)
- Focal loss with per-fold dynamic alpha from inverse class frequency
- Youden-index optimal threshold calibration per fold
- Cross-fold threshold averaging for unbiased test evaluation
- Full metrics suite: AUC-ROC, AUC-PR, sensitivity, specificity, F1, FN, FP
- MLflow experiment tracking for every run

---

## 2. Repository Structure

```
research/
├── data_pipeline.py          # Stage 0: raw data → processed .npz
├── 02_data_validation.py     # Stage 1: quality audit before training
├── 03_train.py               # Stage 2: model definition + training
├── research_config.json      # All hyperparameters and paths
│
├── DATASET/
│   ├── URFD/
│   │   ├── fall/             # fall-NN-cam0-rgb/*.png sequences
│   │   └── adl/              # adl-NN-cam0-rgb/*.png sequences
│   ├── LE2I/
│   │   ├── Coffee_room_01/
│   │   ├── Coffee_room_02/
│   │   ├── Home_01/
│   │   └── Home_02/
│   └── NTU/
│       └── *.skeleton        # S###C###P###R###A###.skeleton files
│
├── processed/
│   ├── urfd/sequences.npz    # (1041, 30, 75) X + y
│   ├── le2i/sequences.npz    # (3658, 30, 75) X + y
│   ├── ntu/sequences.npz     # (535535, 30, 75) X + y  [binary, pipeline only]
│   └── ntu/stage_a_cache.npz # (N, 7, 30, 17) multi-class cache [built on first run]
│
├── outputs/
│   ├── checkpoints/          # .pt files: best per fold + final epoch
│   ├── figures/              # ROC, PR, confusion matrix PNGs
│   └── logs/                 # timestamped .log files
│
├── figures/                  # Validation plots from 02_data_validation.py
│   ├── 01_skeleton_quality.png
│   ├── 02_class_balance.png
│   ├── 03_feature_distributions.png
│   └── 04_temporal_coverage.png
│
└── venv/                     # Python virtual environment
```

---

## 3. Environment Setup

### Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 2000 Ada Generation |
| CUDA | 12.1 |
| Driver | tested with 525+ |

### Creating the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Installing dependencies

```bash
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

pip install numpy==2.2.6 \
    opencv-python==4.13.0.92 \
    ultralytics==8.4.35 \
    scikit-learn==1.7.2 \
    mlflow==2.13.0 \
    matplotlib==3.10.8 \
    seaborn==0.13.2 \
    scipy==1.15.3 \
    tqdm==4.67.3 \
    pyarrow==23.0.1
```

> **Important:** `pyarrow>=16` is required for NumPy 2.x compatibility.
> `pyarrow<16` (e.g. 15.0.2) will crash at import time with
> `_ARRAY_API not found`.  See [Known Issues](#12-known-issues-and-fixes).

### Activating the environment

```bash
source venv/bin/activate
```

All commands below assume the venv is active.

---

## 4. Dataset Preparation

### URFD (UR Fall Detection Dataset)

```
DATASET/URFD/
    fall/
        fall-01-cam0-rgb/   *.png
        fall-02-cam0-rgb/   *.png
        ...
    adl/
        adl-01-cam0-rgb/    *.png
        adl-02-cam0-rgb/    *.png
        ...
```

- Image sequences (PNG frames), one directory per sequence
- Labels: `fall/` → 1, `adl/` → 0

### Le2i (Laboratory of Electronic Imaging)

```
DATASET/LE2I/
    Coffee_room_01/Coffee_room_01/
        Videos/          video (N).avi
        Annotation_files/video (N).txt
    Coffee_room_02/Coffee_room_02/
        Videos/          video (N).avi
        Annotations_files/video (N).txt    ← note: different spelling
    Home_01/Home_01/
        Videos/          video (N).avi
        Annotation_files/video (N).txt
    Home_02/Home_02/
        Videos/          video (N).avi
        Annotation_files/video (N).txt
```

Annotation format (per `.txt`):
```
<fall_start_frame>    ← 1-indexed
<fall_end_frame>      ← 1-indexed
<frame_id>,<flag>,<x1>,<y1>,<x2>,<y2>
...
```

> **Note:** Some Le2i AVI files have a corrupted MP3 audio header that causes
> a heap-corruption crash in OpenCV/FFmpeg.  The pipeline uses an `ffmpeg`
> subprocess with `-an` to bypass this — see `data_pipeline.py:_iter_frames_ffmpeg`.

### NTU RGB+D 120

```
DATASET/NTU/
    S001C001P001R001A001.skeleton
    S001C001P001R001A043.skeleton
    ...
```

- One `.skeleton` file per action clip
- Action ID encoded as `A###` in filename
- `A043` = fall down (used in binary pipeline)
- For Stage A multi-class pretraining, selected action IDs are defined in
  `research_config.json` → `ntu_selected_actions`

---

## 5. Pipeline Scripts

### 5.1 data_pipeline.py

Processes raw datasets into windowed pose-feature sequences.

**Run:**
```bash
python data_pipeline.py
```

**Optional arguments:**
```bash
python data_pipeline.py \
    --datasets urfd le2i ntu \
    --dataset-root DATASET \
    --output-root processed \
    --pose-model yolov8n-pose.pt \
    --window-size 30 \
    --stride 10 \
    --pose-conf 0.3 \
    --device cuda
```

**What it does:**
1. **URFD** — reads PNG frames, runs YOLOv8n-Pose, extracts 75-dim feature
   vectors per frame, slides a 30-frame window with stride 10
2. **Le2i** — decodes AVI videos via ffmpeg subprocess (audio disabled),
   runs YOLOv8n-Pose, applies frame-level fall annotations
3. **NTU** — parses `.skeleton` files directly (no pose model needed),
   maps 25 NTU joints → 17 COCO joints

**Feature layout (75 dims per frame):**
```
indices  0-33  : 17 × (x, y) normalised keypoint coordinates
indices 34-40  : 7 global derived features
    [34] aspect_ratio       — keypoint bounding box H/W
    [35] head_to_hip        — vertical nose-to-hip distance / kp_h
    [36] hip_angle / 180    — L_shoulder–mid_hip–L_knee angle
    [37] left_knee  / 180
    [38] right_knee / 180
    [39] torso_vertical     — cos of torso tilt (≈1 standing, ≈0 fallen)
    [40] upper_conf         — mean YOLOv8 confidence (nose, shoulders, hips)
indices 41-74  : 17 × (vx, vy) velocity (finite difference of coords 0-33)
```

**Outputs:**
```
processed/urfd/sequences.npz   X: (1041, 30, 75)   y: (1041,)
processed/le2i/sequences.npz   X: (3658, 30, 75)   y: (3658,)
processed/ntu/sequences.npz    X: (535535, 30, 75)  y: (535535,)
```

---

### 5.2 02_data_validation.py

Four-check audit of processed data.  Run this before training.

**Run:**
```bash
python 02_data_validation.py
```

**Checks:**

| Check | What it measures | Flag criterion |
|-------|-----------------|----------------|
| Skeleton quality | Fraction of frames with `upper_conf < 0.3` | > 30% low-conf frames |
| Class balance | Fall/non-fall counts per dataset + 70/15/15 split preview | fall ratio < 0.15 |
| Feature distributions | Torso alignment & CoM velocity, fall vs non-fall KS test | p > 0.01 |
| Temporal coverage | Le2i fall durations from raw annotations | < 10 or > 150 frames |

**Outputs (figures/):**
```
figures/01_skeleton_quality.png
figures/02_class_balance.png
figures/03_feature_distributions.png
figures/04_temporal_coverage.png
```

**Validation results on this dataset:**
```
[Skeleton] URFD    163/1041 flagged (15.7%)  [ACTION REQUIRED]
[Skeleton] Le2i    314/3658 flagged (8.6%)   [WARN]
[Balance]  URFD    fall ratio=0.222           [OK]
[Balance]  Le2i    fall ratio=0.137           [WARN — consider re-weighting]
[Signal]   URFD    torso KS=0.271 p=3.72e-12  [OK]
[Signal]   Le2i    torso KS=0.493 p=9.03e-97  [OK]
[Temporal] Le2i    34 annotation(s) flagged   [WARN]
```

> The 34 flagged Le2i annotations are: 31 videos with `fall_start=-1` (ADL
> videos with no fall — correctly treated as non-fall throughout) and 3 Home_01
> videos with fall durations of 8–9 frames.  No re-processing required; the
> class imbalance in Le2i is handled by FocalLoss + WeightedRandomSampler in
> training.

---

### 5.3 03_train.py

Full model definition and training pipeline.

**Stage A — NTU pretraining:**
```bash
python 03_train.py --stage a --config research_config.json --seed 42
```

**Stage B Variant A — frozen backbone:**
```bash
python 03_train.py --stage b --variant a \
    --resume outputs/checkpoints/stage_a_best.pt \
    --config research_config.json --seed 42
```

**Stage B Variant B — full fine-tuning:**
```bash
python 03_train.py --stage b --variant b \
    --resume outputs/checkpoints/stage_a_best.pt \
    --config research_config.json --seed 42
```

**Debug run (10% of data, no MLflow, fast iteration):**
```bash
python 03_train.py --stage b --variant a --debug --no-mlflow
```

**All CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--stage` | required | `a` = NTU pretraining, `b` = fine-tuning |
| `--variant` | `b` | Stage B only: `a` = frozen backbone, `b` = full fine-tune |
| `--config` | `research_config.json` | Path to config file |
| `--seed` | `42` | Global random seed |
| `--resume` | `None` | Checkpoint path to load backbone weights from |
| `--debug` | `False` | Use 10% of data for fast iteration |
| `--no-mlflow` | `False` | Disable MLflow tracking |
| `--device` | auto | `cuda` or `cpu` |

---

## 6. Model Architecture

### FallDetectionNet

Input shape: `(B, 7, T, J)` where T = window size (30), J = 17 joints.

```
Input (B, 7, 30, 17)
    │
    ▼
JointEmbedding
    Conv2d(7 → 64, k=1) + BN2d + GELU       → (B, 64, 30, 17)
    │
    ▼
Channel Projection
    Conv2d(64 → 256, k=1) + BN2d + GELU     → (B, 256, 30, 17)
    │
    ▼
DilatedTemporalBlock  dilation=1  ┐
DilatedTemporalBlock  dilation=2  ┘ → JointAttention
DilatedTemporalBlock  dilation=4  ┐
DilatedTemporalBlock  dilation=8  ┘ → JointAttention
    │
    ▼
MultiScaleTemporalPooling
    Joint mean → (B, 256, 30)
    Scale 1: global avg          → (B, 256)
    Scale 2: adaptive(15) + avg  → (B, 256)
    Scale 3: adaptive(7)  + avg  → (B, 256)
    concat                       → (B, 768)
    │
    ▼
Classification Head
    Linear(768 → 512) + BN1d + GELU + Dropout(0.4)
    Linear(512 → num_classes)                → (B, K)
```

### DilatedTemporalBlock (inverted residual)

Operates on `(B, C, T, J)` by folding J into the batch dimension:

```
(B, C, T, J) → reshape (B·J, C, T)
    Pointwise expand  Conv1d(C → 2C)   + BN + GELU
    Depthwise dilated Conv1d(2C → 2C, k=3, dilation=d, groups=2C) + BN + GELU
    Pointwise project Conv1d(2C → C)   + BN
    + residual connection
    GELU
reshape → (B, C, T, J)
```

### JointAttention

For each time step, derives soft attention weights over the 17 joints:

```
x: (B, C, T, J)
mean over joints  → (B, C, T) → permute → (B, T, C)
MLP: Linear(C → C//4) → GELU → Linear(C//4 → J) → Softmax
attention: (B, 1, T, J)
output = x * attention
```

### The 7 kinematic features per joint

The `(B, 7, T, 17)` input is derived from processed .npz data:

| Index | Feature | Source |
|-------|---------|--------|
| 0 | x coordinate (normalised) | coords[joint, 0] |
| 1 | y coordinate (normalised) | coords[joint, 1] |
| 2 | vx (x velocity) | vel[joint, 0] |
| 3 | vy (y velocity) | vel[joint, 1] |
| 4 | speed = ‖v‖ | derived |
| 5 | ax (x acceleration) | finite diff of vx |
| 6 | ay (y acceleration) | finite diff of vy |

---

## 7. Training Strategy

### Stage A — NTU Pretraining

| Setting | Value |
|---------|-------|
| Task | 10-class action classification |
| Loss | CrossEntropyLoss |
| Optimizer | AdamW lr=1e-3, wd=1e-4 |
| Scheduler | CosineAnnealingLR |
| Max epochs | 100 |
| Early stopping | patience 15 on val loss |
| Split | 80/20 random |

On first run, raw NTU `.skeleton` files are processed and cached at
`processed/ntu/stage_a_cache.npz` (shape `(N, 7, T, 17)`).
Subsequent runs load from cache instantly.

### Stage B — Fine-tuning

| Setting | Variant A | Variant B |
|---------|-----------|-----------|
| Backbone | Frozen | Trainable |
| Backbone LR | — | 1e-5 |
| Head LR | 1e-3 | 1e-4 |
| Weight decay | — | 1e-4 |
| Scheduler | CosineAnnealingLR | Cosine + 5-epoch warmup |
| Max epochs | 30 | 80 |
| Early stopping | patience 10 on val AUC | patience 15 on val AUC |

### Cross-validation

- **5-fold StratifiedGroupKFold** with sequence ID as the group key
- Sequence IDs are reconstructed from the filesystem (no window-level leakage)
- An integrity check raises `ValueError` if any sequence ID appears in both
  train and test splits
- Within each outer fold, the train set is further split 80/20 by sequence ID
  for validation (early stopping and threshold calibration)

### Loss function

**FocalLoss** (binary, Stage B only):
- `gamma = 2.0`
- `alpha` = per-fold inverse class frequency, capped at `2.5`
- Label smoothing `eps = 0.05`: targets 1→0.95, 0→0.05

### Youden threshold calibration

After each fold:
1. Compute ROC on the validation split with the best-AUC checkpoint
2. Find threshold T* = argmax(sensitivity + specificity − 1)
3. Store T* for that fold

At test time for fold k:
- threshold = mean(T* for all folds j ≠ k)
- Apply this cross-fold threshold to fold k's test predictions

This prevents threshold over-fitting to the validation data.

### Data augmentation (training splits only)

| Augmentation | Detail |
|-------------|--------|
| Joint dropout | Each joint zeroed independently with p=0.1 |
| Gaussian noise | σ=0.01 added to x, y channels (indices 0, 1) |
| Temporal flip | Sequence reversed with p=0.3 — non-fall only |

Fall sequences are never temporally flipped: the direction of motion is
discriminative for fall detection.

### Training infrastructure

| Feature | Detail |
|---------|--------|
| Mixed precision | `torch.cuda.amp.GradScaler` + `autocast` |
| Gradient accumulation | 4 steps → effective batch 128 from physical 32 |
| Gradient clipping | max norm 1.0 |
| Class balancing | `WeightedRandomSampler` with inverse-frequency weights |
| Checkpoint format | `state_dict` + metadata dict (never full model) |

---

## 8. Configuration Reference

All hyperparameters live in `research_config.json`.  No values are hardcoded
in the Python files.

```jsonc
{
  // Dataset paths
  "dataset_root":   "DATASET",
  "output_dir":     "outputs",
  "processed_dir":  "processed",

  // Windowing
  "window_size":        30,   // frames per sequence
  "stride_fall":         5,   // stride for fall windows in Stage B DataModule
  "stride_adl":         10,   // stride for ADL windows
  "stride_onset":        3,   // stride near annotated fall onset (reserved)
  "onset_half_window":  10,   // half-width of the onset zone in frames

  // NTU Stage A action selection
  "ntu_selected_actions": [7, 8, 9, 11, 43, 44, 45, 46, 47, 48],
  // 7=sitting_down  8=standing_up  9=clapping  11=reading
  // 43=fall_down  44=headache  45=chest_pain  46=back_pain
  // 47=neck_pain  48=nausea

  // Architecture
  "in_features":        7,
  "num_joints":        17,
  "joint_embed_dim":   64,
  "backbone_channels": 256,
  "dilations":         [1, 2, 4, 8],
  "dropout":           0.4,

  // Training
  "num_folds":    5,
  "batch_size":  32,    // physical; effective = 32 × accum_steps = 128
  "accum_steps":  4,
  "grad_clip":    1.0,

  // Stage A
  "stage_a": {
    "lr": 1e-3, "weight_decay": 1e-4,
    "num_epochs": 100, "early_stopping_patience": 15,
    "ntu_stride": 5
  },

  // Stage B Variant A (frozen)
  "stage_b_variant_a": {
    "lr_head": 1e-3, "num_epochs": 30, "early_stopping_patience": 10
  },

  // Stage B Variant B (full fine-tune)
  "stage_b_variant_b": {
    "lr_backbone": 1e-5, "lr_head": 1e-4, "weight_decay": 1e-4,
    "warmup_epochs": 5, "num_epochs": 80, "early_stopping_patience": 15
  },

  // Focal loss
  "focal_loss": {
    "gamma": 2.0, "alpha_cap": 2.5, "label_smoothing": 0.05
  },

  // Augmentation
  "augmentation": {
    "joint_dropout_prob": 0.1,
    "noise_sigma": 0.01,
    "temporal_flip_prob": 0.3
  }
}
```

---

## 9. Outputs and Artifacts

### Checkpoints (`outputs/checkpoints/`)

| Filename pattern | Contents |
|-----------------|----------|
| `stage_a_best.pt` | Best Stage A checkpoint (lowest val loss) |
| `best_fold{k}_auc{v:.4f}_epoch{e}_variant{x}.pt` | Best per-fold Stage B checkpoint |
| `final_fold{k}_variant{x}.pt` | Final-epoch Stage B checkpoint |

Each `.pt` file contains:
```python
{
    "state_dict":   model.state_dict(),  # weights only — not full model
    "metadata": {
        "fold":       int,
        "epoch":      int,
        "val_auc":    float,
        "variant":    str,
        "timestamp":  str,   # YYYYMMDD_HHMMSS
    },
    "arch_config": {
        "joint_embed_dim":   64,
        "backbone_channels": 256,
        "dilations":         [1, 2, 4, 8],
        "dropout":           0.4,
        "in_features":       7,
        "num_joints":        17,
    },
}
```

### Figures (`outputs/figures/`)

Per fold, per variant:

| File | Description |
|------|-------------|
| `roc_{tag}.png` | ROC curve with Youden point marked |
| `pr_{tag}.png` | Precision-Recall curve |
| `cm_{tag}.png` | Normalised confusion matrix |

### Logs (`outputs/logs/`)

One timestamped `.log` file per run.  One summary line per epoch in the file;
per-batch progress only on console (tqdm).

### MLflow (`mlruns/`)

Run `mlflow ui` to browse experiments:
```bash
mlflow ui --port 5000
```

MLflow run structure:
```
FallDetectionNet (experiment)
├── stage_a_ntu_pretraining
└── stage_b_finetuning
    ├── stage_b_variant_a_frozen    (nested child run)
    └── stage_b_variant_b_fulltune  (nested child run)
```

Each run logs: all hyperparameters, per-epoch losses, per-fold AUC /
sensitivity / specificity / thresholds, training time, checkpoint artifacts,
ROC/PR/confusion matrix image artifacts.

---

## 10. Dataset Statistics

### Processed windows (after data_pipeline.py)

| Dataset | Windows | Fall | Non-fall | Fall ratio |
|---------|---------|------|----------|------------|
| URFD | 1,041 | 231 | 810 | 0.222 |
| Le2i | 3,658 | 501 | 3,157 | 0.137 |
| NTU (binary) | 535,535 | 3,702 | 531,833 | 0.007 |
| **Combined (B)** | **4,699** | **732** | **3,967** | **0.156** |

### Approximate 70/15/15 stratified splits (Stage B)

| Split | Total | Fall | Non-fall |
|-------|-------|------|----------|
| Train | 3,289 | 512 | 2,777 |
| Val | 704 | 110 | 594 |
| Test | 706 | 110 | 596 |

### NTU Stage A selected actions

| Action ID | Name | Label |
|-----------|------|-------|
| A007 | Sitting down | 0 |
| A008 | Standing up from sitting | 1 |
| A009 | Clapping | 2 |
| A011 | Reading | 3 |
| A043 | Fall down | 4 |
| A044 | Headache | 5 |
| A045 | Chest pain | 6 |
| A046 | Back pain | 7 |
| A047 | Neck pain | 8 |
| A048 | Nausea | 9 |

### Validation audit results

```
[Skeleton] URFD    15.7% flagged   [ACTION REQUIRED — high proportion of
                                    low-confidence frames in ADL sequences]
[Skeleton] Le2i     8.6% flagged   [WARN]
[Signal]   URFD    KS=0.271 p<0.001 [OK — torso/velocity separable]
[Signal]   Le2i    KS=0.493 p<0.001 [OK — strong signal]
[Temporal] Le2i    34 flagged      [31 are ADL-only videos correctly treated
                                    as non-fall; 3 have falls < 10 frames]
```

---

## 11. Reproducing Results

Full pipeline from raw data to trained model:

```bash
# 0. Activate environment
source venv/bin/activate

# 1. Process raw datasets (takes ~15-30 min with GPU)
python data_pipeline.py --datasets urfd le2i ntu --device cuda

# 2. Validate data quality
python 02_data_validation.py

# 3. Stage A pretraining on NTU (requires *.skeleton files)
python 03_train.py --stage a --seed 42

# 4a. Stage B Variant A — frozen backbone
python 03_train.py --stage b --variant a \
    --resume outputs/checkpoints/stage_a_best.pt \
    --seed 42

# 4b. Stage B Variant B — full fine-tuning
python 03_train.py --stage b --variant b \
    --resume outputs/checkpoints/stage_a_best.pt \
    --seed 42

# 5. View results in MLflow
mlflow ui --port 5000
```

To skip Stage A (train Stage B from random init):
```bash
python 03_train.py --stage b --variant b --seed 42
```

---

## 12. Known Issues and Fixes

### pyarrow / NumPy 2.x incompatibility

**Symptom:**
```
AttributeError: _ARRAY_API not found
```
seaborn, pandas, and scikit-learn all fail on import.

**Cause:** `pyarrow<16` was compiled against NumPy 1.x and is binary-incompatible
with NumPy 2.x.

**Fix:**
```bash
pip install "pyarrow>=16"
```

### Le2i AVI segfault / heap corruption

**Symptom:**
```
[mp3float @ 0x...] Header missing
malloc(): invalid size (unsorted)
Aborted (core dumped)
```

**Cause:** Several Le2i AVI files contain a malformed MP3 audio stream.
OpenCV's FFmpeg backend initialises the audio decoder before any Python-level
option can suppress it, causing heap corruption in the native library.

**Fix (implemented in data_pipeline.py):** `_iter_frames_ffmpeg()` bypasses
`cv2.VideoCapture` entirely and pipes raw BGR frames from a `ffmpeg` subprocess
with `-an` (no audio).  Any crash in ffmpeg is isolated to the child process.

### mlflow numpy<2 constraint warning

**Symptom:**
```
mlflow 2.13.0 requires numpy<2, but you have numpy 2.2.6
```

**Status:** Warning only — mlflow 2.13.0 functions correctly with numpy 2.2.6
despite the declared constraint.  Upgrade to `mlflow>=2.14` to silence it.

---

## Citation

If you use this code or pipeline in your research, please cite the datasets:

- **URFD:** Kwolek B., Kepski M. (2014). *Human fall detection on embedded platform using depth maps and wireless accelerometer.* Computer Methods and Programs in Biomedicine.
- **Le2i:** Charfi I. et al. (2013). *Optimised spatio-temporal descriptors for real-time fall detection.* Journal of Electronic Imaging.
- **NTU RGB+D 120:** Liu J. et al. (2020). *NTU RGB+D 120: A large-scale benchmark for 3D human activity understanding.* IEEE TPAMI.
