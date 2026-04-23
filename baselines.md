# Baseline Models — FallDetectionNet Comparison

This document describes the five baseline models used in the paper's comparison table.
All baselines share identical experimental conditions:

- Input: `(B, 7, T, 17)` joint features from the YOLOv8-Pose pipeline
- Datasets: URFD + Le2i (combined, 4,699 windows)
- CV: 5-fold StratifiedGroupKFold with sequence-level grouping
- Loss: FocalLoss with per-fold dynamic alpha, γ=2.0, ε=0.05
- Threshold: Cross-fold Youden-index calibration
- Metrics: AUC-ROC, AUC-PR, Sensitivity, Specificity, F1

---

## 1. BiLSTM — Bidirectional LSTM on Pose Features

**Type:** Recurrent (sequence modelling baseline)

**Architecture:**
- Input flattened to `(B, T, 7×17=119)` per-frame vectors
- Two stacked bidirectional LSTM layers (hidden=256 per direction)
- Final forward + backward hidden states concatenated → 512-dim
- Classification head: Linear(512→256) → BN → GELU → Dropout → Linear(256→2)

**Why included:**
Represents the recurrent family. Directly comparable to prior fall detection
work using LSTM on pose sequences. No spatial graph inductive bias — establishes
the value of modelling joint topology.

**Reference:**
Núñez-Marcos, A., Azkune, G., & Arganda-Carreras, I. (2017).
*Vision-Based Fall Detection with Convolutional Neural Networks.*
Wireless Communications and Mobile Computing, 2017.
**DOI:** [10.1155/2017/9474806](https://doi.org/10.1155/2017/9474806)

---

## 2. TCN — Temporal Convolutional Network

**Type:** Temporal convolution (ablation baseline)

**Architecture:**
- Input merged to `(B, 7×17, T)` per-timestep feature maps
- Pointwise projection to 256 channels
- 4 residual dilated causal convolution blocks with dilations {1, 2, 4, 8}
- Global average pooling over T → 256-dim
- Classification head: Linear(256→256) → BN → GELU → Dropout → Linear(256→2)

**Why included:**
Direct ablation of the proposed model's dilated temporal blocks. Isolates the
contribution of joint attention and multi-scale pooling: TCN uses the same
dilation schedule and channel width but flattens joints into the feature
dimension, discarding spatial structure entirely.

**Reference:**
Bai, S., Kolter, J. Z., & Koltun, V. (2018).
*An Empirical Evaluation of Generic Convolutional and Recurrent Networks for
Sequence Modeling.* arXiv:1803.01271.
**DOI:** [10.48550/arXiv.1803.01271](https://doi.org/10.48550/arXiv.1803.01271)

---

## 3. ST-GCN — Spatial-Temporal Graph Convolutional Network

**Type:** Graph convolution (spatial topology baseline)

**Architecture:**
- Fixed COCO-17 skeleton adjacency (symmetrically D⁻⁰·⁵ AD⁻⁰·⁵ normalised, self-loops added)
- 6 ST-GCN blocks: spatial GCN layer (pointwise conv + A multiplication) followed by
  temporal conv (9×1 kernel), with stride-2 downsampling at blocks 3 and 5
- Channel progression: 7→64→64→128→128→256→256
- Global average pooling over (T, J) → 256-dim
- Classification head: Linear(256→256) → BN → GELU → Dropout → Linear(256→2)

**Why included:**
The dominant skeleton-based action recognition backbone in the literature.
Uses the same COCO-17 joint input as the proposed model, enabling a direct
comparison that isolates the effect of the proposed joint attention and
multi-scale temporal pooling over a fixed-topology GCN.

**Reference:**
Yan, S., Xiong, Y., & Lin, D. (2018).
*Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition.*
Proceedings of the AAAI Conference on Artificial Intelligence, 32(1).
**DOI:** [10.1609/aaai.v32i1.12328](https://doi.org/10.1609/aaai.v32i1.12328)

---

## 4. CTR-GCN — Channel-wise Topology Refinement GCN

**Type:** Hybrid (GCN + channel-wise dynamic topology attention)

**Architecture:**
- Fixed base adjacency A (same COCO-17 as ST-GCN)
- Dynamic residual topology ΔA: per-batch channel-mean statistics of the input
  are passed through a learnable linear layer → `(B, J, J)` residual mask
  bounded by tanh; effective adjacency = A + ΔA
- 6 CTR-GCN blocks with the same channel progression as ST-GCN (7→64→128→256)
- Temporal convolution (9×1) within each block; stride-2 downsampling at blocks 3 and 5
- Global average pooling over (T, J) → 256-dim
- Classification head: Linear(256→256) → BN → GELU → Dropout → Linear(256→2)

**Why included:**
Hybrid architecture that extends ST-GCN with input-adaptive topology. Directly
competes with the proposed model's JointAttention module. Distinguishes the
contribution of per-channel topology refinement (CTR-GCN) vs. per-timestep
soft joint re-weighting (proposed).

**Reference:**
Chen, Y., Zhang, Z., Yuan, C., Li, B., Deng, Y., & Hu, W. (2021).
*Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition.*
Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 13359–13368.
**DOI:** [10.1109/ICCV48922.2021.01311](https://doi.org/10.1109/ICCV48922.2021.01311)

---

## 5. SkateFormer — Skeletal-Temporal Transformer

**Type:** Hybrid Transformer (partition-based spatio-temporal self-attention)

**Architecture:**
- Per-joint linear projection: `(B, T, J, 7)` → `(B, T, J, 128)` with d_model=128
- Learnable temporal positional embedding `(1, T, 1, D)` and joint positional embedding `(1, 1, J, D)`
- 4 SkateFormer encoder blocks, each containing:
  - **Partition-based spatio-temporal attention:** The (T×J) sequence is tiled
    into non-overlapping windows of size (t_part=5 × J=17). Self-attention is
    computed within each local window (4 heads). This replaces global
    self-attention with structured local skeletal-temporal windows, reducing
    complexity from O((TJ)²) to O(TJ·t_part·J).
  - **FFN:** Linear(D→4D) → GELU → Dropout → Linear(4D→D)
  - Pre-norm LayerNorm on both sub-layers
- Global average pooling over (T, J) → 128-dim
- Classification head: Linear(128→128) → GELU → Dropout → Linear(128→2)

**Why included:**
Most recent (TPAMI 2024) and architecturally closest hybrid Transformer to the
current state of the art for skeleton-based action recognition. Using this as
a baseline establishes that the proposed model, despite its lighter design, is
competitive with dedicated spatio-temporal Transformers on the small fall
detection datasets (URFD, Le2i) where data scarcity limits Transformer
pre-training advantages.

**Reference:**
Kim, J., & Ko, H. (2024).
*SkateFormer: Skeletal-Temporal Transformer for Human Action Recognition.*
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).
**DOI:** [10.1109/TPAMI.2024.3506983](https://doi.org/10.1109/TPAMI.2024.3506983)

---

## Summary Table

| # | Model       | Type                    | DOI                                  | Paradigm highlight vs. proposed       |
|---|-------------|-------------------------|--------------------------------------|---------------------------------------|
| 1 | BiLSTM      | Recurrent               | 10.1155/2017/9474806                 | No spatial structure                  |
| 2 | TCN         | Temporal conv           | 10.48550/arXiv.1803.01271            | Same dilation; no joint attention     |
| 3 | ST-GCN      | Graph conv              | 10.1609/aaai.v32i1.12328             | Fixed topology; no temporal attention |
| 4 | CTR-GCN     | Hybrid GCN+attention    | 10.1109/ICCV48922.2021.01311         | Dynamic topology; no multi-scale pool |
| 5 | SkateFormer | Hybrid Transformer      | 10.1109/TPAMI.2024.3506983           | Global structure; data-hungry         |

---

## Training Configuration (shared)

All baselines use the same hyperparameters loaded from `research_config.json`:

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 (head), warm-up 5 epochs |
| Weight decay | 1e-4 |
| Scheduler | Cosine annealing + linear warmup |
| Max epochs | 80 |
| Early stopping patience | 15 (val AUC) |
| Batch size (physical) | 32 |
| Gradient accumulation | 4 steps (effective 128) |
| Gradient clipping | max norm 1.0 |
| Class balancing | WeightedRandomSampler |
| Augmentation | Joint dropout p=0.1, Gaussian noise σ=0.01, temporal flip p=0.3 (non-fall only) |
| Loss | FocalLoss γ=2.0, per-fold dynamic α, label smoothing ε=0.05 |
| Threshold | Cross-fold Youden-index |
| CV | 5-fold StratifiedGroupKFold (sequence-level groups) |

---

## Running Baselines

```bash
# All five baselines (sequential)
python 04_baseline_train.py --config research_config.json --seed 42

# Single model
python 04_baseline_train.py --model skateformer --seed 42
python 04_baseline_train.py --model ctrgcn --seed 42

# Debug run (10% data, no MLflow)
python 04_baseline_train.py --model bilstm --debug --no-mlflow
```

Outputs are written under `outputs/`:

```
outputs/
├── checkpoints/
│   ├── bilstm/      best_fold{k}.pt
│   ├── tcn/
│   ├── stgcn/
│   ├── ctrgcn/
│   └── skateformer/
├── figures/
│   ├── bilstm/      roc_*.png  pr_*.png  cm_*.png
│   ├── tcn/
│   ├── stgcn/
│   ├── ctrgcn/
│   └── skateformer/
└── logs/
    └── baselines_YYYYMMDD_HHMMSS.log
```

The final comparison table (mean ± std across 5 folds) is printed to the log
and can be viewed in MLflow under the `FallDetectionNet_Baselines` experiment:

```bash
mlflow ui --port 5000
```
