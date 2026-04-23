# fmt: off
# ──────────────────────────────────────────────────────────────────────────────
# Requirements (same as 03_train.py):
#   torch==2.2.0  numpy==2.2.6  scikit-learn==1.4.0  mlflow==2.10.0
#   matplotlib==3.8.3  seaborn==0.13.2  scipy==1.12.0  tqdm==4.66.2
# ──────────────────────────────────────────────────────────────────────────────
"""
04_baseline_train.py — Baseline Comparison Suite for FallDetectionNet
======================================================================

Five baselines trained under identical conditions (same data, same 5-fold
StratifiedGroupKFold CV, same FocalLoss, same metrics) for fair comparison
in the journal results table.

Baselines
---------
1. BiLSTM      — Bidirectional LSTM on flattened per-frame pose features
2. TCN         — Temporal Convolutional Network (dilated causal convs)
3. ST-GCN      — Spatial-Temporal Graph Convolutional Network
4. CTR-GCN     — Channel-wise Topology Refinement GCN  [Hybrid]
5. SkateFormer — Skeletal-Temporal Transformer          [Hybrid Transformer]

All models accept the same (B, 7, T, 17) joint-feature input produced by
the main data pipeline and output binary logits (B, 2).

Usage
-----
    # Train all baselines
    python 04_baseline_train.py --config research_config.json --seed 42

    # Train a single baseline
    python 04_baseline_train.py --model bilstm --seed 42
    python 04_baseline_train.py --model tcn    --seed 42
    python 04_baseline_train.py --model stgcn  --seed 42
    python 04_baseline_train.py --model ctrgcn --seed 42
    python 04_baseline_train.py --model skateformer --seed 42

    # Debug run (10 % of data, no MLflow)
    python 04_baseline_train.py --model bilstm --debug --no-mlflow
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── third-party ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

try:
    import mlflow
    import mlflow.pytorch
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

log = logging.getLogger("baseline")

# ──────────────────────────────────────────────────────────────────────────────
# COCO-17 skeleton adjacency (used by ST-GCN / CTR-GCN)
# ──────────────────────────────────────────────────────────────────────────────
# Each pair (i, j) is an undirected bone. Self-loops are added in the model.
COCO17_EDGES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # arms
    (5, 11), (6, 12),                          # shoulders → hips
    (11, 12), (11, 13), (13, 15),             # left leg
    (12, 14), (14, 16),                        # right leg
]

NUM_JOINTS = 17


def build_adjacency(num_joints: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build a normalised symmetric adjacency matrix with self-loops.

    Returns:
        (J, J) float32 tensor — symmetrically normalised (D^-0.5 A D^-0.5).
    """
    A = torch.zeros(num_joints, num_joints)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + torch.eye(num_joints)          # self-loops
    D = A.sum(dim=1).pow(-0.5)
    D[D == float("inf")] = 0.0
    return (D.unsqueeze(1) * A * D.unsqueeze(0))


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BaselineConfig:
    processed_dir: str  = "processed"
    output_dir:    str  = "outputs"
    num_folds:     int  = 5
    batch_size:    int  = 32
    accum_steps:   int  = 4
    grad_clip:     float = 1.0
    num_epochs:    int  = 80
    early_stopping_patience: int = 15
    lr:            float = 1e-3
    weight_decay:  float = 1e-4
    warmup_epochs: int  = 5
    dropout:       float = 0.4
    focal_gamma:   float = 2.0
    focal_alpha_cap: float = 2.5
    focal_eps:     float = 0.05

    # augmentation (same as proposed model)
    joint_dropout_prob: float = 0.1
    noise_sigma:        float = 0.01
    temporal_flip_prob: float = 0.3


def load_config(path: str | None) -> BaselineConfig:
    cfg = BaselineConfig()
    if path is None:
        return cfg
    with open(path) as f:
        raw = json.load(f)
    mapping = {
        "processed_dir":  "processed_dir",
        "output_dir":     "output_dir",
        "num_folds":      "num_folds",
        "batch_size":     "batch_size",
        "accum_steps":    "accum_steps",
        "grad_clip":      "grad_clip",
        "dropout":        "dropout",
    }
    for src, dst in mapping.items():
        if src in raw:
            setattr(cfg, dst, raw[src])
    if "focal_loss" in raw:
        fl = raw["focal_loss"]
        if "gamma"          in fl: cfg.focal_gamma     = fl["gamma"]
        if "alpha_cap"      in fl: cfg.focal_alpha_cap = fl["alpha_cap"]
        if "label_smoothing" in fl: cfg.focal_eps      = fl["label_smoothing"]
    if "augmentation" in raw:
        ag = raw["augmentation"]
        if "joint_dropout_prob" in ag: cfg.joint_dropout_prob = ag["joint_dropout_prob"]
        if "noise_sigma"        in ag: cfg.noise_sigma        = ag["noise_sigma"]
        if "temporal_flip_prob" in ag: cfg.temporal_flip_prob = ag["temporal_flip_prob"]
    # stage_b_variant_b shares epochs/lr as a reasonable default for baselines
    if "stage_b_variant_b" in raw:
        bvb = raw["stage_b_variant_b"]
        if "num_epochs"              in bvb: cfg.num_epochs              = bvb["num_epochs"]
        if "early_stopping_patience" in bvb: cfg.early_stopping_patience = bvb["early_stopping_patience"]
        if "warmup_epochs"           in bvb: cfg.warmup_epochs           = bvb["warmup_epochs"]
        if "lr_head"                 in bvb: cfg.lr                      = bvb["lr_head"]
        if "weight_decay"            in bvb: cfg.weight_decay            = bvb["weight_decay"]
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# Feature conversion  (identical to 03_train.py)
# ──────────────────────────────────────────────────────────────────────────────

def npz_window_to_joint_features(window: np.ndarray) -> np.ndarray:
    """(T, 75) → (7, T, 17) joint feature tensor."""
    T = window.shape[0]
    coords = window[:, 0:34].reshape(T, 17, 2)
    vel    = window[:, 41:75].reshape(T, 17, 2)
    x  = coords[:, :, 0]; y  = coords[:, :, 1]
    vx = vel[:, :, 0];    vy = vel[:, :, 1]
    speed = np.sqrt(vx ** 2 + vy ** 2)
    ax = np.zeros_like(vx); ay = np.zeros_like(vy)
    ax[1:] = vx[1:] - vx[:-1]; ay[1:] = vy[1:] - vy[:-1]
    feat = np.stack([x, y, vx, vy, speed, ax, ay], axis=0)
    return feat.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset  (identical augmentation pipeline to 03_train.py)
# ──────────────────────────────────────────────────────────────────────────────

class SkeletonDataset(Dataset):
    """(B, 7, T, 17) skeleton dataset with training augmentation."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_ids: np.ndarray,
        is_train: bool = False,
        cfg: BaselineConfig | None = None,
    ) -> None:
        self.X        = torch.from_numpy(X).float()
        self.y        = torch.from_numpy(y).long()
        self.seq_ids  = seq_ids
        self.is_train = is_train
        self.cfg      = cfg or BaselineConfig()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.is_train:
            x = self._augment(x, y)
        return x, y

    def _augment(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        J = x.shape[-1]
        mask = torch.rand(J) >= cfg.joint_dropout_prob
        x = x * mask.unsqueeze(0).unsqueeze(0)
        x[:2] = x[:2] + torch.randn_like(x[:2]) * cfg.noise_sigma
        if y.item() != 1 and torch.rand(1).item() < cfg.temporal_flip_prob:
            x = x.flip(dims=[1])
        return x

    def class_weights_for_sampler(self) -> torch.Tensor:
        labels = self.y.numpy()
        n_fall    = max((labels == 1).sum(), 1)
        n_nonfall = max((labels == 0).sum(), 1)
        w_fall    = len(labels) / (2.0 * n_fall)
        w_nonfall = len(labels) / (2.0 * n_nonfall)
        return torch.from_numpy(
            np.where(labels == 1, w_fall, w_nonfall)
        ).float()


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Binary focal loss with label smoothing — identical to proposed model."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0,
                 eps: float = 0.05, alpha_cap: float = 2.5) -> None:
        super().__init__()
        self.alpha = min(float(alpha), alpha_cap)
        self.gamma = gamma
        self.eps   = eps

    @classmethod
    def from_labels(cls, y: np.ndarray, gamma: float, eps: float,
                    alpha_cap: float) -> "FocalLoss":
        n_pos = max(int((y == 1).sum()), 1)
        return cls(alpha=len(y) / (2.0 * n_pos), gamma=gamma,
                   eps=eps, alpha_cap=alpha_cap)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p_fall = F.softmax(logits.float(), dim=1)[:, 1]
        t      = targets.float()
        smooth_t  = t * (1.0 - self.eps) + (1.0 - t) * self.eps
        p_clamped = p_fall.clamp(1e-7, 1.0 - 1e-7)
        bce = -(smooth_t * torch.log(p_clamped) +
                (1.0 - smooth_t) * torch.log(1.0 - p_clamped))
        pt = torch.where(targets.bool(), p_fall, 1.0 - p_fall)
        alpha_t = torch.where(targets.bool(),
                              torch.full_like(p_fall, self.alpha),
                              torch.ones_like(p_fall))
        return (alpha_t * (1.0 - pt).pow(self.gamma) * bce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# ── BASELINE 1: BiLSTM ────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class BiLSTM(nn.Module):
    """Bidirectional LSTM on flattened per-frame pose features.

    Each frame is represented as a (7 × 17 = 119)-dim vector.  Two stacked
    Bi-LSTM layers encode the sequence; the final hidden states of both
    directions are concatenated and passed to the classifier.

    Reference: Núñez-Marcos et al. (2017). Vision-Based Fall Detection with
    Convolutional Neural Networks. Wireless Communications and Mobile
    Computing. DOI: 10.1155/2017/9474806
    """

    def __init__(self, in_features: int = 7, num_joints: int = 17,
                 hidden: int = 256, num_layers: int = 2,
                 dropout: float = 0.4, num_classes: int = 2) -> None:
        super().__init__()
        input_dim = in_features * num_joints      # 119
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 7, T, 17)
        Returns:
            (B, num_classes) logits
        """
        B, F, T, J = x.shape
        # flatten features × joints per frame → (B, T, F*J)
        x = x.permute(0, 2, 1, 3).reshape(B, T, F * J)
        _, (h_n, _) = self.lstm(x)               # h_n: (2*layers, B, H)
        # concat final forward + backward hidden states
        fwd = h_n[-2]                             # (B, H)
        bwd = h_n[-1]                             # (B, H)
        return self.head(torch.cat([fwd, bwd], dim=-1))


# ──────────────────────────────────────────────────────────────────────────────
# ── BASELINE 2: TCN ───────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class _TCNBlock(nn.Module):
    """Single residual TCN block: two dilated causal conv layers.

    Uses explicit left-only (causal) padding via F.pad instead of symmetric
    Conv1d padding.  This avoids cuDNN CUDNN_STATUS_NOT_SUPPORTED errors that
    occur with large even-padding dilated Conv1d on some GPU/cuDNN versions.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 dilation: int, dropout: float) -> None:
        super().__init__()
        # Receptive field on the left only (causal): (kernel-1)*dilation zeros prepended
        self.causal_pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=0, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=0, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        # Pad left only so the conv sees only past context
        x = F.pad(x, (self.causal_pad, 0))
        return conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn1(self._causal_conv(self.conv1, x)))
        h = self.drop(h)
        h = F.gelu(self.bn2(self._causal_conv(self.conv2, h)))
        h = self.drop(h)
        return F.gelu(h + self.downsample(x))


class TCN(nn.Module):
    """Temporal Convolutional Network with exponentially increasing dilation.

    Joints are mean-pooled before the temporal stack; the resulting
    (B, C, T) tensor is processed by residual dilated causal convolutions.

    Reference: Bai et al. (2018). An Empirical Evaluation of Generic
    Convolutional and Recurrent Networks for Sequence Modeling.
    arXiv:1803.01271. DOI: 10.48550/arXiv.1803.01271
    """

    def __init__(self, in_features: int = 7, num_joints: int = 17,
                 channels: int = 256, num_blocks: int = 4,
                 kernel: int = 3, dropout: float = 0.4,
                 num_classes: int = 2) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_features * num_joints, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        blocks: list[nn.Module] = []
        for i in range(num_blocks):
            blocks.append(_TCNBlock(channels, channels, kernel,
                                    dilation=2 ** i, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(channels, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 7, T, 17)
        Returns:
            (B, num_classes) logits
        """
        B, F, T, J = x.shape
        # merge features and joints → (B, F*J, T)
        x = x.reshape(B, F * J, T)
        x = self.input_proj(x)                   # (B, C, T)
        x = self.tcn(x)                          # (B, C, T)
        x = x.mean(dim=-1)                       # (B, C) global avg pool
        return self.head(x)


# ──────────────────────────────────────────────────────────────────────────────
# ── BASELINE 3: ST-GCN ────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class _STGCNBlock(nn.Module):
    """One ST-GCN layer: graph conv over joints then temporal conv.

    The temporal conv uses Conv1d with J folded into the batch dimension to
    avoid cuDNN CUDNN_STATUS_NOT_SUPPORTED errors from asymmetric-kernel
    Conv2d (k_h, 1) with asymmetric padding under autocast on Ada GPUs.
    """

    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor,
                 t_kernel: int = 9, stride: int = 1,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("A", A)              # (J, J) fixed normalised adj
        self.stride = stride

        # Spatial GCN: pointwise conv + A multiplication
        self.gcn_w  = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.gcn_bn = nn.BatchNorm2d(out_ch)

        # Temporal conv: Conv1d over T with J folded into batch
        pad = (t_kernel - 1) // 2
        self.tcn_conv = nn.Conv1d(out_ch, out_ch, t_kernel,
                                  stride=stride, padding=pad)
        self.tcn_bn   = nn.BatchNorm1d(out_ch)
        self.drop     = nn.Dropout(dropout)

        # Residual shortcut: separate channel proj (stride=1) + avg_pool stride.
        # A single Conv1d(in, out, 1, stride=s) under fp16 triggers
        # CUDNN_STATUS_NOT_SUPPORTED when in != out and s > 1 on Ada GPUs.
        self.need_proj   = (in_ch != out_ch)
        self.res_stride  = stride
        if self.need_proj:
            self.res_conv = nn.Conv1d(in_ch, out_ch, 1, bias=False)
            self.res_bn   = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, J)
        Returns:
            (B, out_ch, T', J)
        """
        B, C, T, J = x.shape

        # Spatial graph convolution
        h = self.gcn_w(x)                              # (B, out_ch, T, J)
        h = torch.einsum("bctj,jk->bctk", h, self.A)  # (B, out_ch, T, J)
        h = F.gelu(self.gcn_bn(h))

        # Temporal convolution: fold J into batch for Conv1d
        h = h.permute(0, 3, 1, 2).reshape(B * J, -1, T)   # (B*J, out_ch, T)
        h = self.tcn_bn(self.tcn_conv(h))                  # (B*J, out_ch, T')
        T2 = h.shape[-1]
        h = h.reshape(B, J, -1, T2).permute(0, 2, 3, 1)   # (B, out_ch, T', J)

        # Residual shortcut: channel proj then adaptive_avg_pool to match T2
        rx = x.permute(0, 3, 1, 2).reshape(B * J, C, T)   # (B*J, C, T)
        if self.need_proj:
            rx = self.res_bn(self.res_conv(rx))             # (B*J, out_ch, T)
        if self.res_stride > 1:
            rx = F.adaptive_avg_pool1d(rx, T2)              # (B*J, out_ch, T2)
        res = rx.reshape(B, J, -1, T2).permute(0, 2, 3, 1) # (B, out_ch, T', J)

        return self.drop(F.gelu(h + res))


class STGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network for fall detection.

    Follows the layered GCN + temporal conv design with a fixed COCO-17
    skeleton graph. Global average pooling over (T, J) produces the
    representation fed to the classifier.

    Reference: Yan et al. (2018). Spatial Temporal Graph Convolutional
    Networks for Skeleton-Based Action Recognition. AAAI 2018.
    DOI: 10.1609/aaai.v32i1.12328
    """

    def __init__(self, in_features: int = 7, num_joints: int = 17,
                 dropout: float = 0.4, num_classes: int = 2) -> None:
        super().__init__()
        A = build_adjacency(num_joints, COCO17_EDGES)
        self.register_buffer("A", A)

        self.data_bn = nn.BatchNorm1d(in_features * num_joints)
        cfg = [
            (in_features,  64,  1),
            (64,           64,  1),
            (64,           128, 2),
            (128,          128, 1),
            (128,          256, 2),
            (256,          256, 1),
        ]
        layers: list[nn.Module] = []
        for in_c, out_c, stride in cfg:
            layers.append(_STGCNBlock(in_c, out_c, A,
                                      stride=stride, dropout=dropout))
        self.layers = nn.ModuleList(layers)

        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 7, T, 17)
        Returns:
            (B, num_classes) logits
        """
        B, C, T, J = x.shape
        # BN applied on flattened C*J view
        x_bn = x.permute(0, 2, 1, 3).reshape(B * T, C * J)
        x_bn = self.data_bn(x_bn).reshape(B, T, C, J).permute(0, 2, 1, 3)
        h = x_bn
        for layer in self.layers:
            h = layer(h)                           # (B, C', T', J)
        h = h.mean(dim=[2, 3])                     # (B, C') global avg pool
        return self.head(h)


# ──────────────────────────────────────────────────────────────────────────────
# ── BASELINE 4: CTR-GCN  [Hybrid: GCN + channel-wise topology attention] ─────
# ──────────────────────────────────────────────────────────────────────────────

class _CTRGCNBlock(nn.Module):
    """CTR-GCN block: shared + individual topology GCN + temporal conv.

    The topology is the sum of a fixed base adjacency A and a per-sample
    channel-wise residual mask ΔA predicted from the input features.
    This lets each channel attend to a different subset of joints.

    Reference: Chen et al. (2021). Channel-wise Topology Refinement Graph
    Convolution for Skeleton-Based Action Recognition. ICCV 2021.
    DOI: 10.1109/ICCV48922.2021.01311
    """

    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor,
                 t_kernel: int = 9, stride: int = 1,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("A", A)               # (J, J) fixed base topology
        J = A.shape[0]

        # Topology refinement: predict per-channel ΔA from channel-wise statistics
        self.topology_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, J)),           # (B, C, 1, J)
            nn.Flatten(2),                          # (B, C, J) — after squeeze
        )
        # shared linear that maps (J,) → (J, J) per channel group
        self.delta_fc = nn.Linear(J, J * J, bias=False)

        # Main GCN path
        self.gcn_w  = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.gcn_bn = nn.BatchNorm2d(out_ch)

        # Temporal conv: Conv1d with J folded into batch (avoids cuDNN
        # CUDNN_STATUS_NOT_SUPPORTED from asymmetric Conv2d under autocast)
        pad = (t_kernel - 1) // 2
        self.tcn_conv = nn.Conv1d(out_ch, out_ch, t_kernel,
                                  stride=stride, padding=pad, bias=False)
        self.tcn_bn   = nn.BatchNorm1d(out_ch)
        self.drop     = nn.Dropout(dropout)

        # Residual shortcut: separate channel proj (stride=1) + avg_pool stride.
        # Same cuDNN fp16 restriction as _STGCNBlock.
        self.need_proj  = (in_ch != out_ch)
        self.res_stride = stride
        if self.need_proj:
            self.res_conv = nn.Conv1d(in_ch, out_ch, 1, bias=False)
            self.res_bn   = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, J)
        Returns:
            (B, out_ch, T', J)
        """
        B, C, T, J = x.shape

        # ΔA: (B, J, J) channel-mean residual topology
        stats = x.mean(dim=[1, 2])                      # (B, J)
        delta = self.delta_fc(stats).reshape(B, J, J)   # (B, J, J)
        delta = torch.tanh(delta)
        A_dyn = self.A.unsqueeze(0) + delta             # (B, J, J)

        # Spatial graph conv
        h = self.gcn_w(x)                               # (B, out_ch, T, J)
        h = torch.einsum("bctj,bjk->bctk", h, A_dyn)
        h = F.gelu(self.gcn_bn(h))

        # Temporal conv: fold J into batch for Conv1d
        h = h.permute(0, 3, 1, 2).reshape(B * J, -1, T)    # (B*J, out_ch, T)
        h = self.tcn_bn(self.tcn_conv(h))                   # (B*J, out_ch, T')
        T2 = h.shape[-1]
        h = h.reshape(B, J, -1, T2).permute(0, 2, 3, 1)    # (B, out_ch, T', J)

        # Residual shortcut: channel proj then adaptive_avg_pool to match T2
        rx = x.permute(0, 3, 1, 2).reshape(B * J, C, T)    # (B*J, C, T)
        if self.need_proj:
            rx = self.res_bn(self.res_conv(rx))              # (B*J, out_ch, T)
        if self.res_stride > 1:
            rx = F.adaptive_avg_pool1d(rx, T2)              # (B*J, out_ch, T2)
        res = rx.reshape(B, J, -1, T2).permute(0, 2, 3, 1)  # (B, out_ch, T', J)

        return self.drop(F.gelu(h + res))


class CTRGCN(nn.Module):
    """Channel-wise Topology Refinement GCN.

    Hybrid architecture: graph convolution with dynamically refined
    per-channel topology + temporal convolution residual blocks.

    Reference: Chen et al. (2021). Channel-wise Topology Refinement Graph
    Convolution for Skeleton-Based Action Recognition. ICCV 2021.
    DOI: 10.1109/ICCV48922.2021.01311
    """

    def __init__(self, in_features: int = 7, num_joints: int = 17,
                 dropout: float = 0.4, num_classes: int = 2) -> None:
        super().__init__()
        A = build_adjacency(num_joints, COCO17_EDGES)
        self.register_buffer("A", A)

        self.data_bn = nn.BatchNorm1d(in_features * num_joints)

        cfg = [
            (in_features, 64,  1),
            (64,          64,  1),
            (64,          128, 2),
            (128,         128, 1),
            (128,         256, 2),
            (256,         256, 1),
        ]
        layers: list[nn.Module] = []
        for in_c, out_c, stride in cfg:
            layers.append(_CTRGCNBlock(in_c, out_c, A,
                                       stride=stride, dropout=dropout))
        self.layers = nn.ModuleList(layers)

        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 7, T, 17)
        Returns:
            (B, num_classes) logits
        """
        B, C, T, J = x.shape
        x_bn = x.permute(0, 2, 1, 3).reshape(B * T, C * J)
        x_bn = self.data_bn(x_bn).reshape(B, T, C, J).permute(0, 2, 1, 3)
        h = x_bn
        for layer in self.layers:
            h = layer(h)
        h = h.mean(dim=[2, 3])
        return self.head(h)


# ──────────────────────────────────────────────────────────────────────────────
# ── BASELINE 5: SkateFormer  [Hybrid Transformer] ────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class _PartitionAttention(nn.Module):
    """Partition-based spatio-temporal self-attention.

    The full (T × J) sequence is partitioned into non-overlapping local
    windows of size (t_part × j_part).  Attention is computed within each
    window.  This is the core operation in SkateFormer that replaces
    global self-attention with local skeletal-temporal windows.

    Reference: Kim et al. (2024). SkateFormer: Skeletal-Temporal Transformer
    for Human Action Recognition. IEEE TPAMI.
    DOI: 10.1109/TPAMI.2024.3506983
    """

    def __init__(self, dim: int, num_heads: int, t_part: int = 5,
                 j_part: int = 17, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.t_part    = t_part
        self.j_part    = j_part

        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, d_model)
        Returns:
            (B, T, J, d_model)
        """
        B, T, J, D = x.shape
        tp, jp = self.t_part, self.j_part

        # Pad T to a multiple of tp
        pad_t = (tp - T % tp) % tp
        if pad_t:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_t))
        T_pad = x.shape[1]

        # Partition: (B, T_pad/tp, tp, J, D)
        x = x.reshape(B, T_pad // tp, tp, J, D)
        # flatten window tokens: (B * T_pad/tp, tp*J, D)
        Bw = B * (T_pad // tp)
        x  = x.reshape(Bw, tp * J, D)

        qkv = self.qkv(x).reshape(Bw, tp * J, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, Bw, H, N, hd)
        q, k, v = qkv.unbind(0)                    # each (Bw, H, N, hd)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.drop(attn.softmax(dim=-1))
        x    = (attn @ v).transpose(1, 2).reshape(Bw, tp * J, D)
        x    = self.proj(x)                        # (Bw, tp*J, D)

        # Restore: (B, T_pad, J, D)
        x = x.reshape(B, T_pad // tp, tp, J, D).reshape(B, T_pad, J, D)
        # Remove temporal padding
        if pad_t:
            x = x[:, :T]
        return x


class _SkateFormerBlock(nn.Module):
    """SkateFormer encoder block: partition attention + FFN."""

    def __init__(self, dim: int, num_heads: int, t_part: int,
                 j_part: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = _PartitionAttention(dim, num_heads, t_part, j_part, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, D)
        Returns:
            (B, T, J, D)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SkateFormer(nn.Module):
    """Skeletal-Temporal Transformer for fall detection.

    Hybrid Transformer architecture: learnable joint + temporal positional
    embeddings, partition-based spatio-temporal self-attention blocks, and a
    linear classification head.  Operates on (B, 7, T, 17) joint features.

    Reference: Kim et al. (2024). SkateFormer: Skeletal-Temporal Transformer
    for Human Action Recognition. IEEE TPAMI 2024.
    DOI: 10.1109/TPAMI.2024.3506983
    """

    def __init__(self, in_features: int = 7, num_joints: int = 17,
                 seq_len: int = 30, d_model: int = 128, num_heads: int = 4,
                 num_layers: int = 4, t_part: int = 5, mlp_ratio: float = 4.0,
                 dropout: float = 0.4, num_classes: int = 2) -> None:
        super().__init__()
        self.d_model = d_model

        # Per-joint feature projection
        self.input_proj = nn.Linear(in_features, d_model, bias=False)

        # Learnable positional embeddings over time and joints
        self.pos_t = nn.Parameter(torch.zeros(1, seq_len, 1, d_model))
        self.pos_j = nn.Parameter(torch.zeros(1, 1, num_joints, d_model))

        self.blocks = nn.ModuleList([
            _SkateFormerBlock(d_model, num_heads, t_part, num_joints,
                              mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        nn.init.trunc_normal_(self.pos_t, std=0.02)
        nn.init.trunc_normal_(self.pos_j, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 7, T, 17)
        Returns:
            (B, num_classes) logits
        """
        B, F, T, J = x.shape
        # (B, T, J, F) → project to d_model
        x = x.permute(0, 2, 3, 1)                 # (B, T, J, F)
        x = self.input_proj(x)                    # (B, T, J, D)

        # Add positional embeddings (broadcast over T and J)
        x = x + self.pos_t[:, :T] + self.pos_j

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)                          # (B, T, J, D)
        x = x.mean(dim=[1, 2])                    # (B, D) global avg pool
        return self.head(x)


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

def build_model(name: str, num_joints: int = 17,
                in_features: int = 7, seq_len: int = 30,
                dropout: float = 0.4) -> nn.Module:
    """Instantiate a baseline model by name.

    Args:
        name:        One of bilstm | tcn | stgcn | ctrgcn | skateformer.
        num_joints:  Number of skeleton joints (17 for COCO).
        in_features: Kinematic features per joint (7).
        seq_len:     Temporal window length (30 frames).
        dropout:     Dropout rate.

    Returns:
        Initialised nn.Module.
    """
    kwargs = dict(in_features=in_features, num_joints=num_joints,
                  dropout=dropout, num_classes=2)
    if name == "bilstm":
        return BiLSTM(**kwargs)
    if name == "tcn":
        return TCN(**kwargs)
    if name == "stgcn":
        return STGCN(**kwargs)
    if name == "ctrgcn":
        return CTRGCN(**kwargs)
    if name == "skateformer":
        return SkateFormer(**kwargs, seq_len=seq_len)
    raise ValueError(f"Unknown model: {name!r}. "
                     "Choose from: bilstm tcn stgcn ctrgcn skateformer")


ALL_MODELS = ["bilstm", "tcn", "stgcn", "ctrgcn", "skateformer"]


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler with linear warmup + cosine decay
# ──────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer: torch.optim.Optimizer,
                 warmup_epochs: int, total_epochs: int) -> None:
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = (epoch - warmup_epochs) / float(
                max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        super().__init__(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    pr, re, _   = precision_recall_curve(y_true, y_prob)
    return {
        "auc_roc":     float(roc_auc_score(y_true, y_prob)),
        "auc_pr":      float(auc(re, pr)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_roc(y_true: np.ndarray, y_prob: np.ndarray, path: Path,
              threshold: float, tag: str) -> None:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    youden_idx = np.argmin(np.abs(thresholds - threshold))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    ax.scatter(fpr[youden_idx], tpr[youden_idx], color="red",
               zorder=5, label=f"T*={threshold:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="FPR", ylabel="TPR", title=f"ROC — {tag}")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _save_pr(y_true: np.ndarray, y_prob: np.ndarray, path: Path,
             tag: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"AUC={pr_auc:.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"PR — {tag}")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _save_cm(y_true: np.ndarray, y_pred: np.ndarray, path: Path,
             tag: str) -> None:
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=["Non-fall", "Fall"],
                yticklabels=["Non-fall", "Fall"])
    ax.set(title=f"CM — {tag}", xlabel="Predicted", ylabel="True")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_stage_b_data(cfg: BaselineConfig, debug: bool = False
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load URFD + Le2i processed windows → (N, 7, T, 17), y, seq_ids."""
    proc = Path(cfg.processed_dir)

    def _load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        p = proc / name / "sequences.npz"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}. Run data_pipeline.py first.")
        d = np.load(str(p))
        seq_ids = d["seq_ids"].astype(np.int64) if "seq_ids" in d else None
        return d["X"].astype(np.float32), d["y"].astype(np.int64), seq_ids

    X_u, y_u, ids_u = _load("urfd")
    X_l, y_l, ids_l = _load("le2i")

    # Fallback seq_id reconstruction (same logic as 03_train.py)
    if ids_u is None:
        ids_u = np.repeat(np.arange(70), math.ceil(len(X_u) / 70))[:len(X_u)]
    if ids_l is None:
        offset  = int(ids_u.max()) + 1
        n_seqs  = 70
        ids_l   = np.repeat(np.arange(n_seqs), math.ceil(len(X_l) / n_seqs))[:len(X_l)] + offset

    log.info("Converting URFD windows …")
    X_u_jf = np.stack([npz_window_to_joint_features(X_u[i]) for i in range(len(X_u))])
    log.info("Converting Le2i windows …")
    X_l_jf = np.stack([npz_window_to_joint_features(X_l[i]) for i in range(len(X_l))])

    X       = np.concatenate([X_u_jf, X_l_jf], axis=0)
    y       = np.concatenate([y_u, y_l],        axis=0)
    seq_ids = np.concatenate([ids_u, ids_l],    axis=0)

    if debug:
        n   = max(100, len(y) // 10)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), n, replace=False)
        X, y, seq_ids = X[idx], y[idx], seq_ids[idx]
        log.info("Debug mode: %d / %d windows", n, len(y))

    log.info("Data loaded: %d windows  fall=%d  non-fall=%d",
             len(y), (y == 1).sum(), (y == 0).sum())
    return X, y, seq_ids


# ──────────────────────────────────────────────────────────────────────────────
# Training loop (one fold)
# ──────────────────────────────────────────────────────────────────────────────

def train_fold(
    model:      nn.Module,
    train_ds:   SkeletonDataset,
    val_ds:     SkeletonDataset,
    cfg:        BaselineConfig,
    device:     str,
    fold:       int,
    model_name: str,
) -> tuple[nn.Module, float, float]:
    """Train for one CV fold.

    Returns:
        Best model (state restored to best val AUC checkpoint),
        best val AUC, and Youden threshold calibrated on val set.
    """
    sampler = WeightedRandomSampler(
        train_ds.class_weights_for_sampler(), len(train_ds), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size * 2,
                              shuffle=False, num_workers=2, pin_memory=True)

    y_train = train_ds.y.numpy()
    criterion = FocalLoss.from_labels(
        y_train, cfg.focal_gamma, cfg.focal_eps, cfg.focal_alpha_cap
    )
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, cfg.warmup_epochs, cfg.num_epochs)
    scaler    = GradScaler()

    best_auc    = 0.0
    best_thresh = 0.5
    best_state  = None
    patience    = 0

    for epoch in range(1, cfg.num_epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, (xb, yb) in enumerate(
            tqdm(train_loader, desc=f"  Fold {fold} ep {epoch}", leave=False)
        ):
            xb, yb = xb.to(device), yb.to(device)
            with autocast(device_type=device.split(":")[0]):
                loss = criterion(model(xb), yb) / cfg.accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % cfg.accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

            train_loss += loss.item() * cfg.accum_steps

        scheduler.step()

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        all_prob, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                prob = F.softmax(model(xb).float(), dim=1)[:, 1].cpu().numpy()
                all_prob.append(prob)
                all_true.append(yb.numpy())

        y_prob = np.concatenate(all_prob)
        y_true = np.concatenate(all_true)

        if len(np.unique(y_true)) < 2:
            val_auc = 0.5
        else:
            val_auc = float(roc_auc_score(y_true, y_prob))

        train_loss /= max(len(train_loader), 1)
        log.info("  Fold %d  Ep %d/%d  loss=%.4f  val_auc=%.4f",
                 fold, epoch, cfg.num_epochs, train_loss, val_auc)

        if val_auc > best_auc:
            best_auc    = val_auc
            best_thresh = youden_threshold(y_true, y_prob)
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience    = 0
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                log.info("  Early stopping at epoch %d", epoch)
                break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, best_auc, best_thresh


# ──────────────────────────────────────────────────────────────────────────────
# Full cross-validated training run for one model
# ──────────────────────────────────────────────────────────────────────────────

def run_baseline(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    seq_ids: np.ndarray,
    cfg: BaselineConfig,
    device: str,
    seed: int,
    out_dir: Path,
    use_mlflow: bool,
    resume: bool = False,
) -> dict[str, float]:
    """5-fold cross-validated training + evaluation for one baseline.

    Returns:
        Dict of mean ± std metrics across folds.
    """
    log.info("=" * 60)
    log.info("Baseline: %s", model_name.upper())
    log.info("=" * 60)

    fig_dir  = out_dir / "figures" / model_name
    ckpt_dir = out_dir / "checkpoints" / model_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedGroupKFold(n_splits=cfg.num_folds)
    fold_metrics: list[dict[str, float]] = []
    fold_thresholds: list[float] = []

    all_splits = list(skf.split(X, y, seq_ids))

    # --resume: reload metrics from any already-finished folds
    if resume:
        for fi in range(1, cfg.num_folds + 1):
            ckpt_path = ckpt_dir / f"best_fold{fi}.pt"
            if ckpt_path.exists():
                saved = torch.load(ckpt_path, map_location="cpu")
                fold_metrics.append(saved["metrics"])
                fold_thresholds.append(saved["threshold"])
                log.info("  [resume] Fold %d — loaded from %s", fi, ckpt_path)

        if len(fold_metrics) == cfg.num_folds:
            log.info("  [resume] All folds complete — skipping %s", model_name.upper())
            keys = ["auc_roc", "auc_pr", "sensitivity", "specificity", "f1", "accuracy"]
            summary: dict[str, float] = {}
            for k in keys:
                vals = [m[k] for m in fold_metrics]
                summary[f"{k}_mean"] = float(np.mean(vals))
                summary[f"{k}_std"]  = float(np.std(vals))
            return summary

    for fold_idx, (train_val_idx, test_idx) in enumerate(all_splits):
        fold = fold_idx + 1
        set_seed(seed + fold)

        # --resume: skip folds whose checkpoint was already loaded above
        if resume and (fold_idx < len(fold_thresholds)):
            log.info("  [resume] Fold %d already done — skipping", fold)
            continue

        # ── inner split: 80% train / 20% val by sequence ──────────────────
        train_val_seqs = np.unique(seq_ids[train_val_idx])
        rng = np.random.default_rng(seed + fold)
        rng.shuffle(train_val_seqs)
        n_val   = max(1, int(len(train_val_seqs) * 0.2))
        val_seqs = set(train_val_seqs[:n_val])

        train_mask = np.array(
            [seq_ids[i] not in val_seqs for i in train_val_idx]
        )
        train_idx = train_val_idx[train_mask]
        val_idx   = train_val_idx[~train_mask]

        train_ds = SkeletonDataset(X[train_idx], y[train_idx],
                                   seq_ids[train_idx], is_train=True, cfg=cfg)
        val_ds   = SkeletonDataset(X[val_idx],   y[val_idx],
                                   seq_ids[val_idx],   is_train=False)
        test_ds  = SkeletonDataset(X[test_idx],  y[test_idx],
                                   seq_ids[test_idx],  is_train=False)

        model = build_model(
            model_name,
            num_joints=X.shape[-1],
            in_features=X.shape[1],
            seq_len=X.shape[2],
            dropout=cfg.dropout,
        ).to(device)

        model, val_auc, thresh = train_fold(
            model, train_ds, val_ds, cfg, device, fold, model_name
        )
        fold_thresholds.append(thresh)

        # Cross-fold threshold: mean of all OTHER folds' Youden thresholds
        other_thresholds = [t for j, t in enumerate(fold_thresholds)
                            if j != fold_idx]
        test_thresh = (float(np.mean(other_thresholds))
                       if other_thresholds else thresh)

        # ── test evaluation ────────────────────────────────────────────────
        model.eval()
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size * 2,
                                 shuffle=False, num_workers=2)
        all_prob, all_true = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                prob = F.softmax(model(xb.to(device)).float(), dim=1)[:, 1]
                all_prob.append(prob.cpu().numpy())
                all_true.append(yb.numpy())

        y_prob = np.concatenate(all_prob)
        y_true = np.concatenate(all_true)
        metrics = compute_metrics(y_true, y_prob, test_thresh)
        metrics["val_auc"] = val_auc
        fold_metrics.append(metrics)

        tag = f"{model_name}_fold{fold}"
        _save_roc(y_true, y_prob, fig_dir / f"roc_{tag}.png", test_thresh, tag)
        _save_pr( y_true, y_prob, fig_dir / f"pr_{tag}.png",  tag)
        _save_cm( y_true, (y_prob >= test_thresh).astype(int),
                  fig_dir / f"cm_{tag}.png", tag)

        torch.save(
            {"state_dict": model.state_dict(),
             "metrics": metrics,
             "threshold": test_thresh,
             "fold": fold,
             "model": model_name},
            ckpt_dir / f"best_fold{fold}.pt",
        )
        log.info("  Fold %d test: AUC=%.4f  Sens=%.4f  Spec=%.4f  F1=%.4f",
                 fold, metrics["auc_roc"], metrics["sensitivity"],
                 metrics["specificity"], metrics["f1"])

    # ── aggregate results ──────────────────────────────────────────────────
    keys = ["auc_roc", "auc_pr", "sensitivity", "specificity", "f1", "accuracy"]
    summary: dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"]  = float(np.std(vals))

    log.info("─" * 60)
    log.info("%s — 5-fold mean results:", model_name.upper())
    for k in keys:
        log.info("  %-14s %.4f ± %.4f",
                 k, summary[f"{k}_mean"], summary[f"{k}_std"])

    if use_mlflow and _MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=f"baseline_{model_name}", nested=True):
            mlflow.log_params({"model": model_name, "seed": seed})
            mlflow.log_metrics(summary)

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(out_dir: Path) -> None:
    log_path = out_dir / "logs" / f"baselines_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    # Use the module logger directly instead of basicConfig, which is a no-op
    # if any library (e.g. MLflow) already attached root handlers before main().
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove any handlers that libraries attached before us so we own the format
    for h in root.handlers[:]:
        root.removeHandler(h)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(str(log_path), mode="w")
    fh.setFormatter(fmt)

    root.addHandler(sh)
    root.addHandler(fh)

    # Flush file handler on every record so the log is never empty on crash
    fh.setLevel(logging.DEBUG)
    sh.setLevel(logging.DEBUG)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train baseline models for FallDetectionNet comparison"
    )
    p.add_argument("--model",   type=str, default="all",
                   help="Model name or 'all'. "
                        "Choices: bilstm tcn stgcn ctrgcn skateformer all")
    p.add_argument("--config",  type=str, default="research_config.json")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--device",  type=str, default=None,
                   help="cuda or cpu (default: auto-detect)")
    p.add_argument("--debug",   action="store_true",
                   help="Use 10%% of data for fast iteration")
    p.add_argument("--no-mlflow", dest="no_mlflow", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Skip folds whose checkpoint already exists; "
                        "skip models where all 5 fold checkpoints are present")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    cfg    = load_config(args.config if Path(args.config).exists() else None)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)

    # Disable cuDNN auto-tuner before any CUDA work to prevent the one-time
    # CUDNN_STATUS_NOT_SUPPORTED plan-selection warning on Ada GPUs.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark     = False
        torch.backends.cudnn.deterministic = True

    setup_logging(out_dir)
    set_seed(args.seed)
    log.info("Device: %s", device)
    log.info("Config loaded from: %s",
             args.config if Path(args.config).exists() else "defaults")

    X, y, seq_ids = load_stage_b_data(cfg, debug=args.debug)

    use_mlflow = not args.no_mlflow and _MLFLOW_AVAILABLE
    if use_mlflow:
        mlflow.set_experiment("FallDetectionNet_Baselines")

    models_to_run = ALL_MODELS if args.model == "all" else [args.model]
    all_summaries: dict[str, dict[str, float]] = {}

    with (mlflow.start_run(run_name="baselines") if use_mlflow
          else _null_context()):
        for name in models_to_run:
            summary = run_baseline(
                name, X, y, seq_ids, cfg, device,
                args.seed, out_dir, use_mlflow,
                resume=args.resume,
            )
            all_summaries[name] = summary

    # ── print final comparison table ───────────────────────────────────────
    log.info("")
    log.info("=" * 72)
    log.info("BASELINE COMPARISON TABLE")
    log.info("=" * 72)
    header = f"{'Model':<14} {'AUC-ROC':>10} {'AUC-PR':>10} "  \
             f"{'Sens':>8} {'Spec':>8} {'F1':>8}"
    log.info(header)
    log.info("-" * 72)
    for name, s in all_summaries.items():
        log.info(
            "%-14s %6.4f±%5.4f %6.4f±%5.4f %6.4f±%5.4f "
            "%6.4f±%5.4f %6.4f±%5.4f",
            name,
            s["auc_roc_mean"],     s["auc_roc_std"],
            s["auc_pr_mean"],      s["auc_pr_std"],
            s["sensitivity_mean"], s["sensitivity_std"],
            s["specificity_mean"], s["specificity_std"],
            s["f1_mean"],          s["f1_std"],
        )
    log.info("=" * 72)


# ── tiny context manager so the mlflow block works without MLflow ─────────────
from contextlib import contextmanager

@contextmanager
def _null_context():
    yield


if __name__ == "__main__":
    main()
