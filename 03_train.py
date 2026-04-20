# fmt: off
# ──────────────────────────────────────────────────────────────────────────────
# Requirements (tested versions):
#   torch==2.2.0  torchvision==0.17.0  numpy==2.2.6  scikit-learn==1.4.0
#   mlflow==2.10.0  matplotlib==3.8.3  seaborn==0.13.2  scipy==1.12.0
#   tqdm==4.66.2
# ──────────────────────────────────────────────────────────────────────────────
"""
03_train.py — FallDetectionNet: Dilated Spatial-Temporal Pose-Based Fall Detection
===================================================================================

Stage A: Multi-class NTU RGB+D 120 pretraining across selected action classes.
         Builds rich motion representations before the model sees any fall data.

Stage B: Binary fine-tuning on URFD + Le2i with two variants:
         Variant A — frozen backbone, head-only training.
         Variant B — full fine-tuning with differential learning rates.

Usage examples:
    # Stage A pretraining
    python 03_train.py --stage a --config research_config.json --seed 42

    # Stage B Variant A (frozen backbone)
    python 03_train.py --stage b --variant a --resume outputs/checkpoints/stage_a_best.pt

    # Stage B Variant B (full fine-tuning)
    python 03_train.py --stage b --variant b --resume outputs/checkpoints/stage_a_best.pt

    # Debug run (10 % of data, no MLflow)
    python 03_train.py --stage b --variant a --debug --no-mlflow
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import math
import os
import random
import re
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

# ── optional MLflow ───────────────────────────────────────────────────────────
try:
    import mlflow
    import mlflow.pytorch
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

# ── module logger (handlers added in setup_logging) ──────────────────────────
log = logging.getLogger("falldect")

# ──────────────────────────────────────────────────────────────────────────────
# NTU RGB+D joint mapping  (NTU 25-joint → COCO 17-joint)
# ──────────────────────────────────────────────────────────────────────────────
# For each COCO joint index i, NTU_JOINT_MAP[i] is the NTU source joint.
NTU_JOINT_MAP: list[int] = [
    3,   # 0  nose       ← head
    3,   # 1  l_eye      ← head
    3,   # 2  r_eye      ← head
    3,   # 3  l_ear      ← head
    3,   # 4  r_ear      ← head
    4,   # 5  l_shoulder
    8,   # 6  r_shoulder
    5,   # 7  l_elbow
    9,   # 8  r_elbow
    6,   # 9  l_wrist
    10,  # 10 r_wrist
    12,  # 11 l_hip
    16,  # 12 r_hip
    13,  # 13 l_knee
    17,  # 14 r_knee
    14,  # 15 l_ankle
    18,  # 16 r_ankle
]

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AugConfig:
    joint_dropout_prob: float = 0.1
    noise_sigma: float = 0.01
    temporal_flip_prob: float = 0.3


@dataclass
class FocalConfig:
    gamma: float = 2.0
    alpha_cap: float = 2.5
    label_smoothing: float = 0.05


@dataclass
class StageAConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 15
    ntu_stride: int = 5


@dataclass
class StageBVariantAConfig:
    lr_head: float = 1e-3
    num_epochs: int = 30
    early_stopping_patience: int = 10


@dataclass
class StageBVariantBConfig:
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    num_epochs: int = 80
    early_stopping_patience: int = 15


@dataclass
class TrainConfig:
    dataset_root: str = "DATASET"
    output_dir: str = "outputs"
    processed_dir: str = "processed"

    window_size: int = 30
    stride_fall: int = 5
    stride_adl: int = 10
    stride_onset: int = 3
    smooth_k: int = 3         # temporal smoothing: require k consecutive fall windows
    onset_half_window: int = 10

    ntu_selected_actions: list[int] = field(default_factory=lambda: [7,8,9,11,43,44,45,46,47,48])
    ntu_action_names: dict[str, str] = field(default_factory=dict)

    in_features: int = 7
    num_joints: int = 17
    joint_embed_dim: int = 64
    backbone_channels: int = 256
    dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    dropout: float = 0.4

    num_folds: int = 5
    batch_size: int = 32
    accum_steps: int = 4
    grad_clip: float = 1.0

    stage_a: StageAConfig = field(default_factory=StageAConfig)
    stage_b_variant_a: StageBVariantAConfig = field(default_factory=StageBVariantAConfig)
    stage_b_variant_b: StageBVariantBConfig = field(default_factory=StageBVariantBConfig)
    focal_loss: FocalConfig = field(default_factory=FocalConfig)
    augmentation: AugConfig = field(default_factory=AugConfig)

    le2i_scenes: list[list[str]] = field(default_factory=lambda: [
        ["Coffee_room_01", "Coffee_room_01", "Annotation_files"],
        ["Coffee_room_02", "Coffee_room_02", "Annotations_files"],
        ["Home_01",        "Home_01",        "Annotation_files"],
        ["Home_02",        "Home_02",        "Annotation_files"],
    ])


def load_config(path: str | None) -> TrainConfig:
    """Load TrainConfig from a JSON file, falling back to defaults if path is None.

    Args:
        path: Path to research_config.json, or None to use all defaults.

    Returns:
        Populated TrainConfig instance.
    """
    cfg = TrainConfig()
    if path is None:
        return cfg
    with open(path) as f:
        raw = json.load(f)

    simple_fields = [
        "dataset_root", "output_dir", "processed_dir",
        "window_size", "stride_fall", "stride_adl", "stride_onset", "onset_half_window",
        "ntu_selected_actions", "ntu_action_names",
        "in_features", "num_joints", "joint_embed_dim", "backbone_channels",
        "dilations", "dropout", "num_folds", "batch_size", "accum_steps", "grad_clip",
        "le2i_scenes",
    ]
    for key in simple_fields:
        if key in raw:
            setattr(cfg, key, raw[key])

    def _update(dc: Any, d: dict) -> None:
        for k, v in d.items():
            if hasattr(dc, k):
                setattr(dc, k, v)

    if "stage_a" in raw:
        _update(cfg.stage_a, raw["stage_a"])
    if "stage_b_variant_a" in raw:
        _update(cfg.stage_b_variant_a, raw["stage_b_variant_a"])
    if "stage_b_variant_b" in raw:
        _update(cfg.stage_b_variant_b, raw["stage_b_variant_b"])
    if "focal_loss" in raw:
        _update(cfg.focal_loss, raw["focal_loss"])
    if "augmentation" in raw:
        _update(cfg.augmentation, raw["augmentation"])

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, PyTorch, and CUDA for full reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction utilities
# ──────────────────────────────────────────────────────────────────────────────

def compute_joint_features(coords: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Compute 7 kinematic features per joint per frame.

    The 7 features are: x, y, vx, vy, speed, ax, ay.
    All spatial values should be pre-normalised to [0, 1].

    Args:
        coords: (T, J, 2) normalised [x, y] coordinates.
        vel:    (T, J, 2) per-frame [vx, vy] velocity (finite-difference of coords).

    Returns:
        features: (7, T, J) float32 array ready for JointEmbedding.
    """
    x  = coords[:, :, 0]                        # (T, J)
    y  = coords[:, :, 1]                        # (T, J)
    vx = vel[:, :, 0]                           # (T, J)
    vy = vel[:, :, 1]                           # (T, J)
    speed = np.sqrt(vx ** 2 + vy ** 2)         # (T, J)
    ax = np.zeros_like(vx)
    ay = np.zeros_like(vy)
    ax[1:] = vx[1:] - vx[:-1]
    ay[1:] = vy[1:] - vy[:-1]
    feat = np.stack([x, y, vx, vy, speed, ax, ay], axis=0)  # (7, T, J)
    return feat.astype(np.float32)


def npz_window_to_joint_features(window: np.ndarray) -> np.ndarray:
    """Convert a processed-pipeline window to joint feature tensor.

    The pipeline stores windows of shape (T, 75) where:
      - indices  0-33: 17 joint (x, y) coordinates (flattened)
      - indices 34-40: 7 global derived features (unused here)
      - indices 41-74: 17 joint (vx, vy) velocities (flattened)

    Args:
        window: (T, 75) float32 array from processed .npz file.

    Returns:
        (7, T, 17) float32 joint feature tensor.
    """
    T = window.shape[0]
    coords = window[:, 0:34].reshape(T, 17, 2)   # (T, 17, 2)
    vel    = window[:, 41:75].reshape(T, 17, 2)  # (T, 17, 2)
    return compute_joint_features(coords, vel)


def ntu_skeleton_to_joint_features(joints_25: np.ndarray) -> np.ndarray:
    """Convert a single NTU 25-joint frame to normalised 17-joint coordinates.

    Args:
        joints_25: (T, 25, 3) array with x, y, z in camera space.

    Returns:
        coords_norm: (T, 17, 2) normalised [x, y] coordinates in [0, 1].
    """
    T = joints_25.shape[0]
    xy = joints_25[:, NTU_JOINT_MAP, :2].astype(np.float32)  # (T, 17, 2)
    # Normalise within bounding box per frame
    x_min = xy[:, :, 0].min(axis=1, keepdims=True)           # (T, 1)
    x_max = xy[:, :, 0].max(axis=1, keepdims=True)
    y_min = xy[:, :, 1].min(axis=1, keepdims=True)
    y_max = xy[:, :, 1].max(axis=1, keepdims=True)
    bw = np.maximum(x_max - x_min, 1e-6)
    bh = np.maximum(y_max - y_min, 1e-6)
    xy_norm = np.stack([
        (xy[:, :, 0] - x_min) / bw,
        (xy[:, :, 1] - y_min) / bh,
    ], axis=2)                                                # (T, 17, 2)
    return xy_norm


def parse_ntu_skeleton(path: Path) -> np.ndarray | None:
    """Parse an NTU .skeleton file into a joint array.

    Args:
        path: Path to the .skeleton file.

    Returns:
        Array of shape (T, 25, 3) with [x, y, z] per joint, or None on error.
    """
    try:
        lines = path.read_text().strip().splitlines()
    except Exception:
        return None
    try:
        idx = 0
        n_frames = int(lines[idx]); idx += 1
        frames: list[np.ndarray] = []
        for _ in range(n_frames):
            n_bodies = int(lines[idx]); idx += 1
            body_joints: np.ndarray | None = None
            for _ in range(n_bodies):
                idx += 1                              # body info line
                n_joints = int(lines[idx]); idx += 1
                joints = np.zeros((n_joints, 3), dtype=np.float32)
                for j in range(n_joints):
                    vals = lines[idx].split(); idx += 1
                    joints[j, 0] = float(vals[0])
                    joints[j, 1] = float(vals[1])
                    joints[j, 2] = float(vals[2])
                if body_joints is None:
                    body_joints = joints
            if body_joints is not None:
                frames.append(body_joints)
        if not frames:
            return None
        return np.stack(frames, axis=0)               # (T, 25, 3)
    except (IndexError, ValueError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────────────────────────────────────

class JointEmbedding(nn.Module):
    """Project each joint's kinematic features into an embedding space.

    Uses a pointwise 2-D convolution (kernel=1) so that the same linear
    transform is applied independently to every (time-step, joint) position.

    Args:
        in_features:  Number of input kinematic features per joint (default 7).
        embed_dim:    Output embedding dimension per joint (default 64).
    """

    def __init__(self, in_features: int = 7, embed_dim: int = 64) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, embed_dim, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, in_features, T, J) input tensor.

        Returns:
            (B, embed_dim, T, J) embedded tensor after BN + GELU.
        """
        return F.gelu(self.bn(self.conv(x)))


class DilatedTemporalBlock(nn.Module):
    """Depthwise-separable dilated 1-D convolution block along the time axis.

    Implements an inverted-residual structure:
        pointwise expand (C → 2C) → depthwise dilated conv → pointwise project (2C → C)
        → add residual → GELU

    The joint dimension J is folded into the batch dimension before temporal
    processing and restored afterwards, so the block is agnostic to J.

    Args:
        channels: Number of input (and output) channels C.
        dilation: Dilation factor for the temporal convolution.
    """

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        mid = channels * 2
        self.expand  = nn.Conv1d(channels, mid, kernel_size=1, bias=False)
        self.bn_exp  = nn.BatchNorm1d(mid)
        self.dw_conv = nn.Conv1d(
            mid, mid, kernel_size=3,
            padding=dilation, dilation=dilation, groups=mid, bias=False,
        )
        self.bn_dw   = nn.BatchNorm1d(mid)
        self.project = nn.Conv1d(mid, channels, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, T, J) feature tensor.

        Returns:
            (B, C, T, J) tensor with same shape, residual connection applied.
        """
        B, C, T, J = x.shape
        # Fold J into batch for Conv1d: (B*J, C, T)
        h = x.permute(0, 3, 1, 2).reshape(B * J, C, T)
        residual = h

        h = F.gelu(self.bn_exp(self.expand(h)))     # (B*J, 2C, T)
        h = F.gelu(self.bn_dw(self.dw_conv(h)))     # (B*J, 2C, T)
        h = self.bn_proj(self.project(h))            # (B*J,  C, T)
        h = F.gelu(h + residual)                     # residual + GELU

        # Restore: (B, C, T, J)
        return h.reshape(B, J, C, T).permute(0, 2, 3, 1).contiguous()


class JointAttention(nn.Module):
    """Per-timestep soft attention over the joint dimension.

    For each time step, a small MLP derives attention weights across J joints.
    The query is formed by averaging over the channel dimension, providing a
    compact representation of the current temporal context.

    Args:
        channels:   Number of feature channels C.
        num_joints: Number of skeleton joints J (default 17).
    """

    def __init__(self, channels: int, num_joints: int = 17) -> None:
        super().__init__()
        mid = max(channels // 4, num_joints)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, num_joints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, T, J) feature tensor.

        Returns:
            (B, C, T, J) tensor re-weighted by joint attention.
        """
        # Pool over channels → (B, T, C) query
        q = x.mean(dim=-1).permute(0, 2, 1)          # (B, T, C)
        attn = F.gelu(self.fc1(q))                    # (B, T, mid)
        attn = self.fc2(attn).softmax(dim=-1)         # (B, T, J)
        attn = attn.unsqueeze(1)                      # (B, 1, T, J)
        return x * attn


class MultiScaleTemporalPooling(nn.Module):
    """Aggregate temporal features at three scales and concatenate.

    Scales: full sequence, half sequence (adaptive), quarter sequence (adaptive).
    Joint dimension is averaged before temporal pooling.

    Returns a (B, 3*C) vector capturing both fine-grained and coarse motion.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, T, J) feature tensor.

        Returns:
            (B, 3*C) multi-scale pooled vector.
        """
        # Average over joints first
        x_t = x.mean(dim=-1)                          # (B, C, T)
        T = x_t.shape[-1]

        s1 = x_t.mean(dim=-1)                                                   # (B, C)
        s2 = F.adaptive_avg_pool1d(x_t, max(1, T // 2)).mean(dim=-1)           # (B, C)
        s3 = F.adaptive_avg_pool1d(x_t, max(1, T // 4)).mean(dim=-1)           # (B, C)

        return torch.cat([s1, s2, s3], dim=-1)        # (B, 3C)


class FallDetectionNet(nn.Module):
    """Full spatial-temporal fall detection network.

    Architecture:
        JointEmbedding → channel projection → DilatedTemporalBlocks
        (with JointAttention after every 2 blocks) → MultiScaleTemporalPooling
        → classification head.

    Args:
        num_classes:       Number of output classes (K for Stage A, 2 for Stage B).
        joint_embed_dim:   JointEmbedding output dimension (default 64).
        backbone_channels: Temporal backbone channel width (default 256).
        dilations:         Dilation sequence for DilatedTemporalBlocks.
        dropout:           Dropout rate in the classification head (default 0.4).
        num_joints:        Number of skeleton joints (default 17).
        in_features:       Kinematic features per joint (default 7).
    """

    def __init__(
        self,
        num_classes: int,
        joint_embed_dim: int = 64,
        backbone_channels: int = 256,
        dilations: list[int] | None = None,
        dropout: float = 0.4,
        num_joints: int = 17,
        in_features: int = 7,
    ) -> None:
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]

        self._backbone_channels = backbone_channels
        self._dropout = dropout

        self.joint_embed = JointEmbedding(in_features, joint_embed_dim)
        self.channel_proj = nn.Sequential(
            nn.Conv2d(joint_embed_dim, backbone_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(backbone_channels),
        )

        # Interleave DilatedTemporalBlocks with JointAttention every 2 blocks
        layers: list[nn.Module] = []
        for i, d in enumerate(dilations):
            layers.append(DilatedTemporalBlock(backbone_channels, d))
            if (i + 1) % 2 == 0:
                layers.append(JointAttention(backbone_channels, num_joints))
        self.backbone = nn.ModuleList(layers)

        self.pool = MultiScaleTemporalPooling()
        self.head = self._build_head(3 * backbone_channels, num_classes, dropout)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_head(in_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, in_features, T, J) input joint feature tensor.

        Returns:
            (B, num_classes) raw logits.
        """
        x = F.gelu(self.channel_proj(self.joint_embed(x)))  # (B, C, T, J)
        for module in self.backbone:
            x = module(x)
        x = self.pool(x)    # (B, 3C)
        return self.head(x)

    # ------------------------------------------------------------------
    def get_backbone_params(self) -> list[nn.Parameter]:
        """Return backbone parameters only (excludes the classification head).

        Used to set differential learning rates in Stage B Variant B.

        Returns:
            Flat list of all backbone Parameter objects.
        """
        backbone_modules = [self.joint_embed, self.channel_proj,
                            self.backbone, self.pool]
        params: list[nn.Parameter] = []
        for m in backbone_modules:
            params.extend(m.parameters())
        return params

    # ------------------------------------------------------------------
    def replace_head(self, num_classes: int, dropout: float | None = None) -> None:
        """Replace the classification head for fine-tuning.

        Args:
            num_classes: Number of output classes for the new head.
            dropout:     Dropout rate; if None, reuses the original rate.
        """
        dr = dropout if dropout is not None else self._dropout
        self.head = self._build_head(3 * self._backbone_channels, num_classes, dr)


# ──────────────────────────────────────────────────────────────────────────────
# Loss Function
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Binary focal loss with label smoothing and dynamic alpha.

    Alpha is computed from inverse class frequency of the training split and
    capped at ``alpha_cap`` to prevent gradient instability on very imbalanced
    data.  Label smoothing is applied to the target distribution before the
    focal weight is computed.

    Args:
        alpha:     Weight for the positive (fall) class.
        gamma:     Focusing exponent (default 2.0).
        eps:       Label smoothing epsilon (default 0.05).
        alpha_cap: Maximum allowed alpha value (default 2.5).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        eps: float = 0.05,
        alpha_cap: float = 2.5,
    ) -> None:
        super().__init__()
        self.alpha = min(float(alpha), alpha_cap)
        self.gamma = gamma
        self.eps   = eps

    # ------------------------------------------------------------------
    @classmethod
    def from_labels(
        cls,
        y: np.ndarray,
        gamma: float = 2.0,
        eps: float = 0.05,
        alpha_cap: float = 2.5,
    ) -> "FocalLoss":
        """Construct FocalLoss with alpha derived from inverse class frequency.

        Args:
            y:         Integer label array for the training split.
            gamma:     Focusing exponent.
            eps:       Label smoothing epsilon.
            alpha_cap: Maximum allowed alpha value.

        Returns:
            FocalLoss instance with calibrated alpha.
        """
        n_pos = max(int((y == 1).sum()), 1)
        alpha = len(y) / (2.0 * n_pos)
        return cls(alpha=alpha, gamma=gamma, eps=eps, alpha_cap=alpha_cap)

    # ------------------------------------------------------------------
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits:  (B, 2) raw class logits from the model.
            targets: (B,)  integer labels {0, 1}.

        Returns:
            Scalar mean focal loss.
        """
        p_fall = F.softmax(logits.float(), dim=1)[:, 1]   # (B,) — fp32 throughout
        t = targets.float()

        # Label smoothing: 1 → 0.95, 0 → 0.05
        smooth_t = t * (1.0 - self.eps) + (1.0 - t) * self.eps

        # Manual BCE: avoids F.binary_cross_entropy which is blocked inside autocast
        p_clamped = p_fall.clamp(1e-7, 1.0 - 1e-7)
        bce = -(smooth_t * torch.log(p_clamped) + (1.0 - smooth_t) * torch.log(1.0 - p_clamped))

        # Focal weight based on original (unsmoothed) targets
        pt = torch.where(targets.bool(), p_fall, 1.0 - p_fall)
        alpha_t = torch.where(
            targets.bool(),
            torch.full_like(p_fall, self.alpha),
            torch.ones_like(p_fall),
        )
        focal_w = alpha_t * (1.0 - pt).pow(self.gamma)

        return (focal_w * bce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class SkeletonDataset(Dataset):
    """PyTorch Dataset for skeleton-based action/fall sequences.

    Applies data augmentation during training:
      - Joint dropout: zero out random joints with per-joint probability.
      - Gaussian noise: additive noise on the x, y coordinate channels.
      - Temporal flip: reverse the time axis for non-fall sequences only.

    Args:
        X:        (N, 7, T, J) float32 feature array.
        y:        (N,)         integer label array.
        seq_ids:  (N,)         integer sequence-level identifiers for CV splits.
        is_train: If True, augmentation is applied in __getitem__.
        aug_cfg:  AugConfig with augmentation hyper-parameters.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_ids: np.ndarray,
        is_train: bool = False,
        aug_cfg: AugConfig | None = None,
    ) -> None:
        self.X       = torch.from_numpy(X).float()
        self.y       = torch.from_numpy(y).long()
        self.seq_ids = seq_ids
        self.is_train = is_train
        self.aug_cfg  = aug_cfg or AugConfig()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()   # (7, T, J)
        y = self.y[idx]
        if self.is_train:
            x = self._augment(x, y)
        return x, y

    # ------------------------------------------------------------------
    def _augment(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply training-time augmentations.

        Args:
            x: (7, T, J) sample tensor.
            y: Scalar label tensor.

        Returns:
            Augmented (7, T, J) tensor.
        """
        cfg = self.aug_cfg
        J = x.shape[-1]

        # 1. Joint dropout: zero out each joint independently
        mask = torch.rand(J) >= cfg.joint_dropout_prob       # (J,) True = keep
        x = x * mask.unsqueeze(0).unsqueeze(0)               # (7, T, J)

        # 2. Gaussian noise on x, y coordinate channels (indices 0, 1)
        x[:2] = x[:2] + torch.randn_like(x[:2]) * cfg.noise_sigma

        # 3. Temporal flip — only for non-fall samples
        if y.item() != 1 and torch.rand(1).item() < cfg.temporal_flip_prob:
            x = x.flip(dims=[1])                             # reverse T

        return x

    # ------------------------------------------------------------------
    def class_weights_for_sampler(self) -> torch.Tensor:
        """Compute per-sample weights for WeightedRandomSampler.

        Fall samples receive weight proportional to their inverse class
        frequency, approximating fall-onset oversampling.

        Returns:
            (N,) float tensor of per-sample sampling weights.
        """
        labels = self.y.numpy()
        n_fall    = max((labels == 1).sum(), 1)
        n_nonfall = max((labels == 0).sum(), 1)
        w_fall    = len(labels) / (2.0 * n_fall)
        w_nonfall = len(labels) / (2.0 * n_nonfall)
        weights = np.where(labels == 1, w_fall, w_nonfall)
        return torch.from_numpy(weights).float()


# ──────────────────────────────────────────────────────────────────────────────
# DataModule
# ──────────────────────────────────────────────────────────────────────────────

class DataModule:
    """Manages dataset loading, sequence-level CV splits, and DataLoader creation.

    Stage A loads NTU RGB+D 120 skeleton files for multi-class pretraining.
    Stage B loads processed .npz files (URFD + Le2i) for binary fine-tuning.

    Sequence IDs are reconstructed from the filesystem so that every window
    from the same source sequence lands in exactly one CV fold, preventing
    data leakage across the train/validation/test boundaries.

    Args:
        cfg:      Populated TrainConfig.
        device:   Torch device string (e.g. 'cuda' or 'cpu').
        debug:    If True, use only 10 % of available windows.
    """

    def __init__(self, cfg: TrainConfig, device: str, debug: bool = False) -> None:
        self.cfg    = cfg
        self.device = device
        self.debug  = debug

    # ── Stage B ──────────────────────────────────────────────────────────────

    def setup_stage_b(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and concatenate URFD and Le2i processed windows.

        Returns:
            X:       (N, 7, T, J) joint feature array.
            y:       (N,)          binary label array.
            seq_ids: (N,)          sequence-level integer IDs (unique per source video).

        Raises:
            FileNotFoundError: If processed .npz files are missing.
        """
        proc = Path(self.cfg.processed_dir)

        def _load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
            p = proc / name / "sequences.npz"
            if not p.exists():
                raise FileNotFoundError(f"Processed file not found: {p}")
            d = np.load(str(p))
            seq_ids = d["seq_ids"].astype(np.int64) if "seq_ids" in d else None
            return d["X"].astype(np.float32), d["y"].astype(np.int64), seq_ids

        X_u, y_u, ids_u = _load("urfd")
        X_l, y_l, ids_l = _load("le2i")

        if ids_u is None:
            log.info("Reconstructing URFD sequence IDs …")
            ids_u = self._urfd_seq_ids(len(X_u))
            p = proc / "urfd" / "sequences.npz"
            d = np.load(str(p))
            np.savez_compressed(str(p), **dict(d), seq_ids=ids_u)
            log.info("Saved seq_ids into %s", p)

        if ids_l is None:
            log.info("Reconstructing Le2i sequence IDs …")
            ids_l = self._le2i_seq_ids(len(X_l), offset=int(ids_u.max()) + 1)
            p = proc / "le2i" / "sequences.npz"
            d = np.load(str(p))
            np.savez_compressed(str(p), **dict(d), seq_ids=ids_l)
            log.info("Saved seq_ids into %s", p)

        # Convert (N, T, 75) → (N, 7, T, 17)
        log.info("Converting URFD windows to joint features …")
        X_u_jf = np.stack([npz_window_to_joint_features(X_u[i]) for i in range(len(X_u))])
        log.info("Converting Le2i windows to joint features …")
        X_l_jf = np.stack([npz_window_to_joint_features(X_l[i]) for i in range(len(X_l))])

        X       = np.concatenate([X_u_jf, X_l_jf], axis=0)
        y       = np.concatenate([y_u, y_l],        axis=0)
        seq_ids = np.concatenate([ids_u, ids_l],    axis=0)

        if self.debug:
            n = max(100, len(y) // 10)
            rng = np.random.default_rng(0)
            idx = rng.choice(len(y), n, replace=False)
            X, y, seq_ids = X[idx], y[idx], seq_ids[idx]
            log.info("Debug mode: using %d / %d windows", n, len(y))

        self._verify_seq_ids(seq_ids)
        log.info("Stage B data: %d windows  fall=%d  non-fall=%d",
                 len(y), (y == 1).sum(), (y == 0).sum())
        return X, y, seq_ids

    # ── Stage A ──────────────────────────────────────────────────────────────

    def setup_stage_a(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Load or build multi-class NTU RGB+D 120 pretraining data.

        Returns:
            X:           (N, 7, T, J) joint feature array.
            y:           (N,)          class label array (0 … K-1).
            seq_ids:     (N,)          sequence-level IDs.
            num_classes: Number of distinct action classes K.

        Raises:
            FileNotFoundError: If NTU skeleton directory is missing and no
                               cache is found.
        """
        cache_path = Path(self.cfg.processed_dir) / "ntu" / "stage_a_cache.npz"
        if cache_path.exists():
            log.info("Loading Stage A cache from %s", cache_path)
            d = np.load(str(cache_path))
            X, y, seq_ids = d["X"], d["y"], d["seq_ids"]
            num_classes = int(y.max()) + 1
        else:
            X, y, seq_ids, num_classes = self._build_ntu_multiclass()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(cache_path), X=X, y=y, seq_ids=seq_ids)
            log.info("Saved Stage A cache → %s", cache_path)

        if self.debug:
            n = max(200, len(y) // 10)
            rng = np.random.default_rng(0)
            idx = rng.choice(len(y), n, replace=False)
            X, y, seq_ids = X[idx], y[idx], seq_ids[idx]
            num_classes = int(y.max()) + 1

        log.info("Stage A data: %d windows  %d classes", len(y), num_classes)
        return X.astype(np.float32), y.astype(np.int64), seq_ids.astype(np.int64), num_classes

    def _build_ntu_multiclass(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Build multi-class NTU dataset from raw skeleton files.

        Returns:
            X, y, seq_ids, num_classes.

        Raises:
            FileNotFoundError: If the NTU skeleton directory does not exist.
        """
        ntu_root = Path(self.cfg.dataset_root) / "NTU"
        if not ntu_root.exists():
            raise FileNotFoundError(f"NTU directory not found: {ntu_root}")

        selected  = sorted(self.cfg.ntu_selected_actions)
        action_to_cls = {a: i for i, a in enumerate(selected)}
        num_classes   = len(selected)
        W  = self.cfg.window_size
        ST = self.cfg.stage_a.ntu_stride

        all_X: list[np.ndarray] = []
        all_y: list[int]        = []
        all_ids: list[int]      = []
        seq_id = 0

        skeletons = sorted(ntu_root.glob("*.skeleton"))
        log.info("NTU Stage A: scanning %d skeleton files …", len(skeletons))

        for sf in tqdm(skeletons, desc="NTU Stage A", leave=False):
            m = re.search(r"A(\d{3})", sf.name)
            if not m:
                continue
            action_id = int(m.group(1))
            if action_id not in action_to_cls:
                continue

            joints = parse_ntu_skeleton(sf)   # (T, 25, 3) or None
            if joints is None or len(joints) < W:
                continue

            coords = ntu_skeleton_to_joint_features(joints)  # (T, 17, 2)
            vel    = np.zeros_like(coords)
            vel[1:] = coords[1:] - coords[:-1]
            feats  = compute_joint_features(coords, vel)     # (7, T, 17)

            T = feats.shape[1]
            cls_label = action_to_cls[action_id]
            for start in range(0, T - W + 1, ST):
                all_X.append(feats[:, start:start + W, :])
                all_y.append(cls_label)
                all_ids.append(seq_id)
            seq_id += 1

        if not all_X:
            raise RuntimeError("No NTU windows produced — check selected_actions and path.")

        X = np.stack(all_X, axis=0).astype(np.float32)
        y = np.array(all_y,   dtype=np.int64)
        s = np.array(all_ids, dtype=np.int64)
        log.info("Built Stage A: %d windows from %d sequences", len(y), seq_id)
        return X, y, s, num_classes

    # ── Sequence ID reconstruction (Stage B) ─────────────────────────────────

    def _urfd_seq_ids(self, expected_n: int) -> np.ndarray:
        """Reconstruct per-window sequence IDs for URFD.

        Scans the source directories (fall/, adl/) and counts frames per
        sequence to determine how many windows each contributes.

        Args:
            expected_n: Expected total number of URFD windows.

        Returns:
            (N,) integer array where N == expected_n if reconstruction succeeds,
            otherwise a fallback sequential array with a warning.
        """
        cfg = self.cfg
        W   = cfg.window_size
        urfd_root = Path(cfg.dataset_root) / "URFD"
        ids: list[int] = []
        seq_id = 0

        for subdir, stride in [("fall", cfg.stride_fall), ("adl", cfg.stride_adl)]:
            sub_path = urfd_root / subdir
            if not sub_path.exists():
                continue
            for seq_dir in sorted(sub_path.iterdir()):
                if not seq_dir.is_dir():
                    continue
                n_frames = len(list(seq_dir.glob("*.png")))
                n_win    = max(0, (n_frames - W) // stride + 1)
                ids.extend([seq_id] * n_win)
                seq_id += 1

        if len(ids) != expected_n:
            log.warning(
                "URFD seq ID mismatch (got %d, expected %d) — falling back to sequential IDs",
                len(ids), expected_n,
            )
            return np.arange(expected_n, dtype=np.int64)
        return np.array(ids, dtype=np.int64)

    def _le2i_seq_ids(self, expected_n: int, offset: int = 10000) -> np.ndarray:
        """Reconstruct per-window sequence IDs for Le2i.

        Reads annotation files to determine total frame count per video.

        Args:
            expected_n: Expected total number of Le2i windows.
            offset:     Starting ID value to avoid collision with URFD IDs.

        Returns:
            (N,) integer array where N == expected_n if reconstruction succeeds.
        """
        cfg = self.cfg
        W   = cfg.window_size
        le2i_root = Path(cfg.dataset_root) / "LE2I"
        ids: list[int] = []
        seq_id = offset

        for scene_dir, inner_dir, ann_subdir in cfg.le2i_scenes:
            ann_dir   = le2i_root / scene_dir / inner_dir / ann_subdir
            video_dir = le2i_root / scene_dir / inner_dir / "Videos"
            if not video_dir.exists():
                continue

            try:
                video_files = sorted(
                    video_dir.glob("*.avi"),
                    key=lambda p: int(re.search(r"\((\d+)\)", p.name).group(1)),
                )
            except (AttributeError, TypeError):
                continue

            for vf in video_files:
                ann_file = ann_dir / (vf.stem + ".txt")
                if not ann_file.exists():
                    continue
                lines = ann_file.read_text(errors="replace").strip().splitlines()
                if len(lines) < 2:
                    continue
                total_frames = len(lines) - 2      # header is 2 lines
                # pipeline used stride=10 for all Le2i windows
                n_win = max(0, (total_frames - W) // 10 + 1)
                ids.extend([seq_id] * n_win)
                seq_id += 1

        if len(ids) != expected_n:
            log.warning(
                "Le2i seq ID mismatch (got %d, expected %d) — falling back",
                len(ids), expected_n,
            )
            return np.arange(expected_n, dtype=np.int64) + offset
        return np.array(ids, dtype=np.int64)

    # ── Fold generation ───────────────────────────────────────────────────────

    @staticmethod
    def _verify_seq_ids(seq_ids: np.ndarray) -> None:
        """Verify sequence IDs are consistent (placeholder for integrity check).

        Args:
            seq_ids: Sequence ID array.

        Raises:
            ValueError: If the array is empty.
        """
        if len(seq_ids) == 0:
            raise ValueError("seq_ids is empty — no data loaded.")

    def get_folds(
        self,
        y: np.ndarray,
        seq_ids: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate sequence-level stratified k-fold indices.

        Every window belonging to a given sequence is placed in exactly one
        fold.  A ValueError is raised if any sequence ID appears in more than
        one fold (sanity check after splitting).

        Args:
            y:       (N,) label array (used for stratification).
            seq_ids: (N,) sequence-level group IDs.

        Returns:
            List of (train_idx, test_idx) index arrays, one tuple per fold.

        Raises:
            ValueError: If any sequence ID leaks across train and test splits.
        """
        sgkf = StratifiedGroupKFold(n_splits=self.cfg.num_folds, shuffle=True,
                                    random_state=42)
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        dummy_X = np.zeros((len(y), 1))

        for train_idx, test_idx in sgkf.split(dummy_X, y, groups=seq_ids):
            # Integrity check: no shared sequence IDs across train/test
            train_seqs = set(seq_ids[train_idx].tolist())
            test_seqs  = set(seq_ids[test_idx].tolist())
            leaked = train_seqs & test_seqs
            if leaked:
                raise ValueError(
                    f"Sequence leakage detected: {len(leaked)} sequence IDs "
                    "appear in both train and test splits."
                )
            folds.append((train_idx, test_idx))

        return folds

    # ── DataLoader helpers ────────────────────────────────────────────────────

    def make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_ids: np.ndarray,
        is_train: bool,
    ) -> DataLoader:
        """Create a DataLoader for the given split.

        Training loaders use WeightedRandomSampler to balance fall/non-fall
        sampling (approximates fall-onset oversampling).  Validation/test
        loaders are ordered.

        Args:
            X:        (N, 7, T, J) feature array.
            y:        (N,)         label array.
            seq_ids:  (N,)         sequence IDs (passed through to Dataset).
            is_train: If True, apply augmentation and weighted sampling.

        Returns:
            Configured DataLoader instance.
        """
        ds = SkeletonDataset(X, y, seq_ids, is_train=is_train,
                             aug_cfg=self.cfg.augmentation)
        if is_train:
            weights  = ds.class_weights_for_sampler()
            sampler  = WeightedRandomSampler(weights, num_samples=len(weights),
                                             replacement=True)
            return DataLoader(ds, batch_size=self.cfg.batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True,
                              drop_last=True)
        return DataLoader(ds, batch_size=self.cfg.batch_size * 2,
                          shuffle=False, num_workers=4, pin_memory=True)


# ──────────────────────────────────────────────────────────────────────────────
# Temporal smoothing
# ──────────────────────────────────────────────────────────────────────────────

def temporal_smooth(probs: np.ndarray, k: int) -> np.ndarray:
    """Apply sliding-window majority smoothing to predicted probabilities.

    A window is confirmed as a fall only if at least k out of k consecutive
    windows (centred) exceed the decision threshold.  Implemented by replacing
    each probability with the mean of its k-wide neighbourhood — a higher mean
    means the window is surrounded by high-confidence fall predictions, reducing
    isolated false positives while preserving true fall clusters.

    Args:
        probs: (N,) predicted fall probabilities.
        k:     Neighbourhood half-width; effective window = 2k-1.
               k=1 means no smoothing.

    Returns:
        (N,) smoothed probabilities.
    """
    if k <= 1:
        return probs
    kernel = np.ones(k) / k
    return np.convolve(probs, kernel, mode="same")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """Computes all evaluation metrics and generates publication-quality plots.

    Metrics: accuracy, sensitivity, specificity, F1 (fall), macro-F1,
    AUC-ROC, AUC-PR, false negatives, false positives.  ROC and PR curves
    are saved as MLflow image artifacts.

    Args:
        output_dir: Directory where figure files are written.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
        tag: str = "",
    ) -> dict[str, float]:
        """Compute the full metric suite at a given decision threshold.

        Args:
            y_true:    (N,) integer ground-truth labels.
            y_prob:    (N,) predicted fall probability scores.
            threshold: Decision threshold for binary classification.
            tag:       Optional string prefix for log messages.

        Returns:
            Dictionary with keys: accuracy, sensitivity, specificity,
            f1_fall, f1_macro, auc_roc, auc_pr, fn, fp, threshold.
        """
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        accuracy    = accuracy_score(y_true, y_pred)
        f1_fall     = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1_macro    = f1_score(y_true, y_pred, average="macro", zero_division=0)
        auc_roc     = roc_auc_score(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        auc_pr      = auc(rec, prec)

        metrics = dict(
            accuracy=float(accuracy),
            sensitivity=float(sensitivity),
            specificity=float(specificity),
            f1_fall=float(f1_fall),
            f1_macro=float(f1_macro),
            auc_roc=float(auc_roc),
            auc_pr=float(auc_pr),
            fn=int(fn),
            fp=int(fp),
            threshold=float(threshold),
        )

        prefix = f"[{tag}] " if tag else ""
        log.info(
            "%sAUC-ROC=%.4f  AUC-PR=%.4f  Sen=%.4f  Spe=%.4f  "
            "F1-fall=%.4f  FN=%d  FP=%d  thr=%.3f",
            prefix, auc_roc, auc_pr, sensitivity, specificity,
            f1_fall, fn, fp, threshold,
        )
        return metrics

    # ------------------------------------------------------------------
    @staticmethod
    def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Find the decision threshold maximising Youden's J statistic.

        J = sensitivity + specificity − 1.

        Args:
            y_true: (N,) ground-truth labels.
            y_prob: (N,) predicted fall probabilities.

        Returns:
            Optimal threshold in [0, 1].
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j = tpr - fpr
        idx = int(np.argmax(j))
        return float(thresholds[idx])

    # ------------------------------------------------------------------
    def plot_roc(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        tag: str,
        optimal_thr: float | None = None,
    ) -> Path:
        """Save a ROC curve figure.

        Args:
            y_true:      Ground-truth labels.
            y_prob:      Predicted probabilities.
            tag:         Filename tag (e.g. 'fold0_variant_a').
            optimal_thr: If provided, marks the Youden point on the curve.

        Returns:
            Path to the saved PNG file.
        """
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        if optimal_thr is not None:
            idx = int(np.argmin(np.abs(thr - optimal_thr)))
            ax.scatter(fpr[idx], tpr[idx], s=80, zorder=5,
                       color="tomato", label=f"Youden thr={optimal_thr:.3f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {tag}")
        ax.legend()
        out = self.output_dir / f"roc_{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    # ------------------------------------------------------------------
    def plot_pr(self, y_true: np.ndarray, y_prob: np.ndarray, tag: str) -> Path:
        """Save a Precision-Recall curve figure.

        Args:
            y_true: Ground-truth labels.
            y_prob: Predicted probabilities.
            tag:    Filename tag.

        Returns:
            Path to the saved PNG file.
        """
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = auc(rec, prec)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec, prec, lw=2, label=f"AUC-PR = {ap:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve — {tag}")
        ax.legend()
        out = self.output_dir / f"pr_{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    # ------------------------------------------------------------------
    def plot_confusion(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tag: str,
    ) -> Path:
        """Save a normalised confusion matrix figure.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            tag:    Filename tag.

        Returns:
            Path to the saved PNG file.
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize="true")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                    xticklabels=["non-fall", "fall"],
                    yticklabels=["non-fall", "fall"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {tag}")
        out = self.output_dir / f"cm_{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# LR Scheduler helper
# ──────────────────────────────────────────────────────────────────────────────

def cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a cosine annealing scheduler with linear warmup.

    Args:
        optimizer:     The optimizer to schedule.
        warmup_epochs: Number of epochs for linear warmup.
        total_epochs:  Total training epochs.

    Returns:
        LambdaLR scheduler instance.
    """
    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch) / max(warmup_epochs, 1)
        progress = float(epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Encapsulates the full training loop for both Stage A and Stage B.

    Responsibilities:
      - Mixed-precision training with GradScaler.
      - Gradient accumulation and clipping.
      - Early stopping on validation AUC-ROC.
      - Per-fold Youden threshold computation.
      - Cross-fold threshold averaging for test-set evaluation.
      - Checkpoint saving (best AUC per fold + final epoch).
      - MLflow logging.

    Args:
        cfg:        Populated TrainConfig.
        dm:         Initialised DataModule.
        evaluator:  Evaluator instance.
        device:     Torch device string.
        use_mlflow: If True, log to MLflow (requires mlflow to be installed).
    """

    def __init__(
        self,
        cfg: TrainConfig,
        dm: DataModule,
        evaluator: Evaluator,
        device: str,
        use_mlflow: bool = True,
    ) -> None:
        self.cfg        = cfg
        self.dm         = dm
        self.evaluator  = evaluator
        self.device     = torch.device(device)
        self.use_mlflow = use_mlflow and _MLFLOW_AVAILABLE
        self.ckpt_dir   = Path(cfg.output_dir) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Internal epoch loop ───────────────────────────────────────────────────

    def _run_epoch(
        self,
        model: FallDetectionNet,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scaler: GradScaler,
        accum_steps: int,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Run one epoch of training or evaluation.

        Args:
            model:       The network.
            loader:      DataLoader for this split.
            criterion:   Loss function.
            optimizer:   If not None, training mode; else evaluation mode.
            scaler:      GradScaler for AMP.
            accum_steps: Gradient accumulation steps.

        Returns:
            mean_loss: Average loss over all batches.
            all_probs: (N,) predicted fall probabilities.
            all_labels: (N,) ground-truth labels.
        """
        is_train = optimizer is not None
        model.train(is_train)
        total_loss = 0.0
        all_probs:  list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        pbar = tqdm(loader, leave=False, disable=not is_train)
        if is_train:
            optimizer.zero_grad()

        for step, (X, y) in enumerate(pbar):
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                logits = model(X)
                loss   = criterion(logits, y) / accum_steps

            if is_train:
                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                with torch.no_grad():
                    pass  # loss already computed without grad above

            total_loss += loss.item() * accum_steps
            probs = F.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.cpu().numpy())

            if is_train:
                pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

        # Handle remaining accumulated gradients at epoch end
        if is_train and (len(loader) % accum_steps) != 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        mean_loss  = total_loss / max(len(loader), 1)
        all_probs  = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        return mean_loss, all_probs, all_labels

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _save_checkpoint(
        self,
        model: FallDetectionNet,
        metadata: dict,
        name: str,
    ) -> Path:
        """Save model state_dict plus metadata.

        Args:
            model:    The network.
            metadata: Dict with fold, epoch, val_auc, threshold, etc.
            name:     Checkpoint filename (without extension).

        Returns:
            Path to the saved .pt file.
        """
        path = self.ckpt_dir / f"{name}.pt"
        torch.save({
            "state_dict":   model.state_dict(),
            "metadata":     metadata,
            "arch_config": {
                "joint_embed_dim":   self.cfg.joint_embed_dim,
                "backbone_channels": self.cfg.backbone_channels,
                "dilations":         self.cfg.dilations,
                "dropout":           self.cfg.dropout,
                "in_features":       self.cfg.in_features,
                "num_joints":        self.cfg.num_joints,
            },
        }, str(path))
        log.info("Saved checkpoint → %s", path)
        return path

    # ── Stage A ───────────────────────────────────────────────────────────────

    def run_stage_a(self, seed: int = 42) -> Path:
        """Run Stage A multi-class NTU pretraining.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Path to the best Stage A checkpoint.
        """
        set_seed(seed)
        cfg_a  = self.cfg.stage_a
        X, y, seq_ids, num_classes = self.dm.setup_stage_a()

        run_kwargs: dict = {"run_name": "stage_a_ntu_pretraining"}
        ctx = mlflow.start_run(**run_kwargs) if self.use_mlflow else _null_ctx()
        with ctx:
            if self.use_mlflow:
                mlflow.log_params({
                    "stage": "a", "num_classes": num_classes,
                    "lr": cfg_a.lr, "weight_decay": cfg_a.weight_decay,
                    "num_epochs": cfg_a.num_epochs, "seed": seed,
                    "backbone_channels": self.cfg.backbone_channels,
                    "dilations": str(self.cfg.dilations),
                })

            # Simple 80/20 split (no nested CV needed for pretraining)
            rng  = np.random.default_rng(seed)
            idx  = rng.permutation(len(y))
            n_tr = int(0.8 * len(y))
            tr, va = idx[:n_tr], idx[n_tr:]

            loader_tr = self.dm.make_loader(X[tr], y[tr], seq_ids[tr], is_train=True)
            loader_va = self.dm.make_loader(X[va], y[va], seq_ids[va], is_train=False)

            model = FallDetectionNet(
                num_classes=num_classes,
                joint_embed_dim=self.cfg.joint_embed_dim,
                backbone_channels=self.cfg.backbone_channels,
                dilations=self.cfg.dilations,
                dropout=self.cfg.dropout,
                num_joints=self.cfg.num_joints,
                in_features=self.cfg.in_features,
            ).to(self.device)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg_a.lr, weight_decay=cfg_a.weight_decay,
            )
            criterion = nn.CrossEntropyLoss()
            scaler    = GradScaler(enabled=self.device.type == "cuda")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg_a.num_epochs,
            )

            best_val_loss = float("inf")
            patience_ctr  = 0
            best_ckpt: Path | None = None
            t0 = time.time()

            for epoch in range(cfg_a.num_epochs):
                tr_loss, _, _ = self._run_epoch(
                    model, loader_tr, criterion, optimizer, scaler,
                    self.cfg.accum_steps,
                )
                va_loss, va_probs, va_labels = self._run_epoch(
                    model, loader_va, criterion, None, scaler,
                    self.cfg.accum_steps,
                )
                scheduler.step()
                va_acc = (va_probs.argmax(axis=-1) if va_probs.ndim > 1
                          else (va_probs >= 0.5)).mean()

                log.info("Stage A  epoch %3d/%d  tr_loss=%.4f  va_loss=%.4f  va_acc=%.4f",
                         epoch + 1, cfg_a.num_epochs, tr_loss, va_loss, float(va_acc))

                if self.use_mlflow:
                    mlflow.log_metrics({
                        "train_loss": tr_loss, "val_loss": va_loss,
                    }, step=epoch)

                if va_loss < best_val_loss - 1e-4:
                    best_val_loss = va_loss
                    patience_ctr  = 0
                    meta = {"epoch": epoch, "val_loss": va_loss,
                            "stage": "a", "timestamp": time.strftime("%Y%m%d_%H%M%S")}
                    best_ckpt = self._save_checkpoint(model, meta, "stage_a_best")
                else:
                    patience_ctr += 1
                    if patience_ctr >= cfg_a.early_stopping_patience:
                        log.info("Early stopping at epoch %d", epoch + 1)
                        break

            elapsed = time.time() - t0
            if self.use_mlflow:
                mlflow.log_metric("training_time_s", elapsed)
                if best_ckpt:
                    mlflow.log_artifact(str(best_ckpt))

        log.info("Stage A complete in %.1f s.  Best ckpt: %s", elapsed, best_ckpt)
        return best_ckpt  # type: ignore[return-value]

    # ── Stage B ───────────────────────────────────────────────────────────────

    def run_stage_b(
        self,
        variant: str,
        resume: str | None = None,
        seed: int = 42,
    ) -> dict[str, list[dict]]:
        """Run Stage B fine-tuning for a single variant (A or B).

        Args:
            variant: 'a' for frozen backbone, 'b' for full fine-tuning.
            resume:  Path to Stage A checkpoint to initialise backbone.
            seed:    Random seed.

        Returns:
            Dictionary mapping 'fold_results' to a list of per-fold metric dicts.
        """
        set_seed(seed)
        X, y, seq_ids = self.dm.setup_stage_b()
        folds = self.dm.get_folds(y, seq_ids)

        variant_name = f"stage_b_variant_{variant}_{'frozen' if variant == 'a' else 'fulltune'}"

        parent_ctx = (
            mlflow.start_run(run_name="stage_b_finetuning")
            if self.use_mlflow else _null_ctx()
        )
        with parent_ctx:
            child_ctx = (
                mlflow.start_run(run_name=variant_name, nested=True)
                if self.use_mlflow else _null_ctx()
            )
            with child_ctx:
                if self.use_mlflow:
                    mlflow.log_params({
                        "stage": "b", "variant": variant,
                        "resume": str(resume), "seed": seed,
                        "num_folds": self.cfg.num_folds,
                    })

                # First pass: train all folds, collect val thresholds + test preds
                fold_thresholds: list[float] = []
                fold_test_probs:  list[np.ndarray] = []
                fold_test_labels: list[np.ndarray] = []
                fold_histories:   list[dict] = []

                for fold_k, (train_val_idx, test_idx) in enumerate(folds):
                    log.info("─── Fold %d/%d ───", fold_k + 1, self.cfg.num_folds)

                    # Split train_val into train (80%) + val (20%) by sequence
                    tv_seqs = seq_ids[train_val_idx]
                    unique_seqs = np.unique(tv_seqs)
                    rng = np.random.default_rng(seed + fold_k)
                    rng.shuffle(unique_seqs)
                    n_val_seqs = max(1, int(0.2 * len(unique_seqs)))
                    val_seqs   = set(unique_seqs[:n_val_seqs].tolist())
                    val_mask   = np.array([s in val_seqs for s in tv_seqs])
                    tr_idx  = train_val_idx[~val_mask]
                    val_idx = train_val_idx[val_mask]

                    loader_tr  = self.dm.make_loader(X[tr_idx],  y[tr_idx],
                                                     seq_ids[tr_idx],  True)
                    loader_val = self.dm.make_loader(X[val_idx], y[val_idx],
                                                     seq_ids[val_idx], False)
                    loader_tst = self.dm.make_loader(X[test_idx], y[test_idx],
                                                     seq_ids[test_idx], False)

                    model, optimizer, scheduler, criterion, scaler, num_epochs, patience = \
                        self._build_stage_b_components(variant, y[tr_idx], resume)

                    thr, test_probs, test_labels, history = self._train_fold(
                        model, loader_tr, loader_val, loader_tst,
                        criterion, optimizer, scheduler, scaler,
                        num_epochs, patience,
                        fold_k=fold_k, variant=variant,
                    )
                    fold_thresholds.append(thr)
                    fold_test_probs.append(test_probs)
                    fold_test_labels.append(test_labels)
                    fold_histories.append(history)

                # Second pass: evaluate each test fold with cross-fold threshold
                fold_results: list[dict] = []
                for fold_k in range(len(folds)):
                    other_thr  = [fold_thresholds[j] for j in range(len(folds))
                                  if j != fold_k]
                    cross_thr  = float(np.mean(other_thr)) if other_thr else 0.5
                    tag = f"{variant_name}_fold{fold_k}"
                    smoothed_probs = temporal_smooth(
                        fold_test_probs[fold_k], self.cfg.smooth_k)
                    metrics = self.evaluator.compute_metrics(
                        fold_test_labels[fold_k], smoothed_probs,
                        threshold=cross_thr, tag=tag,
                    )
                    metrics["fold"] = fold_k
                    metrics["own_threshold"] = fold_thresholds[fold_k]
                    fold_results.append(metrics)

                    if self.use_mlflow:
                        mlflow.log_metrics(
                            {f"fold{fold_k}_{k}": v for k, v in metrics.items()
                             if isinstance(v, float)},
                            step=fold_k,
                        )
                    # Generate + log plots (use smoothed probs)
                    roc_path = self.evaluator.plot_roc(
                        fold_test_labels[fold_k], smoothed_probs,
                        tag=tag, optimal_thr=cross_thr,
                    )
                    pr_path = self.evaluator.plot_pr(
                        fold_test_labels[fold_k], smoothed_probs, tag=tag,
                    )
                    y_pred = (smoothed_probs >= cross_thr).astype(int)
                    cm_path = self.evaluator.plot_confusion(
                        fold_test_labels[fold_k], y_pred, tag=tag,
                    )
                    if self.use_mlflow:
                        for p in [roc_path, pr_path, cm_path]:
                            mlflow.log_artifact(str(p))

                # Aggregate across folds + generate paper figures
                smoothed_test_probs = [
                    temporal_smooth(p, self.cfg.smooth_k) for p in fold_test_probs
                ]
                self._log_aggregate(fold_results, variant_name,
                                    fold_histories=fold_histories,
                                    fold_test_probs=smoothed_test_probs,
                                    fold_test_labels=fold_test_labels)

        return {"fold_results": fold_results}

    # ── Stage B component builder ─────────────────────────────────────────────

    def _build_stage_b_components(
        self,
        variant: str,
        y_train: np.ndarray,
        resume: str | None,
    ) -> tuple:
        """Instantiate model, optimizer, scheduler, criterion, scaler.

        Args:
            variant:  'a' or 'b'.
            y_train:  Training labels for this fold (used to compute alpha).
            resume:   Path to Stage A checkpoint, or None.

        Returns:
            (model, optimizer, scheduler, criterion, scaler, num_epochs, patience)
        """
        fl_cfg = self.cfg.focal_loss
        criterion = FocalLoss.from_labels(
            y_train, gamma=fl_cfg.gamma, eps=fl_cfg.label_smoothing,
            alpha_cap=fl_cfg.alpha_cap,
        )

        model = FallDetectionNet(
            num_classes=2,
            joint_embed_dim=self.cfg.joint_embed_dim,
            backbone_channels=self.cfg.backbone_channels,
            dilations=self.cfg.dilations,
            dropout=self.cfg.dropout,
            num_joints=self.cfg.num_joints,
            in_features=self.cfg.in_features,
        ).to(self.device)

        if resume:
            ckpt = torch.load(resume, map_location=self.device, weights_only=False)
            # Load backbone weights; head will be freshly initialised
            state = ckpt["state_dict"]
            head_keys = {k for k in state if k.startswith("head.")}
            backbone_state = {k: v for k, v in state.items() if k not in head_keys}
            missing, unexpected = model.load_state_dict(backbone_state, strict=False)
            log.info("Loaded backbone from %s  (missing=%d, unexpected=%d)",
                     resume, len(missing), len(unexpected))

        scaler = GradScaler("cuda", enabled=self.device.type == "cuda")

        if variant == "a":
            # Freeze all backbone params
            for p in model.get_backbone_params():
                p.requires_grad = False
            cfg_v = self.cfg.stage_b_variant_a
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg_v.lr_head,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg_v.num_epochs,
            )
            num_epochs = cfg_v.num_epochs
            patience   = cfg_v.early_stopping_patience

        else:  # variant == "b"
            cfg_v = self.cfg.stage_b_variant_b
            optimizer = torch.optim.AdamW([
                {"params": model.get_backbone_params(), "lr": cfg_v.lr_backbone},
                {"params": model.head.parameters(),     "lr": cfg_v.lr_head},
            ], weight_decay=cfg_v.weight_decay)
            scheduler = cosine_schedule_with_warmup(
                optimizer,
                warmup_epochs=cfg_v.warmup_epochs,
                total_epochs=cfg_v.num_epochs,
            )
            num_epochs = cfg_v.num_epochs
            patience   = cfg_v.early_stopping_patience

        return model, optimizer, scheduler, criterion, scaler, num_epochs, patience

    # ── Per-fold training ─────────────────────────────────────────────────────

    def _train_fold(
        self,
        model: FallDetectionNet,
        loader_tr: DataLoader,
        loader_val: DataLoader,
        loader_tst: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: GradScaler,
        num_epochs: int,
        patience: int,
        fold_k: int,
        variant: str,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Train a single fold.

        Args:
            model, loader_tr, loader_val, loader_tst: Standard components.
            criterion, optimizer, scheduler, scaler:  Training utilities.
            num_epochs:  Maximum epochs.
            patience:    Early stopping patience on val AUC-ROC.
            fold_k:      Fold index (for checkpoint naming).
            variant:     'a' or 'b'.

        Returns:
            val_threshold: Youden-optimal threshold from the validation fold.
            test_probs:    (N_test,) predicted probabilities on the test fold.
            test_labels:   (N_test,) ground-truth labels for the test fold.
        """
        best_val_auc = 0.0
        patience_ctr = 0
        best_ckpt_path: Path | None = None
        t0 = time.time()
        history: dict[str, list] = {"tr_loss": [], "va_loss": [], "val_auc": []}

        for epoch in range(num_epochs):
            tr_loss, _, _          = self._run_epoch(
                model, loader_tr, criterion, optimizer, scaler, self.cfg.accum_steps)
            va_loss, va_probs, va_y = self._run_epoch(
                model, loader_val, criterion, None, scaler, self.cfg.accum_steps)
            scheduler.step()

            try:
                val_auc = roc_auc_score(va_y, va_probs)
            except ValueError:
                val_auc = 0.5

            history["tr_loss"].append(tr_loss)
            history["va_loss"].append(va_loss)
            history["val_auc"].append(val_auc)

            log.info("  [fold %d  ep %3d/%d]  tr=%.4f  va=%.4f  auc=%.4f",
                     fold_k, epoch + 1, num_epochs, tr_loss, va_loss, val_auc)

            if val_auc > best_val_auc + 1e-4:
                best_val_auc   = val_auc
                patience_ctr   = 0
                ckpt_name = (f"best_fold{fold_k}_auc{val_auc:.4f}"
                             f"_epoch{epoch+1}_variant{variant}")
                best_ckpt_path = self._save_checkpoint(
                    model,
                    {"fold": fold_k, "epoch": epoch, "val_auc": val_auc,
                     "variant": variant,
                     "timestamp": time.strftime("%Y%m%d_%H%M%S")},
                    ckpt_name,
                )
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    log.info("  Early stop fold %d at epoch %d", fold_k, epoch + 1)
                    break

        # Save final epoch checkpoint
        self._save_checkpoint(
            model,
            {"fold": fold_k, "epoch": epoch, "val_auc": val_auc,
             "variant": variant, "final": True,
             "timestamp": time.strftime("%Y%m%d_%H%M%S")},
            f"final_fold{fold_k}_variant{variant}",
        )

        # Load best checkpoint for threshold computation + test evaluation
        if best_ckpt_path:
            ckpt = torch.load(best_ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])

        # Recompute val probs with best model for Youden threshold
        _, va_probs_best, va_y_best = self._run_epoch(
            model, loader_val, criterion, None, scaler, self.cfg.accum_steps)
        val_threshold = self.evaluator.youden_threshold(va_y_best, va_probs_best)

        _, test_probs, test_labels = self._run_epoch(
            model, loader_tst, criterion, None, scaler, self.cfg.accum_steps)

        elapsed = time.time() - t0
        log.info("  Fold %d done in %.1f s  best_val_auc=%.4f  youden_thr=%.3f",
                 fold_k, elapsed, best_val_auc, val_threshold)

        if self.use_mlflow:
            mlflow.log_metrics({
                f"fold{fold_k}_best_val_auc":    best_val_auc,
                f"fold{fold_k}_youden_threshold": val_threshold,
                f"fold{fold_k}_training_time_s":  elapsed,
            })
            if best_ckpt_path:
                mlflow.log_artifact(str(best_ckpt_path))

        return val_threshold, test_probs, test_labels, history

    # ── Aggregate reporting ───────────────────────────────────────────────────

    def _log_aggregate(
        self,
        fold_results: list[dict],
        tag: str,
        fold_histories: list[dict] | None = None,
        fold_test_probs: list[np.ndarray] | None = None,
        fold_test_labels: list[np.ndarray] | None = None,
    ) -> None:
        """Log mean ± std of all metrics across folds and save paper figures."""
        metric_keys = ["accuracy", "sensitivity", "specificity",
                       "f1_fall", "f1_macro", "auc_roc", "auc_pr"]
        log.info("═══ Aggregate (%s) ═══", tag)
        agg: dict[str, float] = {}
        for k in metric_keys:
            vals = [r[k] for r in fold_results if k in r]
            if vals:
                m, s = float(np.mean(vals)), float(np.std(vals))
                log.info("  %-15s %.4f ± %.4f", k, m, s)
                agg[f"{k}_mean"] = m
                agg[f"{k}_std"]  = s
        fn_total = sum(r.get("fn", 0) for r in fold_results)
        fp_total = sum(r.get("fp", 0) for r in fold_results)
        log.info("  Total FN (missed falls): %d", fn_total)
        log.info("  Total FP (false alarms): %d", fp_total)
        if self.use_mlflow:
            mlflow.log_metrics(agg)

        fig_dir = self.evaluator.output_dir
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # ── Figure 1: Loss curves (train & val) per fold ──────────────────────
        if fold_histories:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for i, h in enumerate(fold_histories):
                ep = range(1, len(h["tr_loss"]) + 1)
                axes[0].plot(ep, h["tr_loss"], color=colors[i % len(colors)],
                             alpha=0.8, label=f"Fold {i}")
                axes[1].plot(ep, h["va_loss"], color=colors[i % len(colors)],
                             alpha=0.8, label=f"Fold {i}")
            for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Focal Loss")
                ax.set_title(title)
                ax.legend(fontsize=8)
            fig.suptitle(f"Loss Curves — {tag}", fontweight="bold")
            fig.tight_layout()
            out = fig_dir / f"loss_curves_{tag}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("Saved loss curves → %s", out)

        # ── Figure 2: Val AUC-ROC per epoch per fold ──────────────────────────
        if fold_histories:
            fig, ax = plt.subplots(figsize=(8, 4))
            for i, h in enumerate(fold_histories):
                ep = range(1, len(h["val_auc"]) + 1)
                ax.plot(ep, h["val_auc"], color=colors[i % len(colors)],
                        alpha=0.8, label=f"Fold {i}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation AUC-ROC")
            ax.set_title(f"AUC-ROC per Epoch — {tag}", fontweight="bold")
            ax.legend(fontsize=8)
            fig.tight_layout()
            out = fig_dir / f"auc_curve_{tag}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("Saved AUC curve → %s", out)

        # ── Figure 3: Overlaid ROC curves with mean ± std band ────────────────
        if fold_test_probs and fold_test_labels:
            fig, ax = plt.subplots(figsize=(6, 5))
            mean_fpr = np.linspace(0, 1, 200)
            tprs: list[np.ndarray] = []
            aucs: list[float] = []
            for i, (probs, labels) in enumerate(zip(fold_test_probs, fold_test_labels)):
                fpr, tpr, _ = roc_curve(labels, probs)
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc_val = auc(fpr, tpr)
                aucs.append(roc_auc_val)
                ax.plot(mean_fpr, tprs[-1], color=colors[i % len(colors)],
                        alpha=0.4, lw=1, label=f"Fold {i} (AUC={roc_auc_val:.3f})")
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            std_tpr = np.std(tprs, axis=0)
            mean_auc = float(np.mean(aucs))
            std_auc  = float(np.std(aucs))
            ax.plot(mean_fpr, mean_tpr, color="navy", lw=2,
                    label=f"Mean ROC (AUC={mean_auc:.3f} ± {std_auc:.3f})")
            ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                            alpha=0.15, color="navy", label="± 1 std")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"Mean ROC Curve — {tag}", fontweight="bold")
            ax.legend(fontsize=7, loc="lower right")
            fig.tight_layout()
            out = fig_dir / f"roc_mean_{tag}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("Saved mean ROC → %s", out)

        # ── Figure 4: Aggregated confusion matrix (sum across folds) ──────────
        if fold_test_probs and fold_test_labels and fold_results:
            thresholds = [r.get("threshold", 0.5) for r in fold_results]
            cm_total = np.zeros((2, 2), dtype=int)
            for probs, labels, thr in zip(fold_test_probs, fold_test_labels, thresholds):
                y_pred = (probs >= thr).astype(int)
                cm_total += confusion_matrix(labels, y_pred, labels=[0, 1])
            cm_norm = cm_total.astype(float) / cm_total.sum(axis=1, keepdims=True)
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            for ax, data, fmt, title in zip(
                axes,
                [cm_total, cm_norm],
                ["d", ".2f"],
                ["Counts", "Normalised"],
            ):
                sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                            xticklabels=["Non-Fall", "Fall"],
                            yticklabels=["Non-Fall", "Fall"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(title)
            fig.suptitle(f"Aggregated Confusion Matrix — {tag}", fontweight="bold")
            fig.tight_layout()
            out = fig_dir / f"cm_aggregate_{tag}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("Saved aggregate confusion matrix → %s", out)

        # ── Figure 5: Per-metric bar chart (mean ± std across folds) ──────────
        display_keys = ["accuracy", "sensitivity", "specificity",
                        "f1_fall", "auc_roc", "auc_pr"]
        means = [agg.get(f"{k}_mean", 0.0) for k in display_keys]
        stds  = [agg.get(f"{k}_std",  0.0) for k in display_keys]
        labels_bar = ["Accuracy", "Sensitivity", "Specificity",
                      "F1-Fall", "AUC-ROC", "AUC-PR"]
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(labels_bar))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=colors[:len(labels_bar)], alpha=0.85)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.005,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_bar, rotation=15, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.set_title(f"Performance Summary — {tag}", fontweight="bold")
        fig.tight_layout()
        out = fig_dir / f"metrics_bar_{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved metrics bar chart → %s", out)


# ──────────────────────────────────────────────────────────────────────────────
# Utility: null context manager (when MLflow is disabled)
# ──────────────────────────────────────────────────────────────────────────────

class _NullCtx:
    """No-op context manager used when MLflow is disabled."""
    def __enter__(self) -> "_NullCtx": return self
    def __exit__(self, *_: Any) -> None: pass

def _null_ctx() -> _NullCtx:
    return _NullCtx()


# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path, debug: bool) -> None:
    """Configure logging to console and a timestamped file.

    tqdm progress bars go to the console only; the log file receives one
    summary line per epoch via the standard logger.

    Args:
        output_dir: Directory for the log file.
        debug:      If True, set log level to DEBUG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    ts    = time.strftime("%Y%m%d_%H%M%S")

    fmt     = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                                 datefmt="%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)

    fh = logging.FileHandler(output_dir / f"train_{ts}.log")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(fh)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    p = argparse.ArgumentParser(
        description="FallDetectionNet training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",    type=str, default="research_config.json",
                   help="Path to research_config.json")
    p.add_argument("--stage",     type=str, choices=["a", "b"], required=True,
                   help="Training stage: 'a' = NTU pretraining, 'b' = fine-tuning")
    p.add_argument("--variant",   type=str, choices=["a", "b"], default="b",
                   help="Stage B variant: 'a' = frozen backbone, 'b' = full fine-tune")
    p.add_argument("--seed",      type=int, default=42,
                   help="Global random seed")
    p.add_argument("--resume",    type=str, default=None,
                   help="Path to checkpoint to resume / initialise from")
    p.add_argument("--debug",     action="store_true",
                   help="Use 10 %% of data for fast iteration")
    p.add_argument("--no-mlflow", action="store_true",
                   help="Disable MLflow tracking for local testing")
    p.add_argument("--device",    type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Torch device string")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Usage examples ────────────────────────────────────────────────────────
    # Stage A pretraining (requires DATASET/NTU/*.skeleton):
    #   python 03_train.py --stage a --config research_config.json --seed 42
    #
    # Stage B Variant A — frozen backbone:
    #   python 03_train.py --stage b --variant a \
    #       --resume outputs/checkpoints/stage_a_best.pt
    #
    # Stage B Variant B — full fine-tuning with differential LRs:
    #   python 03_train.py --stage b --variant b \
    #       --resume outputs/checkpoints/stage_a_best.pt
    #
    # Debug run (no MLflow, 10 % of data):
    #   python 03_train.py --stage b --variant a --debug --no-mlflow
    # ─────────────────────────────────────────────────────────────────────────

    args   = build_parser().parse_args()
    cfg    = load_config(args.config)
    out_dir = Path(cfg.output_dir)

    setup_logging(out_dir / "logs", args.debug)
    set_seed(args.seed)

    log.info("FallDetectionNet — stage=%s  device=%s  seed=%d  debug=%s",
             args.stage, args.device, args.seed, args.debug)

    use_mlflow = not args.no_mlflow and _MLFLOW_AVAILABLE
    if use_mlflow:
        mlflow.set_experiment("FallDetectionNet")

    dm        = DataModule(cfg, device=args.device, debug=args.debug)
    evaluator = Evaluator(out_dir / "figures")
    trainer   = Trainer(cfg, dm, evaluator, device=args.device,
                        use_mlflow=use_mlflow)

    if args.stage == "a":
        best_ckpt = trainer.run_stage_a(seed=args.seed)
        log.info("Stage A complete.  Best checkpoint: %s", best_ckpt)

    elif args.stage == "b":
        results = trainer.run_stage_b(
            variant=args.variant, resume=args.resume, seed=args.seed,
        )
        aucs = [r["auc_roc"] for r in results["fold_results"]]
        log.info("Stage B variant-%s complete.  Mean AUC-ROC = %.4f ± %.4f",
                 args.variant, float(np.mean(aucs)), float(np.std(aucs)))