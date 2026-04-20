"""
Fall Detection Data Pipeline
Processes URFD, Le2i, and NTU RGB+D 120 datasets into temporal feature sequences
for journal-quality fall detection research.

Datasets:
  - URFD:  Image sequences (fall/ vs adl/), YOLOv8 pose extraction
  - Le2i:  AVI videos with frame-level annotations, YOLOv8 pose extraction
  - NTU:   Pre-extracted .skeleton files (25 joints), A043 = fall action

Output:
  processed/{dataset}/sequences.npz  with keys: X (N, T, F), y (N,), meta (N,)
"""

import os
import re
import logging
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # Paths
    dataset_root: str = "DATASET"
    output_root: str = "processed"
    pose_model_path: str = "yolov8n-pose.pt"

    # Temporal windowing
    window_size: int = 30       # frames per sequence
    stride: int = 10            # step between windows

    # YOLOv8 inference
    pose_conf: float = 0.3
    pose_iou: float = 0.45
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # NTU fall action IDs (1-indexed in filename A###)
    ntu_fall_actions: tuple = (43,)   # A043 = fall down
    ntu_max_frames: int = 300         # skip pathologically long sequences

    # Sequence sampling for non-fall sequences (None = use all)
    max_nontfall_per_source: Optional[int] = None


CFG = PipelineConfig()

# ──────────────────────────────────────────────
# COCO 17-keypoint indices (YOLOv8 output)
# ──────────────────────────────────────────────
KP_NOSE        = 0
KP_L_EYE       = 1
KP_R_EYE       = 2
KP_L_EAR       = 3
KP_R_EAR       = 4
KP_L_SHOULDER  = 5
KP_R_SHOULDER  = 6
KP_L_ELBOW     = 7
KP_R_ELBOW     = 8
KP_L_WRIST     = 9
KP_R_WRIST     = 10
KP_L_HIP       = 11
KP_R_HIP       = 12
KP_L_KNEE      = 13
KP_R_KNEE      = 14
KP_L_ANKLE     = 15
KP_R_ANKLE     = 16

NUM_KP = 17
FEATURE_DIM = NUM_KP * 2 + 7   # 34 coords + 7 derived = 41


# ──────────────────────────────────────────────
# Feature extraction helpers
# ──────────────────────────────────────────────

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b formed by rays b->a and b->c (degrees)."""
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def extract_pose_features(kps: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """
    Build a fixed-length feature vector from 17 COCO keypoints.

    Parameters
    ----------
    kps  : (17, 2) normalized [x, y] in [0, 1] w.r.t. bounding box
    conf : (17,)   per-keypoint confidence scores

    Returns
    -------
    features : (FEATURE_DIM,)  = 34 coords + 7 derived
    """
    # --- 34 normalized coordinates ---
    coords = kps.flatten()   # (34,)

    # --- Derived features ---
    # 1. Bounding box aspect ratio (H / W) from raw kp spread
    xs, ys = kps[:, 0], kps[:, 1]
    kp_w = xs.max() - xs.min() + 1e-8
    kp_h = ys.max() - ys.min() + 1e-8
    aspect_ratio = kp_h / kp_w   # large when standing, small when lying

    # 2. Head (nose) to mid-hip vertical distance (normalized by kp_h)
    mid_hip = (kps[KP_L_HIP] + kps[KP_R_HIP]) / 2.0
    head_to_hip = (mid_hip[1] - kps[KP_NOSE][1]) / (kp_h + 1e-8)

    # 3. Hip angle (L_shoulder - mid_hip - L_knee)
    hip_angle = _angle(kps[KP_L_SHOULDER], mid_hip, kps[KP_L_KNEE])

    # 4. Left knee angle
    left_knee_angle = _angle(kps[KP_L_HIP], kps[KP_L_KNEE], kps[KP_L_ANKLE])

    # 5. Right knee angle
    right_knee_angle = _angle(kps[KP_R_HIP], kps[KP_R_KNEE], kps[KP_R_ANKLE])

    # 6. Shoulder–hip vertical alignment (cos of tilt)
    mid_shoulder = (kps[KP_L_SHOULDER] + kps[KP_R_SHOULDER]) / 2.0
    torso_vec = mid_shoulder - mid_hip
    torso_vertical = torso_vec[1] / (np.linalg.norm(torso_vec) + 1e-8)

    # 7. Mean keypoint confidence (upper body)
    upper_conf = conf[[KP_L_SHOULDER, KP_R_SHOULDER,
                        KP_L_HIP, KP_R_HIP, KP_NOSE]].mean()

    derived = np.array([
        aspect_ratio,
        head_to_hip,
        hip_angle / 180.0,
        left_knee_angle / 180.0,
        right_knee_angle / 180.0,
        torso_vertical,
        float(upper_conf),
    ], dtype=np.float32)

    return np.concatenate([coords.astype(np.float32), derived])


def _zero_frame_features() -> np.ndarray:
    """Return zero feature vector for frames with no detected person."""
    return np.zeros(FEATURE_DIM, dtype=np.float32)


def add_velocity(seq: np.ndarray) -> np.ndarray:
    """
    Append per-frame velocity (Δ of first 34 coord features) to sequence.

    Parameters
    ----------
    seq : (T, FEATURE_DIM)

    Returns
    -------
    seq_vel : (T, FEATURE_DIM + 34)
    """
    vel = np.zeros_like(seq[:, :34])
    vel[1:] = seq[1:, :34] - seq[:-1, :34]
    return np.concatenate([seq, vel], axis=1)


def make_windows(seq: np.ndarray, label: int, window: int, stride: int):
    """Slide a window over seq and return list of (window_array, label)."""
    samples = []
    T = len(seq)
    for start in range(0, T - window + 1, stride):
        samples.append((seq[start:start + window], label))
    return samples


# ──────────────────────────────────────────────
# YOLOv8 pose extractor
# ──────────────────────────────────────────────

class PoseExtractor:
    """Wraps YOLOv8-pose inference and normalises keypoints."""

    def __init__(self, model_path: str, conf: float, iou: float, device: str):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        log.info("Loaded pose model: %s on %s", model_path, device)

    def extract(self, frame_bgr: np.ndarray):
        """
        Run pose estimation on a single BGR frame.

        Returns
        -------
        kps_norm : (17, 2) float32  keypoints normalised to [0,1] in bbox
        conf     : (17,)  float32  keypoint confidences
        valid    : bool   True if at least one person detected with sufficient conf
        """
        results = self.model(
            frame_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        r = results[0]

        if r.keypoints is None or len(r.keypoints) == 0:
            return None, None, False

        # Pick the most-confident person (largest bounding box area)
        if r.boxes is not None and len(r.boxes) > 1:
            areas = r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]
            idx = int(areas.argmax())
        else:
            idx = 0

        kps_xy = r.keypoints.xy[idx].cpu().numpy()    # (17, 2) pixel coords
        kps_conf = r.keypoints.conf[idx].cpu().numpy() # (17,)

        # Normalise to [0, 1] within bounding box
        h, w = frame_bgr.shape[:2]
        if r.boxes is not None and len(r.boxes) > 0:
            x1, y1, x2, y2 = r.boxes.xyxy[idx].cpu().numpy()
            bw = max(x2 - x1, 1.0)
            bh = max(y2 - y1, 1.0)
            kps_norm = np.stack([
                (kps_xy[:, 0] - x1) / bw,
                (kps_xy[:, 1] - y1) / bh,
            ], axis=1)
        else:
            kps_norm = kps_xy / np.array([[w, h]], dtype=np.float32)

        return kps_norm.astype(np.float32), kps_conf.astype(np.float32), True


# ──────────────────────────────────────────────
# URFD Dataset Loader
# ──────────────────────────────────────────────

def load_urfd(cfg: PipelineConfig, extractor: PoseExtractor):
    """
    URFD structure:
      DATASET/URFD/fall/fall-NN-cam0-rgb/*.png  → label 1
      DATASET/URFD/adl/adl-NN-cam0-rgb/*.png    → label 0
    """
    urfd_root = Path(cfg.dataset_root) / "URFD"
    sequences = []
    seq_ids = []
    seq_id = 0

    for label, subdir in [(1, "fall"), (0, "adl")]:
        sub_path = urfd_root / subdir
        seq_dirs = sorted(sub_path.iterdir())
        for seq_dir in tqdm(seq_dirs, desc=f"URFD/{subdir}", leave=False):
            if not seq_dir.is_dir():
                continue
            frames_paths = sorted(
                seq_dir.glob("*.png"),
                key=lambda p: int(re.search(r"(\d+)", p.stem).group(1))
            )
            if len(frames_paths) < cfg.window_size:
                continue

            frame_features = []
            for fp in frames_paths:
                img = cv2.imread(str(fp))
                if img is None:
                    frame_features.append(_zero_frame_features())
                    continue
                kps, conf, valid = extractor.extract(img)
                if valid:
                    frame_features.append(extract_pose_features(kps, conf))
                else:
                    frame_features.append(_zero_frame_features())

            seq_arr = np.stack(frame_features, axis=0)  # (T, F)
            seq_arr = add_velocity(seq_arr)
            windows = make_windows(seq_arr, label, cfg.window_size, cfg.stride)
            seq_ids.extend([seq_id] * len(windows))
            sequences.extend(windows)
            seq_id += 1

    X = np.stack([w for w, _ in sequences], axis=0)
    y = np.array([l for _, l in sequences], dtype=np.int64)
    s = np.array(seq_ids, dtype=np.int64)
    log.info("URFD: %d windows (fall=%d, adl=%d)", len(y),
             (y == 1).sum(), (y == 0).sum())
    return X, y, s


# ──────────────────────────────────────────────
# Le2i Dataset Loader
# ──────────────────────────────────────────────

def _parse_le2i_annotation(ann_path: Path):
    """
    Le2i annotation format:
      line 0: fall_start_frame (1-indexed)
      line 1: fall_end_frame   (1-indexed)
      lines 2+: frame_idx,person_flag,x1,y1,x2,y2

    Returns (fall_start, fall_end) as 0-indexed ints, or (None, None).
    """
    lines = ann_path.read_text(errors="replace").strip().splitlines()
    if len(lines) < 2:
        return None, None
    try:
        fall_start = int(lines[0].strip()) - 1
        fall_end   = int(lines[1].strip()) - 1
        return fall_start, fall_end
    except ValueError:
        return None, None


LE2I_SCENES = [
    ("Coffee_room_01", "Coffee_room_01", "Annotation_files"),
    ("Coffee_room_02", "Coffee_room_02", "Annotations_files"),
    ("Home_01",        "Home_01",        "Annotation_files"),
    ("Home_02",        "Home_02",        "Annotation_files"),
]


def _video_dimensions(vf: Path):
    """Return (width, height) of the first video stream via ffprobe."""
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            str(vf),
        ],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        return None, None
    try:
        w, h = map(int, probe.stdout.strip().split(","))
        return w, h
    except ValueError:
        return None, None


def _iter_frames_ffmpeg(vf: Path):
    """
    Yield BGR frames from a video file by piping raw output from an ffmpeg
    subprocess.  Audio streams are disabled (-an) so the broken mp3float
    decoder in the Le2i AVIs never runs inside the Python process.
    """
    w, h = _video_dimensions(vf)
    if w is None:
        return

    proc = subprocess.Popen(
        [
            "ffmpeg", "-i", str(vf),
            "-an",                        # no audio
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    frame_bytes = h * w * 3
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def load_le2i(cfg: PipelineConfig, extractor: PoseExtractor):
    """
    Le2i structure (annotated scenes only):
      DATASET/LE2I/<scene>/<scene>/Videos/video (N).avi
      DATASET/LE2I/<scene>/<scene>/Annotation_files/video (N).txt
    """
    le2i_root = Path(cfg.dataset_root) / "LE2I"
    sequences = []
    seq_ids = []
    seq_id = 10000  # offset to avoid collision with URFD IDs

    for scene_dir, inner_dir, ann_subdir in LE2I_SCENES:
        video_dir = le2i_root / scene_dir / inner_dir / "Videos"
        ann_dir   = le2i_root / scene_dir / inner_dir / ann_subdir

        if not video_dir.exists():
            log.warning("Le2i: missing video dir %s", video_dir)
            continue

        video_files = sorted(video_dir.glob("*.avi"),
                             key=lambda p: int(re.search(r"\((\d+)\)", p.name).group(1)))

        for vf in tqdm(video_files, desc=f"Le2i/{scene_dir}", leave=False):
            ann_file = ann_dir / (vf.stem + ".txt")
            if not ann_file.exists():
                log.debug("No annotation for %s – skipping", vf.name)
                continue

            fall_start, fall_end = _parse_le2i_annotation(ann_file)
            if fall_start is None:
                continue

            frame_features = []
            frame_labels   = []
            for fid, frame in enumerate(_iter_frames_ffmpeg(vf)):
                kps, conf, valid = extractor.extract(frame)
                if valid:
                    frame_features.append(extract_pose_features(kps, conf))
                else:
                    frame_features.append(_zero_frame_features())
                lbl = 1 if fall_start <= fid <= fall_end else 0
                frame_labels.append(lbl)

            if not frame_features:
                log.warning("Cannot read frames from %s", vf)
                continue

            if len(frame_features) < cfg.window_size:
                continue

            seq_arr = np.stack(frame_features, axis=0)
            seq_arr = add_velocity(seq_arr)
            labels_arr = np.array(frame_labels)

            # Window-level label: 1 if any fall frame in window
            T = len(seq_arr)
            n_before = len(sequences)
            for start in range(0, T - cfg.window_size + 1, cfg.stride):
                end = start + cfg.window_size
                window = seq_arr[start:end]
                wlabel  = int(labels_arr[start:end].any())
                sequences.append((window, wlabel))
            seq_ids.extend([seq_id] * (len(sequences) - n_before))
            seq_id += 1

    X = np.stack([w for w, _ in sequences], axis=0)
    y = np.array([l for _, l in sequences], dtype=np.int64)
    s = np.array(seq_ids, dtype=np.int64)
    log.info("Le2i: %d windows (fall=%d, non-fall=%d)", len(y),
             (y == 1).sum(), (y == 0).sum())
    return X, y, s


# ──────────────────────────────────────────────
# NTU RGB+D 120 Skeleton Loader
# ──────────────────────────────────────────────

# NTU 25-joint indices mapped to COCO-compatible subset (17 joints)
# NTU joint ordering:  0=base_spine, 1=mid_spine, 2=neck, 3=head,
#   4=l_shoulder, 5=l_elbow, 6=l_wrist, 7=l_hand,
#   8=r_shoulder, 9=r_elbow, 10=r_wrist, 11=r_hand,
#   12=l_hip, 13=l_knee, 14=l_ankle, 15=l_foot,
#   16=r_hip, 17=r_knee, 18=r_ankle, 19=r_foot,
#   20=spine, 21=l_hand_tip, 22=l_thumb, 23=r_hand_tip, 24=r_thumb
NTU_TO_COCO = {
    KP_NOSE:       3,   # head → nose
    KP_L_EYE:      3,
    KP_R_EYE:      3,
    KP_L_EAR:      3,
    KP_R_EAR:      3,
    KP_L_SHOULDER: 4,
    KP_R_SHOULDER: 8,
    KP_L_ELBOW:    5,
    KP_R_ELBOW:    9,
    KP_L_WRIST:    6,
    KP_R_WRIST:    10,
    KP_L_HIP:      12,
    KP_R_HIP:      16,
    KP_L_KNEE:     13,
    KP_R_KNEE:     17,
    KP_L_ANKLE:    14,
    KP_R_ANKLE:    18,
}


def _parse_skeleton_file(path: Path):
    """
    Parse NTU .skeleton file into array of shape (T, 25, 3) [x, y, z].
    Uses the first body (person) per frame.
    Returns None on parse error.
    """
    try:
        lines = path.read_text().strip().splitlines()
    except Exception:
        return None

    idx = 0
    n_frames = int(lines[idx]); idx += 1
    frames = []

    for _ in range(n_frames):
        n_bodies = int(lines[idx]); idx += 1
        body_joints = None

        for b in range(n_bodies):
            idx += 1  # body info line
            n_joints = int(lines[idx]); idx += 1
            joints = np.zeros((n_joints, 3), dtype=np.float32)
            for j in range(n_joints):
                vals = lines[idx].split(); idx += 1
                joints[j, 0] = float(vals[0])   # x
                joints[j, 1] = float(vals[1])   # y
                joints[j, 2] = float(vals[2])   # z
            if body_joints is None:
                body_joints = joints

        if body_joints is not None:
            frames.append(body_joints)

    if len(frames) == 0:
        return None
    return np.stack(frames, axis=0)   # (T, 25, 3)


def _ntu_skeleton_to_pose_features(joints_25: np.ndarray) -> np.ndarray:
    """
    Convert NTU 25-joint frame to FEATURE_DIM feature vector.
    joints_25 : (25, 3)  [x, y, z] in camera space
    """
    # Map to COCO 17 keypoints using x,y only
    kps = np.zeros((NUM_KP, 2), dtype=np.float32)
    for coco_idx, ntu_idx in NTU_TO_COCO.items():
        kps[coco_idx] = joints_25[ntu_idx, :2]

    # Normalise to [0, 1] within joint bounding box
    xs, ys = kps[:, 0], kps[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    bw = max(x_max - x_min, 1e-6)
    bh = max(y_max - y_min, 1e-6)
    kps_norm = np.stack([(xs - x_min) / bw, (ys - y_min) / bh], axis=1)

    # All joints treated as high-confidence for skeleton data
    conf = np.ones(NUM_KP, dtype=np.float32)
    return extract_pose_features(kps_norm, conf)


def load_ntu(cfg: PipelineConfig):
    """
    NTU RGB+D 120 structure:
      DATASET/NTU/S###C###P###R###A###.skeleton
    A043 = fall down → label 1
    All other action IDs → label 0
    """
    ntu_root = Path(cfg.dataset_root) / "NTU"
    skeleton_files = sorted(ntu_root.glob("*.skeleton"))
    log.info("NTU: found %d skeleton files", len(skeleton_files))

    sequences = []
    skipped = 0

    for sf in tqdm(skeleton_files, desc="NTU", leave=False):
        # Parse action ID from filename: e.g. S001C001P001R001A043
        m = re.search(r"A(\d{3})", sf.name)
        if not m:
            skipped += 1
            continue
        action_id = int(m.group(1))
        label = 1 if action_id in cfg.ntu_fall_actions else 0

        joints_seq = _parse_skeleton_file(sf)
        if joints_seq is None or len(joints_seq) < cfg.window_size:
            skipped += 1
            continue
        if len(joints_seq) > cfg.ntu_max_frames:
            joints_seq = joints_seq[:cfg.ntu_max_frames]

        frame_features = [_ntu_skeleton_to_pose_features(f) for f in joints_seq]
        seq_arr = np.stack(frame_features, axis=0)
        seq_arr = add_velocity(seq_arr)

        windows = make_windows(seq_arr, label, cfg.window_size, cfg.stride)
        sequences.extend(windows)

    X = np.stack([w for w, _ in sequences], axis=0)
    y = np.array([l for _, l in sequences], dtype=np.int64)
    log.info("NTU: %d windows (fall=%d, non-fall=%d), skipped=%d",
             len(y), (y == 1).sum(), (y == 0).sum(), skipped)
    return X, y


# ──────────────────────────────────────────────
# Save / Load processed data
# ──────────────────────────────────────────────

def save_processed(X: np.ndarray, y: np.ndarray, name: str, out_root: str,
                   seq_ids: np.ndarray | None = None):
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sequences.npz"
    arrays = dict(X=X, y=y)
    if seq_ids is not None:
        arrays["seq_ids"] = seq_ids
    np.savez_compressed(str(out_path), **arrays)
    log.info("Saved %s → %s  shape=%s", name, out_path, X.shape)


def load_processed(name: str, out_root: str = "processed"):
    path = Path(out_root) / name / "sequences.npz"
    data = np.load(str(path))
    return data["X"], data["y"]


# ──────────────────────────────────────────────
# Normalisation (z-score per feature across time)
# ──────────────────────────────────────────────

def fit_normaliser(X: np.ndarray):
    """Compute mean/std over all windows and time steps. X: (N, T, F)."""
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0) + 1e-8
    return mean, std


def apply_normaliser(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

def run_pipeline(cfg: PipelineConfig, datasets: list[str]):
    extractor = None
    if any(d in datasets for d in ("urfd", "le2i")):
        extractor = PoseExtractor(
            cfg.pose_model_path, cfg.pose_conf, cfg.pose_iou, cfg.device
        )

    if "urfd" in datasets:
        log.info("=== Processing URFD ===")
        X, y, seq_ids = load_urfd(cfg, extractor)
        save_processed(X, y, "urfd", cfg.output_root, seq_ids=seq_ids)

    if "le2i" in datasets:
        log.info("=== Processing Le2i ===")
        X, y, seq_ids = load_le2i(cfg, extractor)
        save_processed(X, y, "le2i", cfg.output_root, seq_ids=seq_ids)

    if "ntu" in datasets:
        log.info("=== Processing NTU RGB+D 120 ===")
        X, y = load_ntu(cfg)
        save_processed(X, y, "ntu", cfg.output_root)

    log.info("Pipeline complete. Output in: %s/", cfg.output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fall Detection Data Pipeline")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["urfd", "le2i", "ntu"],
        choices=["urfd", "le2i", "ntu"],
        help="Which datasets to process",
    )
    parser.add_argument("--dataset-root",  default=CFG.dataset_root)
    parser.add_argument("--output-root",   default=CFG.output_root)
    parser.add_argument("--pose-model",    default=CFG.pose_model_path)
    parser.add_argument("--window-size",   type=int, default=CFG.window_size)
    parser.add_argument("--stride",        type=int, default=CFG.stride)
    parser.add_argument("--pose-conf",     type=float, default=CFG.pose_conf)
    parser.add_argument("--device",        default=CFG.device)
    args = parser.parse_args()

    CFG.dataset_root   = args.dataset_root
    CFG.output_root    = args.output_root
    CFG.pose_model_path = args.pose_model
    CFG.window_size    = args.window_size
    CFG.stride         = args.stride
    CFG.pose_conf      = args.pose_conf
    CFG.device         = args.device

    run_pipeline(CFG, args.datasets)
