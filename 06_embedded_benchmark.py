#!/usr/bin/env python3
"""
06_embedded_benchmark.py
========================
Hardware benchmark script for FallDetectionNet and optional YOLOv8-pose inference.

Use this file on the same GPU machine used for the paper, and also copy it to
embedded targets such as Jetson Orin/Nano or Raspberry Pi + accelerator to produce
real deployment evidence.

Examples
--------
# Classifier-only benchmark from a checkpoint
python 06_embedded_benchmark.py --checkpoint outputs/checkpoints/best_fold0_auc0.963_variantb.pt

# Benchmark classifier + YOLOv8 pose on an image
python 06_embedded_benchmark.py --checkpoint outputs/checkpoints/best_fold0_auc0.963_variantb.pt \
  --include-yolo --image sample.jpg --yolo-model yolov8n-pose.pt

# On Jetson/RPi, run the same command and compare outputs/benchmarks/*.json
python 06_embedded_benchmark.py --device auto --batch-size 1 --iters 500

Outputs
-------
outputs/benchmarks/benchmark_<timestamp>.json
outputs/benchmarks/benchmark_<timestamp>.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import platform
import statistics as stats
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def import_train_module(path: str = "03_train.py"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Could not find {p.resolve()}")
    spec = importlib.util.spec_from_file_location("falldetect_train", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def shell(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5).strip()
    except Exception:
        return None


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
        })
    # Jetson-specific hints if available.
    info["tegrastats_available"] = shell(["bash", "-lc", "command -v tegrastats || true"])
    info["nvidia_smi"] = shell(["bash", "-lc", "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1"])
    return info


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model(train_mod: Any, config_path: str | None, checkpoint: str | None, device: torch.device):
    cfg = train_mod.load_config(config_path if config_path and Path(config_path).exists() else None)
    model = train_mod.FallDetectionNet(
        num_classes=2,
        joint_embed_dim=cfg.joint_embed_dim,
        backbone_channels=cfg.backbone_channels,
        dilations=cfg.dilations,
        dropout=cfg.dropout,
        num_joints=cfg.num_joints,
        in_features=cfg.in_features,
    ).to(device)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint {checkpoint} (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()
    return model, cfg


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(np.asarray(xs), p))


def benchmark_torch_model(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    iters: int,
    warmup: int,
    use_fp16: bool,
) -> dict[str, float]:
    x = torch.randn(batch_size, 7, 30, 17, device=device)
    if use_fp16 and device.type == "cuda":
        model = model.half()
        x = x.half()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times_ms: list[float] = []
    with torch.no_grad():
        for _ in range(iters):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(x)
                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)
            else:
                t0 = time.perf_counter()
                _ = model(x)
                elapsed = (time.perf_counter() - t0) * 1000.0
            times_ms.append(float(elapsed))

    mean_ms = float(stats.mean(times_ms))
    med_ms = float(stats.median(times_ms))
    return {
        "classifier_mean_ms": mean_ms,
        "classifier_median_ms": med_ms,
        "classifier_p95_ms": percentile(times_ms, 95),
        "classifier_p99_ms": percentile(times_ms, 99),
        "classifier_fps_windows": 1000.0 / mean_ms if mean_ms > 0 else float("nan"),
    }


def try_flops(model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    """Try ptflops first. If unavailable, return null values."""
    try:
        from ptflops import get_model_complexity_info  # type: ignore
        # ptflops expects input shape without batch.
        macs, params = get_model_complexity_info(
            model,
            (7, 30, 17),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        return {"classifier_macs": float(macs), "classifier_flops_approx": float(macs) * 2.0, "ptflops_params": int(params)}
    except Exception as e:
        return {"classifier_macs": None, "classifier_flops_approx": None, "ptflops_error": str(e)}


def benchmark_yolo(image_path: str | None, model_path: str, device: str, iters: int, warmup: int) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        return {"yolo_error": f"ultralytics/cv2 import failed: {e}"}

    if image_path and Path(image_path).exists():
        img = cv2.imread(image_path)
        if img is None:
            return {"yolo_error": f"Could not read image: {image_path}"}
    else:
        # Fallback synthetic frame. Useful for device speed smoke tests, but not a realistic pose workload.
        img = np.zeros((480, 640, 3), dtype=np.uint8)

    model = YOLO(model_path)
    times_ms: list[float] = []
    for _ in range(warmup):
        _ = model.predict(img, conf=0.3, iou=0.45, device=device, verbose=False)
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model.predict(img, conf=0.3, iou=0.45, device=device, verbose=False)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = float(stats.mean(times_ms))
    return {
        "yolo_mean_ms_per_frame": mean_ms,
        "yolo_median_ms_per_frame": float(stats.median(times_ms)),
        "yolo_p95_ms_per_frame": percentile(times_ms, 95),
        "yolo_fps_frames": 1000.0 / mean_ms if mean_ms > 0 else float("nan"),
        "yolo_model": model_path,
        "yolo_image": image_path or "synthetic_blank_640x480",
    }


def write_outputs(results: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"benchmark_{ts}.json"
    csv_path = out_dir / f"benchmark_{ts}.csv"
    json_path.write_text(json.dumps(results, indent=2))
    flat = flatten_dict(results)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
    return json_path, csv_path


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FallDetectionNet on GPU/CPU/embedded devices")
    parser.add_argument("--train-file", default="03_train.py")
    parser.add_argument("--config", default="research_config.json")
    parser.add_argument("--checkpoint", default=None, help="Optional Stage-B checkpoint")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--fp16", action="store_true", help="Use FP16 for classifier on CUDA")
    parser.add_argument("--include-yolo", action="store_true", help="Also benchmark YOLOv8-pose")
    parser.add_argument("--yolo-model", default="yolov8n-pose.pt")
    parser.add_argument("--image", default=None, help="Representative frame for YOLO benchmark")
    parser.add_argument("--output-dir", default="outputs/benchmarks")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")

    train_mod = import_train_module(args.train_file)
    model, cfg = build_model(train_mod, args.config, args.checkpoint, device)
    total_params, trainable_params = count_params(model)

    print(f"Device: {device}")
    print(f"Parameters: total={total_params:,}, trainable={trainable_params:,}")

    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hardware_info(),
        "settings": vars(args),
        "model": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "input_shape": [args.batch_size, 7, 30, 17],
        },
    }

    results["classifier"] = benchmark_torch_model(
        model=model,
        device=device,
        batch_size=args.batch_size,
        iters=args.iters,
        warmup=args.warmup,
        use_fp16=args.fp16,
    )
    results["classifier"].update(try_flops(model, device))

    if args.include_yolo:
        yolo_res = benchmark_yolo(
            image_path=args.image,
            model_path=args.yolo_model,
            device="0" if device.type == "cuda" else "cpu",
            iters=max(20, min(args.iters, 200)),
            warmup=max(5, min(args.warmup, 20)),
        )
        results["yolo"] = yolo_res
        if "yolo_mean_ms_per_frame" in yolo_res:
            cls_ms = results["classifier"]["classifier_mean_ms"]
            yolo_ms = yolo_res["yolo_mean_ms_per_frame"]
            results["end_to_end_estimate"] = {
                "estimated_ms_per_window": float(yolo_ms + cls_ms),
                "note": "Estimated as one pose frame benchmark plus one classifier window benchmark. For deployment, benchmark the complete streaming pipeline on the target device.",
            }

    json_path, csv_path = write_outputs(results, Path(args.output_dir))
    print(json.dumps(results, indent=2))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
