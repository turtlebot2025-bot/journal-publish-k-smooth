#!/usr/bin/env python3
"""
05_ablation_study.py
====================
Ablation study runner for FallDetectionNet.

This script reuses the existing project code in 03_train.py, keeps the same
Stage-B data loading, sequence-level StratifiedGroupKFold protocol, loss,
augmentation, temporal smoothing, and metric computation, and only changes the
model component under test.

Recommended paper ablations:
  1. full_b            : full FallDetectionNet-B, pretrained + full fine-tuning
  2. scratch_b         : same architecture trained from random initialization
  3. no_attention_b    : JointAttention modules replaced with Identity
  4. single_scale_b    : multi-scale temporal pooling replaced by global pooling

Examples
--------
# Quick smoke test on 10% of data, 1 fold only
python 05_ablation_study.py --debug --max-folds 1 --epochs 3 \
  --configs full_b,no_attention_b,single_scale_b,scratch_b \
  --resume outputs/checkpoints/stage_a_best.pt

# Full ablation run for the manuscript
python 05_ablation_study.py \
  --configs full_b,no_attention_b,single_scale_b,scratch_b \
  --resume outputs/checkpoints/stage_a_best.pt \
  --config research_config.json

Outputs
-------
outputs/ablation/ablation_fold_metrics.csv
outputs/ablation/ablation_summary.csv
outputs/ablation/ablation_table_latex.txt
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler


LOG = logging.getLogger("ablation")


def import_train_module(path: str = "03_train.py"):
    """Import 03_train.py even though the filename starts with a digit."""
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


class SingleScaleTemporalPooling(nn.Module):
    """Ablation: replace multi-scale temporal pooling with global average pooling."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, J) -> (B, C)
        return x.mean(dim=(-1, -2))


def replace_attention_with_identity(model: nn.Module, train_mod: Any) -> int:
    """Replace JointAttention layers inside model.backbone with nn.Identity."""
    replaced = 0
    for i, module in enumerate(model.backbone):
        if isinstance(module, train_mod.JointAttention):
            model.backbone[i] = nn.Identity()
            replaced += 1
    return replaced


def build_ablation_model(train_mod: Any, cfg: Any, device: torch.device, ablation: str) -> nn.Module:
    """Build FallDetectionNet with optional structural ablations."""
    model = train_mod.FallDetectionNet(
        num_classes=2,
        joint_embed_dim=cfg.joint_embed_dim,
        backbone_channels=cfg.backbone_channels,
        dilations=cfg.dilations,
        dropout=cfg.dropout,
        num_joints=cfg.num_joints,
        in_features=cfg.in_features,
    ).to(device)

    if ablation in {"no_attention_b", "no_attention_single_scale_b"}:
        n = replace_attention_with_identity(model, train_mod)
        LOG.info("Replaced %d JointAttention modules with Identity", n)

    if ablation in {"single_scale_b", "no_attention_single_scale_b"}:
        model.pool = SingleScaleTemporalPooling().to(device)
        model.head = model._build_head(cfg.backbone_channels, 2, cfg.dropout).to(device)
        LOG.info("Replaced multi-scale pooling with single global average pooling")

    return model


def load_backbone_if_needed(model: nn.Module, resume: str | None, device: torch.device) -> None:
    """Load Stage-A backbone weights while ignoring classification-head mismatches."""
    if not resume:
        return
    ckpt = torch.load(resume, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    # Do not load any old classification head. Also allow ablation-induced missing/unexpected keys.
    backbone_state = {k: v for k, v in state.items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    LOG.info(
        "Loaded backbone from %s (missing=%d, unexpected=%d)",
        resume, len(missing), len(unexpected),
    )


def configure_optimizer(train_mod: Any, cfg: Any, model: nn.Module, y_train: np.ndarray, variant: str, device: torch.device):
    fl_cfg = cfg.focal_loss
    criterion = train_mod.FocalLoss.from_labels(
        y_train,
        gamma=fl_cfg.gamma,
        eps=fl_cfg.label_smoothing,
        alpha_cap=fl_cfg.alpha_cap,
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    if variant == "a":
        for p in model.get_backbone_params():
            p.requires_grad = False
        cfg_v = cfg.stage_b_variant_a
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg_v.lr_head)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_v.num_epochs)
        return criterion, optimizer, scheduler, scaler, cfg_v.num_epochs, cfg_v.early_stopping_patience

    cfg_v = cfg.stage_b_variant_b
    optimizer = torch.optim.AdamW(
        [
            {"params": model.get_backbone_params(), "lr": cfg_v.lr_backbone},
            {"params": model.head.parameters(), "lr": cfg_v.lr_head},
        ],
        weight_decay=cfg_v.weight_decay,
    )
    scheduler = train_mod.cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=cfg_v.warmup_epochs,
        total_epochs=cfg_v.num_epochs,
    )
    return criterion, optimizer, scheduler, scaler, cfg_v.num_epochs, cfg_v.early_stopping_patience


def parse_config_name(name: str) -> dict[str, Any]:
    """Map a short config name to ablation choices."""
    valid = {
        "full_a": dict(variant="a", use_pretrain=True, structural="full"),
        "full_b": dict(variant="b", use_pretrain=True, structural="full"),
        "scratch_b": dict(variant="b", use_pretrain=False, structural="full"),
        "no_attention_b": dict(variant="b", use_pretrain=True, structural="no_attention_b"),
        "single_scale_b": dict(variant="b", use_pretrain=True, structural="single_scale_b"),
        "no_attention_single_scale_b": dict(variant="b", use_pretrain=True, structural="no_attention_single_scale_b"),
    }
    if name not in valid:
        raise ValueError(f"Unknown ablation config '{name}'. Valid: {', '.join(valid)}")
    return valid[name]


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


def run_one_ablation(
    name: str,
    train_mod: Any,
    cfg: Any,
    dm: Any,
    trainer: Any,
    X: np.ndarray,
    y: np.ndarray,
    seq_ids: np.ndarray,
    resume: str | None,
    seed: int,
    max_folds: int | None,
    override_epochs: int | None,
) -> list[dict[str, Any]]:
    spec = parse_config_name(name)
    variant = spec["variant"]
    structural = spec["structural"]
    use_resume = resume if spec["use_pretrain"] else None
    folds = dm.get_folds(y, seq_ids)
    if max_folds is not None:
        folds = folds[:max_folds]

    LOG.info("Running ablation=%s variant=%s structural=%s pretrained=%s", name, variant, structural, bool(use_resume))
    fold_thresholds: list[float] = []
    fold_test_probs: list[np.ndarray] = []
    fold_test_labels: list[np.ndarray] = []

    for fold_k, (train_val_idx, test_idx) in enumerate(folds):
        LOG.info("[%s] fold %d/%d", name, fold_k + 1, len(folds))

        # Same inner validation split style as 03_train.py.
        tv_seqs = seq_ids[train_val_idx]
        unique_seqs = np.unique(tv_seqs)
        rng = np.random.default_rng(seed + fold_k)
        rng.shuffle(unique_seqs)
        n_val_seqs = max(1, int(0.2 * len(unique_seqs)))
        val_seqs = set(unique_seqs[:n_val_seqs].tolist())
        val_mask = np.array([s in val_seqs for s in tv_seqs])
        tr_idx = train_val_idx[~val_mask]
        val_idx = train_val_idx[val_mask]

        loader_tr = dm.make_loader(X[tr_idx], y[tr_idx], seq_ids[tr_idx], True)
        loader_val = dm.make_loader(X[val_idx], y[val_idx], seq_ids[val_idx], False)
        loader_tst = dm.make_loader(X[test_idx], y[test_idx], seq_ids[test_idx], False)

        model = build_ablation_model(train_mod, cfg, trainer.device, structural)
        load_backbone_if_needed(model, use_resume, trainer.device)
        criterion, optimizer, scheduler, scaler, num_epochs, patience = configure_optimizer(
            train_mod, cfg, model, y[tr_idx], variant, trainer.device
        )
        if override_epochs is not None:
            num_epochs = int(override_epochs)
            patience = max(2, min(patience, int(override_epochs)))

        thr, test_probs, test_labels, history = trainer._train_fold(
            model,
            loader_tr,
            loader_val,
            loader_tst,
            criterion,
            optimizer,
            scheduler,
            scaler,
            num_epochs,
            patience,
            fold_k=fold_k,
            variant=f"ablation_{name}",
        )
        fold_thresholds.append(float(thr))
        fold_test_probs.append(test_probs)
        fold_test_labels.append(test_labels)

    # Evaluate with the same cross-fold threshold logic as 03_train.py.
    fold_results: list[dict[str, Any]] = []
    for fold_k in range(len(folds)):
        other_thr = [fold_thresholds[j] for j in range(len(folds)) if j != fold_k]
        cross_thr = float(np.mean(other_thr)) if other_thr else float(fold_thresholds[fold_k])
        smoothed = train_mod.temporal_smooth(fold_test_probs[fold_k], cfg.smooth_k)
        metrics = trainer.evaluator.compute_metrics(
            fold_test_labels[fold_k], smoothed, threshold=cross_thr, tag=f"{name}_fold{fold_k}"
        )
        metrics.update(
            dict(
                ablation=name,
                fold=fold_k,
                variant=variant,
                structural=structural,
                pretrained=bool(use_resume),
                own_threshold=float(fold_thresholds[fold_k]),
                cross_fold_threshold=float(cross_thr),
            )
        )
        fold_results.append(metrics)
    return fold_results


def write_outputs(results: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_csv = out_dir / "ablation_fold_metrics.csv"
    if not results:
        raise RuntimeError("No ablation results produced")

    keys = sorted({k for r in results for k in r.keys()})
    with fold_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    metrics = ["auc_roc", "auc_pr", "sensitivity", "specificity", "f1_fall", "accuracy", "fn", "fp"]
    by_name: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_name.setdefault(r["ablation"], []).append(r)

    summary_rows = []
    for name, rows in by_name.items():
        row: dict[str, Any] = {"ablation": name, "n_folds": len(rows)}
        for m in metrics:
            vals = [float(x[m]) for x in rows if m in x]
            mu, sd = mean_std(vals)
            row[f"{m}_mean"] = mu
            row[f"{m}_std"] = sd
        summary_rows.append(row)

    summary_csv = out_dir / "ablation_summary.csv"
    with summary_csv.open("w", newline="") as f:
        fieldnames = ["ablation", "n_folds"] + [f"{m}_{s}" for m in metrics for s in ["mean", "std"]]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    latex = out_dir / "ablation_table_latex.txt"
    with latex.open("w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\caption{Ablation study of FallDetectionNet components. Values are reported as mean $\\pm$ standard deviation over sequence-level folds.}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{@{}lllll@{}}\n\\toprule\n")
        f.write("Configuration & AUC-ROC & Sensitivity & Specificity & F1-score \\\\\n\\midrule\n")
        for r in summary_rows:
            f.write(
                f"{r['ablation'].replace('_', '-')} & "
                f"{r['auc_roc_mean']:.4f} $\\pm$ {r['auc_roc_std']:.4f} & "
                f"{r['sensitivity_mean']:.4f} $\\pm$ {r['sensitivity_std']:.4f} & "
                f"{r['specificity_mean']:.4f} $\\pm$ {r['specificity_std']:.4f} & "
                f"{r['f1_fall_mean']:.4f} $\\pm$ {r['f1_fall_std']:.4f} \\\\\n"
            )
        f.write("\\botrule\n\\end{tabular}\n\\end{table}\n")

    LOG.info("Wrote %s", fold_csv)
    LOG.info("Wrote %s", summary_csv)
    LOG.info("Wrote %s", latex)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FallDetectionNet ablation study")
    parser.add_argument("--train-file", default="03_train.py", help="Path to 03_train.py")
    parser.add_argument("--config", default="research_config.json", help="Path to research_config.json")
    parser.add_argument("--resume", default="outputs/checkpoints/stage_a_best.pt", help="Stage-A checkpoint path")
    parser.add_argument("--configs", default="full_b,no_attention_b,single_scale_b,scratch_b",
                        help="Comma-separated configs: full_a,full_b,scratch_b,no_attention_b,single_scale_b,no_attention_single_scale_b")
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Use 10%% of data via existing DataModule debug mode")
    parser.add_argument("--max-folds", type=int, default=None, help="Limit folds for smoke tests")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs for quick tests")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_mod = import_train_module(args.train_file)
    cfg = train_mod.load_config(args.config if Path(args.config).exists() else None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_mod.set_seed(args.seed)

    dm = train_mod.DataModule(cfg, device=device, debug=args.debug)
    evaluator = train_mod.Evaluator(Path(args.output_dir) / "figures")
    trainer = train_mod.Trainer(cfg, dm, evaluator, device=device, use_mlflow=not args.no_mlflow)

    X, y, seq_ids = dm.setup_stage_b()
    LOG.info("Loaded Stage-B data: X=%s y=%s sequences=%d", X.shape, y.shape, len(np.unique(seq_ids)))

    all_results: list[dict[str, Any]] = []
    for cfg_name in [c.strip() for c in args.configs.split(",") if c.strip()]:
        t0 = time.time()
        results = run_one_ablation(
            cfg_name, train_mod, cfg, dm, trainer, X, y, seq_ids,
            resume=args.resume, seed=args.seed, max_folds=args.max_folds,
            override_epochs=args.epochs,
        )
        for r in results:
            r["elapsed_config_s"] = time.time() - t0
        all_results.extend(results)

    write_outputs(all_results, Path(args.output_dir))


if __name__ == "__main__":
    main()
