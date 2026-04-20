"""
02 — Data Validation & Quality Control
=======================================
Runs four checks on the processed URFD and Le2i sequences and saves all
figures to figures/.  Prints a go/no-go summary at the end.

Checks
------
1. Skeleton quality audit   — sequences with >30 % low-confidence frames
2. Class balance report     — exact counts + 70/15/15 stratified split preview
3. Feature distribution     — torso vertical alignment & CoM velocity,
                               fall vs non-fall (KS test)
4. Temporal coverage        — Le2i raw annotation fall durations
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────
PROCESSED_ROOT = Path("processed")
LE2I_ROOT      = Path("DATASET/LE2I")
FIGURES_DIR    = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ── Feature layout (must match data_pipeline.py) ───────────────────────────
# coords   : 0-33   (17 keypoints × 2)
# derived  : 34-40  (aspect_ratio, head_to_hip, hip_angle/180,
#                    left_knee/180, right_knee/180, torso_vert, upper_conf)
# velocity : 41-74  (Δ of coords 0-33)
IDX_UPPER_CONF  = 40
IDX_TORSO_VERT  = 39
VEL_OFFSET      = 41
LOW_CONF_THRESH = 0.3
LOW_CONF_FRAC   = 0.30   # flag sequence if > this fraction of frames are low-conf

LE2I_SCENES = [
    ("Coffee_room_01", "Coffee_room_01", "Annotation_files"),
    ("Coffee_room_02", "Coffee_room_02", "Annotations_files"),
    ("Home_01",        "Home_01",        "Annotation_files"),
    ("Home_02",        "Home_02",        "Annotation_files"),
]

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def load_dataset(name):
    path = PROCESSED_ROOT / name / "sequences.npz"
    d = np.load(str(path))
    return d["X"].astype(np.float32), d["y"].astype(np.int64)


def torso_angle_mean(X):
    """Mean torso_vert over the window. X: (N, T, F)"""
    return X[:, :, IDX_TORSO_VERT].mean(axis=1)


def com_velocity(X):
    """
    Mean CoM velocity magnitude per window.
    Velocity features are Δ of the 34 keypoint coords → indices 41-74.
    """
    vel = X[:, :, VEL_OFFSET:VEL_OFFSET + 34]   # (N, T, 34)
    vx  = vel[:, :, 0::2].mean(axis=2)           # (N, T)
    vy  = vel[:, :, 1::2].mean(axis=2)           # (N, T)
    return np.sqrt(vx**2 + vy**2).mean(axis=1)   # (N,)


# ──────────────────────────────────────────────────────────────────────────
# 0 · Load data
# ──────────────────────────────────────────────────────────────────────────

print("Loading processed datasets…")
X_urfd, y_urfd = load_dataset("urfd")
X_le2i, y_le2i = load_dataset("le2i")
print(f"  URFD  X={X_urfd.shape}  fall={y_urfd.sum()}  non-fall={(y_urfd==0).sum()}")
print(f"  Le2i  X={X_le2i.shape}  fall={y_le2i.sum()}  non-fall={(y_le2i==0).sum()}")

datasets = {"URFD": (X_urfd, y_urfd), "Le2i": (X_le2i, y_le2i)}


# ──────────────────────────────────────────────────────────────────────────
# 1 · Skeleton quality audit
# ──────────────────────────────────────────────────────────────────────────

print("\n─── 1. Skeleton Quality Audit ───")

def skeleton_audit(X, y, name):
    upper_conf    = X[:, :, IDX_UPPER_CONF]
    low_conf_frac = (upper_conf < LOW_CONF_THRESH).mean(axis=1)
    flagged       = low_conf_frac > LOW_CONF_FRAC
    return pd.DataFrame({
        "dataset":       name,
        "label":         y,
        "mean_conf":     upper_conf.mean(axis=1),
        "low_conf_frac": low_conf_frac,
        "flagged":       flagged,
    })

audit_urfd = skeleton_audit(X_urfd, y_urfd, "URFD")
audit_le2i = skeleton_audit(X_le2i, y_le2i, "Le2i")
audit_all  = pd.concat([audit_urfd, audit_le2i], ignore_index=True)

summary = (
    audit_all.groupby(["dataset", "label"])
    .agg(
        total    =("flagged", "count"),
        flagged  =("flagged", "sum"),
        flag_pct =("flagged", lambda x: round(100 * x.mean(), 1)),
        mean_conf=("mean_conf", lambda x: round(x.mean(), 3)),
    )
    .reset_index()
)
summary["label"] = summary["label"].map({0: "non-fall", 1: "fall"})
print(summary.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (name, df) in zip(axes, [("URFD", audit_urfd), ("Le2i", audit_le2i)]):
    for lbl, color, label in [(0, "steelblue", "non-fall"), (1, "tomato", "fall")]:
        ax.hist(df[df["label"] == lbl]["low_conf_frac"],
                bins=30, alpha=0.6, color=color, label=label, density=True)
    ax.axvline(LOW_CONF_FRAC, color="black", ls="--", lw=1.5,
               label=f"flag threshold ({LOW_CONF_FRAC:.0%})")
    ax.set_title(f"{name} — Low-confidence frame fraction")
    ax.set_xlabel("Fraction of frames with upper_conf < 0.3")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_skeleton_quality.png", dpi=150)
plt.close()
print("  → figures/01_skeleton_quality.png")


# ──────────────────────────────────────────────────────────────────────────
# 2 · Class balance report
# ──────────────────────────────────────────────────────────────────────────

print("\n─── 2. Class Balance Report ───")

y_combined = np.concatenate([y_urfd, y_le2i])

def balance_row(y, name):
    nf, f = int((y == 0).sum()), int((y == 1).sum())
    return dict(dataset=name, fall=f, non_fall=nf,
                total=len(y), fall_ratio=round(f / nf, 3))

bal_df = pd.DataFrame([
    balance_row(y_urfd, "URFD"),
    balance_row(y_le2i, "Le2i"),
    balance_row(y_combined, "Combined"),
])
print(bal_df.to_string(index=False))

def print_splits(y, name, splits=(0.70, 0.15, 0.15)):
    n = len(y)
    n_train = int(n * splits[0])
    n_val   = int(n * splits[1])
    n_test  = n - n_train - n_val
    print(f"\n  {name} (70/15/15):")
    for sn, sz in [("train", n_train), ("val", n_val), ("test", n_test)]:
        f = round(sz * y.mean())
        print(f"    {sn:5s}: {sz:5d}  fall={f:4d}  non-fall={sz-f:4d}")

for name, y in [("URFD", y_urfd), ("Le2i", y_le2i), ("Combined", y_combined)]:
    print_splits(y, name)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (name, y) in zip(axes, [("URFD", y_urfd), ("Le2i", y_le2i), ("Combined", y_combined)]):
    counts = [(y == 0).sum(), (y == 1).sum()]
    bars = ax.bar(["non-fall", "fall"], counts,
                  color=["steelblue", "tomato"], edgecolor="white")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{cnt:,}", ha="center", va="bottom", fontsize=11)
    ax.set_title(name)
    ax.set_ylabel("Window count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.suptitle("Class balance per dataset", fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_class_balance.png", dpi=150)
plt.close()
print("\n  → figures/02_class_balance.png")


# ──────────────────────────────────────────────────────────────────────────
# 3 · Feature distribution check
# ──────────────────────────────────────────────────────────────────────────

print("\n─── 3. Feature Distribution Check ───")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for col, (name, (X, y)) in enumerate(datasets.items()):
    torso = torso_angle_mean(X)
    com_v = com_velocity(X)

    for row, (feat, feat_label) in enumerate([
        (torso, "Torso vertical alignment"),
        (com_v, "CoM velocity magnitude"),
    ]):
        ax = axes[row, col]
        for lbl, color, label in [(0, "steelblue", "non-fall"), (1, "tomato", "fall")]:
            sns.kdeplot(feat[y == lbl], ax=ax, color=color,
                        label=label, fill=True, alpha=0.35)
        ks_stat, ks_p = stats.ks_2samp(feat[y == 0], feat[y == 1])
        status = "good" if ks_p < 0.01 else "WEAK"
        ax.set_title(f"{name} — {feat_label}\nKS={ks_stat:.3f}  p={ks_p:.2e}  [{status}]")
        ax.set_xlabel(feat_label)
        ax.set_ylabel("Density")
        ax.legend()
        print(f"  {name:6s}  {feat_label:30s}  KS={ks_stat:.3f}  p={ks_p:.2e}  [{status}]")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_feature_distributions.png", dpi=150)
plt.close()
print("  → figures/03_feature_distributions.png")


# ──────────────────────────────────────────────────────────────────────────
# 4 · Temporal coverage check (Le2i raw annotations)
# ──────────────────────────────────────────────────────────────────────────

print("\n─── 4. Temporal Coverage Check (Le2i) ───")

TOO_SHORT = 10
TOO_LONG  = 150

records = []
for scene_dir, inner_dir, ann_subdir in LE2I_SCENES:
    ann_dir = LE2I_ROOT / scene_dir / inner_dir / ann_subdir
    if not ann_dir.exists():
        print(f"  Missing annotation dir: {ann_dir}")
        continue
    for ann_file in sorted(ann_dir.glob("*.txt")):
        lines = ann_file.read_text(errors="replace").strip().splitlines()
        if len(lines) < 2:
            continue
        try:
            fall_start = int(lines[0].strip()) - 1
            fall_end   = int(lines[1].strip()) - 1
        except ValueError:
            continue
        total_frames = len(lines) - 2
        records.append({
            "scene":             scene_dir,
            "video":             ann_file.stem,
            "fall_start":        fall_start,
            "fall_end":          fall_end,
            "duration":          fall_end - fall_start + 1,
            "total_frames":      total_frames,
            "pre_fall_frames":   fall_start,
            "post_fall_frames":  max(0, total_frames - fall_end - 1),
        })

ann_df = pd.DataFrame(records)
print(f"  {len(ann_df)} annotated videos across {ann_df['scene'].nunique()} scenes")
print(f"\n  Fall duration stats (frames):")
print(ann_df["duration"].describe().round(1).to_string())

ann_df["flag_too_short"]    = ann_df["duration"] < TOO_SHORT
ann_df["flag_too_long"]     = ann_df["duration"] > TOO_LONG
ann_df["flag_out_of_range"] = (ann_df["fall_start"] < 0) | \
                               (ann_df["fall_end"] >= ann_df["total_frames"])
ann_df["flag_any"]          = ann_df[["flag_too_short", "flag_too_long",
                                       "flag_out_of_range"]].any(axis=1)

print(f"\n  Flagged too short  (<{TOO_SHORT} f): {ann_df['flag_too_short'].sum()}")
print(f"  Flagged too long   (>{TOO_LONG} f): {ann_df['flag_too_long'].sum()}")
print(f"  Flagged out-of-range:             {ann_df['flag_out_of_range'].sum()}")
print(f"  Flagged any:                      {ann_df['flag_any'].sum()} / {len(ann_df)}")

if ann_df["flag_any"].any():
    print("\n  Flagged videos:")
    print(ann_df[ann_df["flag_any"]][
        ["scene", "video", "fall_start", "fall_end", "duration", "total_frames"]
    ].to_string(index=False))

print("\n  Per-scene fall duration (frames):")
print(ann_df.groupby("scene")["duration"]
      .agg(videos="count", min="min", median="median", max="max")
      .to_string())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.hist(ann_df["duration"], bins=25, color="tomato", edgecolor="white")
ax.axvline(TOO_SHORT, color="navy", ls="--", lw=1.5, label=f"min={TOO_SHORT}")
ax.axvline(TOO_LONG,  color="navy", ls=":",  lw=1.5, label=f"max={TOO_LONG}")
ax.set_title("Fall duration (frames)")
ax.set_xlabel("Frames")
ax.set_ylabel("Count")
ax.legend()

ax = axes[1]
ax.hist(ann_df["pre_fall_frames"], bins=25, color="steelblue", edgecolor="white")
ax.set_title("Pre-fall context (frames before fall)")
ax.set_xlabel("Frames")
ax.set_ylabel("Count")

ax = axes[2]
ax.hist(ann_df["post_fall_frames"], bins=25, color="mediumseagreen", edgecolor="white")
ax.set_title("Post-fall context (frames after fall)")
ax.set_xlabel("Frames")
ax.set_ylabel("Count")

plt.suptitle("Le2i — Temporal coverage", fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_temporal_coverage.png", dpi=150)
plt.close()
print("  → figures/04_temporal_coverage.png")


# ──────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 68)
print("VALIDATION SUMMARY")
print("=" * 68)

for name, df in [("URFD", audit_urfd), ("Le2i", audit_le2i)]:
    pct = 100 * df["flagged"].mean()
    status = "OK" if pct < 5 else ("WARN" if pct < 15 else "ACTION REQUIRED")
    print(f"[Skeleton] {name:6s}  {df['flagged'].sum():4d}/{len(df)} flagged "
          f"({pct:.1f}%)  [{status}]")

for name, y in [("URFD", y_urfd), ("Le2i", y_le2i)]:
    ratio = y.mean()
    status = "OK" if 0.15 < ratio < 0.60 else "WARN — consider re-weighting"
    print(f"[Balance]  {name:6s}  fall ratio={ratio:.3f}  [{status}]")

for name, (X, y) in datasets.items():
    ks, p = stats.ks_2samp(torso_angle_mean(X)[y == 0], torso_angle_mean(X)[y == 1])
    status = "OK" if p < 0.01 else "WARN — weak torso signal"
    print(f"[Signal]   {name:6s}  torso KS={ks:.3f} p={p:.2e}  [{status}]")

n_flag_ann = int(ann_df["flag_any"].sum())
status = "OK" if n_flag_ann == 0 else f"WARN — {n_flag_ann} suspect annotation(s)"
print(f"[Temporal] Le2i    {n_flag_ann} annotation(s) flagged  [{status}]")

print()
print("DECISIONS TO RECORD BEFORE NOTEBOOK 03:")
print("  [ ] Low-conf sequences: exclude / interpolate / keep with sample weight")
print("  [ ] Class imbalance:    none / class weights / oversample minority")
print("  [ ] Flagged Le2i videos: exclude / re-label / keep as-is")
print("=" * 68)
