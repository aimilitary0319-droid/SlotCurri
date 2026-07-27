"""Calibration data for the attention-mass curriculum thresholds on YTVIS-2021.

Measures two things on the validation set:
  1. GT object sizes as a fraction of the frame (= the mass a perfect single-object
     slot would need) -> tells us where p_start / p_end should sit.
  2. Per-slot attention mass of the current soft checkpoint (29k steps) -> confirms
     the current activation state (expected: 2 default slots split the scene).

Usage (inside the container):
  python event_analysis/ytvis_mass_calibration.py \
      [checkpoint_path] [n_gt_batches] [n_model_batches]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from slotcurri import configuration, data, models

CFG = "configs/slotcurri/ytvis2021_attnmass.yaml"
CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "logs/_ytvis_attnmass/checkpoints/slotcurri_step=step=29000.ckpt"
N_GT_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 100
N_MODEL_BATCHES = int(sys.argv[3]) if len(sys.argv) > 3 else 20

config = configuration.load_config(CFG)
dataset = data.build(config.dataset)
dataset.setup("validate")
loader = dataset.val_dataloader()

# ------------------------------------------------------------------
# 1) GT object sizes (fraction of pixels per object per frame)
# ------------------------------------------------------------------
obj_sizes = []          # per (frame, object) pixel fraction, background excluded
objs_per_frame = []
for bi, batch in enumerate(loader):
    if batch is None:
        continue
    if bi >= N_GT_BATCHES:
        break
    seg = batch["segmentations"]  # (B, T, C, H, W) one-hot (bool/int)
    if seg.dtype != torch.bool:
        seg = seg > 0.5
    B, T, C, H, W = seg.shape
    frac = seg.float().mean(dim=(-2, -1))  # (B, T, C) pixel fraction per channel
    # channel 0 = background in the ytvis one-hot masks
    fg = frac[:, :, 1:]
    present = fg > 0
    obj_sizes.append(fg[present].reshape(-1))
    objs_per_frame.append(present.sum(-1).float().reshape(-1))

obj_sizes = torch.cat(obj_sizes).numpy()
objs_per_frame = torch.cat(objs_per_frame).numpy()

print("=== GT object sizes (YTVIS val, %d batches) ===" % N_GT_BATCHES)
print("objects per frame: mean %.2f | median %.0f | max %.0f"
      % (objs_per_frame.mean(), np.median(objs_per_frame), objs_per_frame.max()))
qs = [5, 10, 25, 50, 75, 90, 95]
pct = np.percentile(obj_sizes, qs)
print("object size (fraction of frame):")
for q, v in zip(qs, pct):
    print("  p%02d: %.4f" % (q, v))
print("mean: %.4f" % obj_sizes.mean())
for thr in [0.35, 0.226, 0.15, 0.10, 0.071, 0.05, 0.03]:
    frac_above = (obj_sizes >= thr).mean()
    print("  objects with size >= %.3f: %5.1f%%" % (thr, 100 * frac_above))

# ------------------------------------------------------------------
# 2) Current checkpoint slot masses
# ------------------------------------------------------------------
print("\n=== slot attention mass @ checkpoint (%s) ===" % CKPT)
model = models.build(config.model, config.optimizer, None, None)
ckpt = torch.load(CKPT, map_location="cpu")
missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
print("load_state_dict: missing=%d unexpected=%d" % (len(missing), len(unexpected)))
model = model.cuda().eval()

n_default = len(model.amc_default_idx)
masses = []  # (N, S) per-frame slot masses
with torch.no_grad():
    for bi, batch in enumerate(loader):
        if batch is None:
            continue
        if bi >= N_MODEL_BATCHES:
            break
        if "batch_padding_mask" in batch:
            batch = model._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                continue
        batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        out = model.forward(batch, train=False, cycle=False)
        att = out["processor"]["state_attn_mask"]  # (B, T, S, F)
        mass = att.sum(-1) / att.shape[-1]          # (B, T, S)
        masses.append(mass.flatten(0, 1).cpu())

masses = torch.cat(masses).numpy()  # (N_frames, S)
S = masses.shape[1]
print("frames measured:", masses.shape[0], "| slots:", S, "| default:", model.amc_default_idx)
print("%6s %8s %8s %8s %8s" % ("slot", "mean", "p50", "p90", "max"))
for s in range(S):
    col = masses[:, s]
    tag = " (default)" if s in model.amc_default_idx else ""
    print("%6d %8.4f %8.4f %8.4f %8.4f%s"
          % (s, col.mean(), np.median(col), np.percentile(col, 90), col.max(), tag))
nd = masses[:, n_default:].reshape(-1)
print("non-default slot mass: mean %.4f | p90 %.4f | p99 %.4f | max %.4f"
      % (nd.mean(), np.percentile(nd, 90), np.percentile(nd, 99), nd.max()))

# ------------------------------------------------------------------
# plot
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
ax.hist(obj_sizes, bins=np.logspace(np.log10(max(obj_sizes.min(), 1e-4)), 0, 60))
ax.set_xscale("log")
for thr, c, l in [(0.35, "r", "old p_start 0.35"), (0.071, "orange", "p_end 0.071"),
                  (0.15, "g", "candidate p_start 0.15"), (0.03, "b", "candidate p_end 0.03")]:
    ax.axvline(thr, color=c, ls="--", label=l)
ax.set_xlabel("GT object size (fraction of frame)")
ax.set_ylabel("# (frame, object)")
ax.set_title("YTVIS val: GT object sizes")
ax.legend(fontsize=8)

ax = axes[1]
for s in range(S):
    tag = "default" if s in model.amc_default_idx else None
    ax.hist(masses[:, s], bins=60, range=(0, 1), histtype="step",
            label=f"slot {s}" + (" (def)" if tag else ""))
ax.set_xlabel("slot attention mass")
ax.set_ylabel("# frames")
ax.set_title("current ckpt (29k): per-slot mass")
ax.legend(fontsize=7)
fig.tight_layout()
out_png = "event_analysis/ytvis_mass_calibration.png"
fig.savefig(out_png, dpi=130)
print("\nsaved plot ->", out_png)
