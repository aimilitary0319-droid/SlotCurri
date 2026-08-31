"""How DINO tokens change along the v33 mix schedule with the final rec operator.

Operator: tau=0.2, window=3, n_steps=3
Blend:    x_used = (1 - s) * x_smoothed + s * x_raw
s(t):     cosine 0 -> 1 over 30k steps  (feature_curriculum_mix)

Usage (inside the slotcurri container):
  python event_analysis/featcur_dino_vis_schedule.py --data-dir /workspace/dataset
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_dino_vis import object_crop, overlay_edges, pca_basis, pca_map, upsample  # noqa: E402
from featcur_dino_vis_parts import part_axis, part_cos  # noqa: E402
from featcur_strength_probe import apply_smooth  # noqa: E402
from slotcurri.models import feature_curriculum_mix

TAU = 0.2
WINDOW = 3
N_STEPS = 3
ANNEAL = 30000
SCHEDULE_STEPS = [0, 5000, 10000, 15000, 20000, 25000, 30000]


def mix_at(step: int) -> float:
    return feature_curriculum_mix(step, ANNEAL, "cosine")


def blend(raw: torch.Tensor, sm: torch.Tensor, mix: float) -> torch.Tensor:
    if mix >= 1.0:
        return raw
    if mix <= 0.0:
        return sm
    return (1.0 - mix) * sm + mix * raw


@torch.no_grad()
def attach_schedule(fr, device):
    raw = fr["raw"].to(device)
    sm = apply_smooth(raw, TAU, WINDOW, N_STEPS)
    ia, ib = fr["ia"].to(device), fr["ib"].to(device)
    axis = part_axis(raw, ia, ib)
    mid = 0.5 * (raw[ia].float().mean(0) + raw[ib].float().mean(0))
    grid = int(round(math.sqrt(raw.shape[0])))
    obj = torch.cat([ia, ib])
    raw_proj = ((raw.float() - mid) @ axis).reshape(grid, grid)
    ys, xs = obj // grid, obj % grid
    vmax = max(float(raw_proj[ys, xs].abs().quantile(0.95).item()), 1e-3)
    basis = pca_basis(raw)
    proj_raw = (raw.float() - raw.float().mean(0)) @ basis
    lo = proj_raw.quantile(0.02, dim=0)
    hi = proj_raw.quantile(0.98, dim=0)

    feats = {}
    part_maps = {}
    pca_maps = {}
    scores = {}
    for step in SCHEDULE_STEPS:
        m = mix_at(step)
        feat = blend(raw, sm, m)
        feats[step] = feat
        part_maps[step] = ((feat.float() - mid) @ axis).reshape(grid, grid).cpu().numpy()
        pca_maps[step] = pca_map(feat, basis, lo, hi, grid)
        scores[step] = part_cos(feat, ia, ib)

    fr["part_maps"] = part_maps
    fr["pca_maps"] = pca_maps
    fr["scores"] = scores
    fr["vmax"] = vmax
    fr["mixes"] = {step: mix_at(step) for step in SCHEDULE_STEPS}
    return fr


def _crop_box(fr):
    rgb = fr["rgb"]
    h, w = rgb.shape[:2]
    dummy = np.zeros((h, w, 3))
    crop = object_crop(rgb / 255.0, dummy, fr["gt_grid"], fr["oid"], pad=6)
    if crop is None:
        return 0, h, 0, w
    _, _, box = crop
    return box


def draw_mix_curve(ax) -> None:
    xs = np.linspace(0, ANNEAL, 300)
    ys = [mix_at(int(s)) for s in xs]
    ax.plot(xs / 1000.0, ys, color="black", lw=1.8)
    for step in SCHEDULE_STEPS:
        ax.scatter([step / 1000.0], [mix_at(step)], zorder=3, color="#c0392b", s=28)
        ax.annotate(
            f"{mix_at(step):.2f}",
            (step / 1000.0, mix_at(step)),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7,
        )
    ax.set_xlim(-0.5, 31)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("train step (k)", fontsize=8)
    ax.set_ylabel("mix s  (0=smoothed, 1=raw)", fontsize=8)
    ax.set_title(
        f"cosine 30k   ·   x = (1−s)·smooth(τ={TAU}, w={WINDOW}, steps={N_STEPS}) + s·raw",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)


def draw_part_axis(frames, out_path: str) -> None:
    n_row = len(frames)
    n_col = 1 + len(SCHEDULE_STEPS)
    fig = plt.figure(figsize=(2.55 * n_col, 3.05 * n_row + 1.6))
    gs = fig.add_gridspec(n_row + 1, n_col, height_ratios=[0.55] + [1] * n_row)
    axc = fig.add_subplot(gs[0, :])
    draw_mix_curve(axc)

    cmap = plt.get_cmap("coolwarm")
    for r, fr in enumerate(frames):
        y0, y1, x0, x1 = _crop_box(fr)
        rgb = fr["rgb"]
        ax = fig.add_subplot(gs[r + 1, 0])
        ax.imshow(rgb[y0:y1, x0:x1])
        ax.set_ylabel(f"clip {fr['clip']}", fontsize=9)
        ax.set_title("RGB", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        norm = TwoSlopeNorm(vmin=-fr["vmax"], vcenter=0.0, vmax=fr["vmax"])
        h, w = rgb.shape[:2]
        for c, step in enumerate(SCHEDULE_STEPS, start=1):
            m = upsample(fr["part_maps"][step][..., None], h, w)[y0:y1, x0:x1, 0]
            ax = fig.add_subplot(gs[r + 1, c])
            im = ax.imshow(m, cmap=cmap, norm=norm)
            mix = fr["mixes"][step]
            ax.set_title(f"{step//1000}k  s={mix:.2f}\npart-cos {fr['scores'][step]:.2f}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == n_col - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Part axis along the curriculum. Red/blue = raw head vs torso. "
        "Left = train start (fully smoothed). Right = 30k (raw DINO).",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def draw_pca(frames, out_path: str) -> None:
    n_row = len(frames)
    n_col = 1 + len(SCHEDULE_STEPS)
    fig, axes = plt.subplots(n_row, n_col, figsize=(2.4 * n_col, 2.45 * n_row))
    if n_row == 1:
        axes = axes[None, :]
    for r, fr in enumerate(frames):
        rgb = fr["rgb"]
        h, w = rgb.shape[:2]
        edge = fr.get("edge")
        show = overlay_edges(rgb, edge) if edge is not None else rgb / 255.0
        ax = axes[r, 0]
        ax.imshow(show)
        ax.set_ylabel(f"clip {fr['clip']}", fontsize=8)
        ax.set_title("RGB + GT", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for c, step in enumerate(SCHEDULE_STEPS, start=1):
            pca = upsample(fr["pca_maps"][step], h, w)
            if edge is not None:
                pca = overlay_edges(pca, edge, color=(1, 1, 1))
            ax = axes[r, c]
            ax.imshow(pca)
            if r == 0:
                ax.set_title(f"{step//1000}k\ns={fr['mixes'][step]:.2f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "Same PCA (fit on raw) along the 30k cosine schedule. "
        "Start = instance-like blob. End = raw part structure.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--min-patches", type=int, default=16)
    ap.add_argument("--out-dir", default="event_analysis/featcur_mix_sweep")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Reuse the hard part-split clips from the earlier figures.
    import featcur_dino_vis_parts as vp

    vp.TARGET_CLIPS = {"ytvis": [4, 50, 23, 18, 14, 8]}
    frames = vp.collect_targets(args.data_dir, device, args.min_patches)
    # edge from patch-id grid, nearest-upsampled, for PCA overlay
    for fr in frames:
        g = fr["gt_grid"].numpy()
        edge = np.zeros(g.shape, dtype=bool)
        edge[1:] |= g[1:] != g[:-1]
        edge[:, 1:] |= g[:, 1:] != g[:, :-1]
        h, w = fr["rgb"].shape[:2]
        fr["edge"] = (
            torch.from_numpy(edge.astype(np.float32))[None, None]
            .repeat(1, 1, 1, 1)
        )
        t = torch.from_numpy(edge.astype(np.float32))[None, None]
        t = F.interpolate(t, size=(h, w), mode="nearest")[0, 0].numpy() > 0.5
        fr["edge"] = t

    frames = [attach_schedule(fr, device) for fr in frames]
    print("mixes:", {s: f"{mix_at(s):.3f}" for s in SCHEDULE_STEPS}, flush=True)
    for fr in frames:
        print(
            f" clip {fr['clip']}: "
            + ", ".join(f"{s//1000}k={fr['scores'][s]:.3f}" for s in SCHEDULE_STEPS),
            flush=True,
        )

    part_frames = [fr for fr in frames if fr["clip"] in (4, 50, 23, 18)]
    draw_part_axis(part_frames, os.path.join(args.out_dir, "dino_schedule_part_axis.png"))
    draw_pca(frames, os.path.join(args.out_dir, "dino_schedule_pca.png"))
    print("wrote dino_schedule_part_axis.png\nwrote dino_schedule_pca.png", flush=True)


if __name__ == "__main__":
    main()
