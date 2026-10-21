"""Show the two Ncut groups on the image.

Columns: RGB + GT edges, median cut of v, mean cut of y = D^{-1/2} v,
patches whose group changed. Blue and red are the two groups. The sign of
the eigenvector is arbitrary, so the colors are not foreground and background.

Usage (inside the slotcurri image):
  python event_analysis/featcur_group_vis.py --data-dir /workspace/dataset
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_dino_vis import (  # noqa: E402
    denorm_frame,
    instance_contours,
    overlay_edges,
    pca_basis,
    pca_map,
    upsample,
)
from featcur_strength_probe import extract_tokens, patch_ids_movi, patch_ids_ytvis  # noqa: E402
from featcur_xrel_probe import DATASETS, object_ids  # noqa: E402
from slotcurri import configuration, data  # noqa: E402
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor  # noqa: E402

BLUE = np.array([0.17, 0.48, 0.71], dtype=np.float32)
RED = np.array([0.84, 0.10, 0.11], dtype=np.float32)


def tint(rgb: np.ndarray, region_n: torch.Tensor, grid: int, alpha: float = 0.55) -> np.ndarray:
    hw = region_n.detach().cpu().numpy().reshape(grid, grid).astype(np.float32)
    mask = upsample(hw[..., None].repeat(3, axis=-1), rgb.shape[0], rgb.shape[1])[..., 0]
    color = np.where(mask[..., None] > 0.5, RED, BLUE)
    base = rgb.astype(np.float32) / 255.0
    return np.clip((1.0 - alpha) * base + alpha * color, 0, 1)


def disagree_map(rgb, med, rw, grid) -> np.ndarray:
    diff = (med != rw).detach().cpu().numpy().reshape(grid, grid).astype(np.float32)
    mask = upsample(diff[..., None].repeat(3, axis=-1), rgb.shape[0], rgb.shape[1])[..., 0]
    base = rgb.astype(np.float32) / 255.0
    mark = np.array([0.95, 0.85, 0.15], dtype=np.float32)
    out = base.copy()
    out[mask > 0.5] = 0.35 * out[mask > 0.5] + 0.65 * mark
    return np.clip(out, 0, 1)


@torch.no_grad()
def collect(name, data_dir, device, max_scan, n_keep):
    cfg_path, min_obj, bg_id, _label = DATASETS[name]
    cfg = configuration.load_config(cfg_path)
    cfg.dataset.val_batch_size = 1
    cfg.dataset.num_val_workers = 0
    dm = data.build(cfg.dataset, data_dir=data_dir)
    dm.setup("validate")
    bb = cfg.model.encoder.backbone
    backbone = TimmExtractor(
        model=bb.model,
        pretrained=True,
        frozen=True,
        features=bb.features,
        model_kwargs=dict(bb.get("model_kwargs") or {}) or None,
    ).to(device).eval()
    med_cut = NcutRelationalLeveling(chunk_size=1, n_iter=16, barrier=True).to(device)
    rw_cut = NcutRelationalLeveling(
        chunk_size=1, n_iter=16, barrier=True, fiedler_cut="rw_mean"
    ).to(device)
    id_fn = patch_ids_ytvis if name == "ytvis" else patch_ids_movi
    rows = []
    n_clips = 0
    for batch in dm.val_dataloader():
        if n_clips >= max_scan or len(rows) >= n_keep:
            break
        if "batch_padding_mask" in batch and torch.is_tensor(batch["batch_padding_mask"]) and bool(
            batch["batch_padding_mask"].any()
        ):
            n_clips += 1
            continue
        video, seg = batch["video"][0], batch["segmentations"][0]
        frame_i = int(video.shape[0] // 2)
        frame = video[frame_i]
        raw = extract_tokens(backbone, frame.unsqueeze(0).to(device), bb.features).float()[0]
        grid = int(round(math.sqrt(raw.shape[0])))
        gt = id_fn(seg[frame_i].to(device), grid)
        n_obj = len(object_ids(gt, bg_id, min_n=1))
        n_clips += 1
        if n_obj < min_obj:
            continue
        med = med_cut.explain(raw.unsqueeze(0))["region"][0]
        rw = rw_cut.explain(raw.unsqueeze(0))["region"][0]
        rows.append(
            {
                "clip": n_clips - 1,
                "n_obj": n_obj,
                "rgb": denorm_frame(frame),
                "edge": instance_contours(seg[frame_i], bg_id),
                "grid": grid,
                "med": med.cpu(),
                "rw": rw.cpu(),
                "raw": raw.cpu(),
                "flip": float((med != rw).float().mean()),
            }
        )
        print(f"  {name} clip {n_clips - 1} n_obj={n_obj} flip={rows[-1]['flip']:.2f}", flush=True)
    return rows


def pick(rows, k=4):
    if len(rows) <= k:
        return rows
    ordered = sorted(rows, key=lambda row: (row["n_obj"], row["flip"]))
    idx = np.linspace(0, len(ordered) - 1, k).round().astype(int)
    chosen = [ordered[i] for i in idx]
    # Always include the frame whose groups change the most.
    top = max(rows, key=lambda row: row["flip"])
    if all(item["clip"] != top["clip"] for item in chosen):
        chosen[-1] = top
    return chosen


def within_group_px(raw: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    """Row-normalize ReLU-cosine only inside the group. Empty rows keep X."""
    z = F.normalize(raw.float(), dim=-1)
    weight = (z @ z.T).clamp_min(0)
    weight.fill_diagonal_(0)
    same = region.unsqueeze(0) == region.unsqueeze(1)
    weight = weight.masked_fill(~same, 0)
    row = weight.sum(-1, keepdim=True)
    mixed = weight @ raw.float() / row.clamp_min(1e-6)
    return torch.where(row < 1e-6, raw.float(), mixed)


def n_colors(tokens: torch.Tensor, thr: float = 0.98) -> int:
    xn = F.normalize(tokens.float(), dim=-1)
    kept = []
    for vec in xn:
        if not kept or float((torch.stack(kept) @ vec).max()) < thr:
            kept.append(vec)
    return len(kept)


def draw_px(name, rows, out_path):
    """PX inside the y-mean groups. PCA basis is fit on raw so one color means one feature."""
    titles = ["RGB + GT", "mean of y", "raw DINO", "PX inside groups"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(11.2, 2.7 * len(rows)))
    if len(rows) == 1:
        axes = np.array([axes])
    for r, fr in enumerate(rows):
        rgb, edge, grid = fr["rgb"], fr["edge"], fr["grid"]
        raw = fr["raw"]
        rel = within_group_px(raw, fr["rw"])
        basis = pca_basis(raw)
        centered = raw.float() - raw.float().mean(0)
        proj = centered @ basis
        lo, hi = proj.quantile(0.02, dim=0), proj.quantile(0.98, dim=0)
        h, w = rgb.shape[:2]
        panels = [
            overlay_edges(rgb, edge),
            overlay_edges(tint(rgb, fr["rw"], grid), edge, (1, 1, 0.2)),
            overlay_edges(upsample(pca_map(raw, basis, lo, hi, grid), h, w), edge, (1, 1, 1)),
            overlay_edges(upsample(pca_map(rel, basis, lo, hi, grid), h, w), edge, (1, 1, 1)),
        ]
        keys = (n_colors(raw), n_colors(rel))
        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(f"clip {fr['clip']}\n{fr['n_obj']} obj", fontsize=9)
            if c == 2:
                ax.set_xlabel(f"colors {keys[0]}", fontsize=8)
            if c == 3:
                ax.set_xlabel(f"colors {keys[1]}", fontsize=8)
    fig.suptitle(
        f"{name}: PX averages ReLU-cosine only inside each y-mean group. "
        "PCA colors are the raw basis. Yellow lines are GT edges.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("wrote", out_path, flush=True)


def draw(name, rows, out_path):
    titles = ["RGB + GT", "median of v", "mean of y", "group changed"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(11.2, 2.7 * len(rows)))
    if len(rows) == 1:
        axes = np.array([axes])
    for r, fr in enumerate(rows):
        rgb, edge, grid = fr["rgb"], fr["edge"], fr["grid"]
        panels = [
            overlay_edges(rgb, edge),
            overlay_edges(tint(rgb, fr["med"], grid), edge, (1, 1, 0.2)),
            overlay_edges(tint(rgb, fr["rw"], grid), edge, (1, 1, 0.2)),
            overlay_edges(disagree_map(rgb, fr["med"], fr["rw"], grid), edge, (1, 1, 1)),
        ]
        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(
                    f"clip {fr['clip']}\n{fr['n_obj']} obj\nflip {fr['flip']:.0%}",
                    fontsize=9,
                )
    fig.suptitle(
        f"{name}: blue / red are the two Ncut groups. "
        "Yellow lines are GT instance edges. "
        "y = D^{-1/2} v, cut at mean(y).",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("wrote", out_path, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/workspace/dataset")
    parser.add_argument("--out-dir", default="event_analysis/featcur_fiedler_mean")
    parser.add_argument("--datasets", default="ytvis,movi_c,movi_e")
    parser.add_argument("--max-scan", type=int, default=24)
    parser.add_argument("--n-keep", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.datasets.split(","):
        rows = collect(name, args.data_dir, device, args.max_scan, args.n_keep)
        chosen = pick(rows)
        draw(name, chosen, os.path.join(args.out_dir, f"{name}_groups.png"))
        draw_px(name, chosen, os.path.join(args.out_dir, f"{name}_px.png"))


if __name__ == "__main__":
    main()
