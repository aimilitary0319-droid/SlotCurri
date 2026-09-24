"""Windowed ReLU-cosine P vs global ncut vs n8 vs 50/50 mix.

Same mix=0 Key-curriculum probe as featcur_n8_vs_global_probe.py, plus:

  r{k}:  W_ij = ReLU(z_i^T z_j) iff Chebyshev(i,j) <= k (absolute patch radius)
  half:  0.5 * X^global + 0.5 * X^n8

r=1 is the n8 graph. r=inf is global. The question is whether a mid radius
levels within an object without collapsing distant same-appearance instances,
and whether averaging the two extremes is actually an interpolation.

Usage (inside the slotcurri image, a free GPU):
  python event_analysis/featcur_window_vs_global_probe.py --data-dir /workspace/dataset
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_dino_vis import denorm_frame, pca_basis, pca_map, upsample  # noqa: E402
from featcur_n8_vs_global_probe import (  # noqa: E402
    DATASETS,
    mean_token_cos,
    nanmean,
    object_ids,
    score,
)
from featcur_strength_probe import extract_tokens, patch_ids_movi, patch_ids_ytvis  # noqa: E402
from slotcurri import configuration, data
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor

RADII = (3, 5, 8)
METHODS = ("raw", "n8", "r3", "r5", "r8", "half", "global")
METHOD_LABELS = {
    "raw": "raw",
    "n8": "n8 (r=1)",
    "r3": "window r=3",
    "r5": "window r=5",
    "r8": "window r=8",
    "half": "0.5 glob+n8",
    "global": "global",
}


def _cheb_mask(n_tokens: int, radius: int, device: torch.device) -> torch.Tensor:
    grid = int(round(math.sqrt(n_tokens)))
    if grid * grid != n_tokens:
        raise ValueError(f"window P expects a square grid, got N={n_tokens}")
    idx = torch.arange(n_tokens, device=device)
    ys, xs = idx // grid, idx % grid
    cheb = torch.maximum(
        (ys[:, None] - ys[None, :]).abs(), (xs[:, None] - xs[None, :]).abs()
    )
    return cheb > int(radius)


@torch.no_grad()
def window_relu_cosine_level(
    x: torch.Tensor, radius: int, eps: float = 1e-6
) -> torch.Tensor:
    """X^rel = P X, P = row-normalize ReLU-cosine on Chebyshev radius.

    x: (B, N, D). Isolated patches (row sum 0) keep the original token.
    Same P as barrier=false NcutRelationalLeveling, with W zero outside the window.
    """
    if radius <= 0:
        return x
    with torch.cuda.amp.autocast(enabled=False):
        xf = x.float()
        z = F.normalize(xf, dim=-1)
        w = torch.bmm(z, z.transpose(1, 2)).clamp_min(0.0)
        w.diagonal(dim1=-2, dim2=-1).zero_()
        w.masked_fill_(_cheb_mask(xf.shape[1], radius, xf.device), 0.0)
        row = w.sum(dim=-1, keepdim=True)
        rel = torch.bmm(w, xf)
        rel = torch.where(row < eps, xf, rel / row.clamp_min(eps))
        return rel.to(dtype=x.dtype)


@torch.no_grad()
def collect(name, cfg_path, min_obj, bg_id, data_dir, device, max_scan, n_vis):
    cfg = configuration.load_config(cfg_path)
    cfg.dataset.val_batch_size = 1
    cfg.dataset.num_val_workers = 0
    dm = data.build(cfg.dataset, data_dir=data_dir)
    dm.setup("validate")
    bb = cfg.model.encoder.backbone
    kwargs = dict(bb.get("model_kwargs") or {})
    backbone = (
        TimmExtractor(
            model=bb.model,
            pretrained=True,
            frozen=True,
            features=bb.features,
            model_kwargs=kwargs or None,
        )
        .to(device)
        .eval()
    )
    glob = NcutRelationalLeveling(chunk_size=4, n_iter=16, barrier=False).to(device)
    n8_mod = NcutRelationalLeveling(chunk_size=8, barrier=False, n8=True).to(device)
    id_fn = patch_ids_ytvis if name == "ytvis" else patch_ids_movi
    rows: List[dict] = []
    vis: List[dict] = []
    n_clips = 0
    checked_r1 = False
    print(f"\n=== {name}  scan≤{max_scan}  min_obj={min_obj} ===", flush=True)
    for batch in dm.val_dataloader():
        if n_clips >= max_scan:
            break
        if "batch_padding_mask" in batch and torch.is_tensor(
            batch["batch_padding_mask"]
        ) and bool(batch["batch_padding_mask"].any()):
            n_clips += 1
            continue
        video, seg = batch["video"][0], batch["segmentations"][0]
        ti = int(video.shape[0] // 2)
        frame = video[ti].to(device)
        raw = extract_tokens(backbone, frame.unsqueeze(0), bb.features).float()[0]
        grid = int(round(math.sqrt(raw.shape[0])))
        gt = id_fn(seg[ti].to(device), grid)
        n_obj = len(object_ids(gt, bg_id))
        n_clips += 1
        if n_obj < min_obj:
            continue
        xb = raw.unsqueeze(0)
        global_x = glob(xb, 0.0)[0]
        n8_x = n8_mod(xb, 0.0)[0]
        if not checked_r1:
            r1 = window_relu_cosine_level(xb, 1)[0]
            d = float((r1 - n8_x).abs().max().item())
            print(f"  sanity  window r=1 vs n8 module  max|Δ|={d:.3e}", flush=True)
            checked_r1 = True
        win = {f"r{r}": window_relu_cosine_level(xb, r)[0] for r in RADII}
        half = 0.5 * global_x + 0.5 * n8_x
        feats = {"raw": raw, "n8": n8_x, "half": half, "global": global_x, **win}
        rec = {
            "dataset": name,
            "clip": n_clips - 1,
            "n_obj": n_obj,
            "grid": grid,
            "cos_raw_n8": mean_token_cos(raw, n8_x),
            "cos_raw_r3": mean_token_cos(raw, win["r3"]),
            "cos_raw_r5": mean_token_cos(raw, win["r5"]),
            "cos_raw_r8": mean_token_cos(raw, win["r8"]),
            "cos_raw_half": mean_token_cos(raw, half),
            "cos_raw_global": mean_token_cos(raw, global_x),
        }
        for m in METHODS:
            for k, v in score(feats[m], gt, grid, bg_id).items():
                rec[f"{m}_{k}"] = v
        rows.append(rec)
        print(
            f"  clip {rec['clip']:3d} n_obj={n_obj:2d}  keys "
            f"raw/n8/r3/r5/r8/half/glob="
            f"{rec['raw_unique_keys']:.0f}/"
            f"{rec['n8_unique_keys']:.0f}/"
            f"{rec['r3_unique_keys']:.0f}/"
            f"{rec['r5_unique_keys']:.0f}/"
            f"{rec['r8_unique_keys']:.0f}/"
            f"{rec['half_unique_keys']:.0f}/"
            f"{rec['global_unique_keys']:.0f}  "
            f"obj-cos "
            f"{rec['raw_obj_mean_cos']:.3f}/"
            f"{rec['n8_obj_mean_cos']:.3f}/"
            f"{rec['r3_obj_mean_cos']:.3f}/"
            f"{rec['r5_obj_mean_cos']:.3f}/"
            f"{rec['r8_obj_mean_cos']:.3f}/"
            f"{rec['half_obj_mean_cos']:.3f}/"
            f"{rec['global_obj_mean_cos']:.3f}",
            flush=True,
        )
        if len(vis) < n_vis:
            vis.append(
                {
                    "clip": rec["clip"],
                    "n_obj": n_obj,
                    "grid": grid,
                    "rgb": denorm_frame(frame.cpu()),
                    "raw": raw.cpu(),
                    "n8": n8_x.cpu(),
                    "r3": win["r3"].cpu(),
                    "r5": win["r5"].cpu(),
                    "r8": win["r8"].cpu(),
                    "half": half.cpu(),
                    "global": global_x.cpu(),
                    "gt": gt.cpu(),
                }
            )
    return rows, vis


def summarize(rows: List[dict]) -> Dict[str, float]:
    keys = [
        "cos_raw_n8",
        "cos_raw_r3",
        "cos_raw_r5",
        "cos_raw_r8",
        "cos_raw_half",
        "cos_raw_global",
    ]
    for m in METHODS:
        keys.extend(
            [
                f"{m}_unique_keys",
                f"{m}_std",
                f"{m}_pair_cos",
                f"{m}_within_var",
                f"{m}_cohesion",
                f"{m}_obj_mean_cos",
                f"{m}_part_kmeans",
                f"{m}_part_spatial",
                f"{m}_p_other",
            ]
        )
    out = {"n_frames": float(len(rows))}
    for k in keys:
        out[k] = nanmean([float(r[k]) for r in rows])
    return out


def print_summary(name: str, s: Dict[str, float]) -> None:
    print(f"\n----- {name}  n={int(s['n_frames'])} frames (mix=0) -----")
    print(
        "  token cosine to raw:  "
        f"n8 {s['cos_raw_n8']:.3f}  r3 {s['cos_raw_r3']:.3f}  "
        f"r5 {s['cos_raw_r5']:.3f}  r8 {s['cos_raw_r8']:.3f}  "
        f"half {s['cos_raw_half']:.3f}  global {s['cos_raw_global']:.3f}"
    )
    hdr = "".join(f"{METHOD_LABELS[m]:>12s}" for m in METHODS)
    print(f"  {'':16s}{hdr}")
    for label, key in (
        ("unique keys", "unique_keys"),
        ("token std", "std"),
        ("pair cos", "pair_cos"),
        ("within-obj var", "within_var"),
        ("cohesion", "cohesion"),
        ("obj-mean cos", "obj_mean_cos"),
        ("part kmeans cos", "part_kmeans"),
        ("part spatial cos", "part_spatial"),
        ("P mass other-obj", "p_other"),
    ):
        vals = "".join(f"{s[f'{m}_{key}']:12.3f}" for m in METHODS)
        print(f"  {label:16s}{vals}")
    print(
        "  read: unique keys / obj-mean cos are the merge axes. "
        "half should sit near global if the scene-mean dominates. "
        "window r interpolates n8 → global on the graph support."
    )


def draw(name: str, frames: List[dict], out_path: str) -> None:
    if not frames:
        return
    cols: List[Tuple[str, str]] = [
        ("rgb", "RGB"),
        ("raw", "raw DINO"),
        ("n8", "n8  r=1"),
        ("r3", "window r=3"),
        ("r5", "window r=5"),
        ("r8", "window r=8"),
        ("half", "0.5 glob+n8"),
        ("global", "global ncut"),
    ]
    n_row = len(frames)
    fig, axes = plt.subplots(n_row, len(cols), figsize=(2.15 * len(cols), 2.55 * n_row))
    if n_row == 1:
        axes = np.array([axes])
    for r, fr in enumerate(frames):
        rgb, h, w = fr["rgb"], *fr["rgb"].shape[:2]
        grid = fr["grid"]
        raw = fr["raw"]
        basis = pca_basis(raw)
        proj = (raw.float() - raw.mean(0)) @ basis
        lo, hi = proj.quantile(0.02, dim=0), proj.quantile(0.98, dim=0)
        for c, (key, title) in enumerate(cols):
            if key == "rgb":
                im = rgb if rgb.max() <= 1.5 else rgb / 255.0
            else:
                im = upsample(pca_map(fr[key], basis, lo, hi, grid), h, w)
            axes[r, c].imshow(np.clip(im, 0, 1))
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(title, fontsize=9)
        axes[r, 0].set_ylabel(f"clip {fr['clip']}\n{fr['n_obj']} obj", fontsize=8)
    fig.suptitle(
        f"{name}: PCA on raw. Windowed ReLU-cosine P at Chebyshev r vs "
        "0.5·global+0.5·n8 vs dense global. mix=0.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/workspace/dataset")
    p.add_argument("--out-dir", default="event_analysis/featcur_window_vs_global")
    p.add_argument("--max-scan", type=int, default=40)
    p.add_argument("--n-vis", type=int, default=4)
    p.add_argument("--datasets", nargs="+", default=["ytvis", "movi_c"])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    all_summ = {}
    for name in args.datasets:
        cfg, min_obj, bg_id = DATASETS[name]
        rows, vis = collect(
            name, cfg, min_obj, bg_id, args.data_dir, device, args.max_scan, args.n_vis
        )
        csv_path = os.path.join(args.out_dir, f"{name}_per_frame.csv")
        if rows:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print("wrote", csv_path, flush=True)
        summ = summarize(rows)
        all_summ[name] = summ
        print_summary(name, summ)
        draw(
            name,
            vis,
            os.path.join(args.out_dir, f"{name}_pca_window_half_global.png"),
        )
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(all_summ, f, indent=2)
    print("wrote", os.path.join(args.out_dir, "summary.json"), flush=True)


if __name__ == "__main__":
    main()
