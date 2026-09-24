"""Row-centered ReLU-cosine P vs global ncut vs n8 vs 50/50 mix.

Same mix=0 Key-curriculum probe as featcur_n8_vs_global_probe.py. The new
graph has no mix coefficient:

    τ_i = mean_{k≠i} ReLU(z_i^T z_k)
    W_ij = ReLU(z_i^T z_j - (τ_i + τ_j)/2),  W_ii = 0
    X^rel = P X,  P = row-normalize(W)

Isolated rows (row sum 0) keep the original token. Question: does centering
by the scene's own typical affinity stop global collapse on MOVi-C-like
frames without a glob_n8_mix knob?

Usage (inside the slotcurri image, a free GPU):
  python event_analysis/featcur_rowcenter_vs_global_probe.py --data-dir /workspace/dataset
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

METHODS = ("raw", "n8", "half", "center", "global")
METHOD_LABELS = {
    "raw": "raw",
    "n8": "n8",
    "half": "0.5 glob+n8",
    "center": "row-center",
    "global": "global",
}


@torch.no_grad()
def row_centered_relu_cosine_level(
    x: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """X^rel = P X with W_ij = ReLU(cos_ij - (τ_i+τ_j)/2).

    x: (B, N, D). Isolated patches keep the original token.
    """
    with torch.cuda.amp.autocast(enabled=False):
        xf = x.float()
        z = F.normalize(xf, dim=-1)
        cos = torch.bmm(z, z.transpose(1, 2))
        relu_cos = cos.clamp_min(0.0)
        relu_cos.diagonal(dim1=-2, dim2=-1).zero_()
        n_tokens = xf.shape[1]
        tau = relu_cos.sum(dim=-1) / max(n_tokens - 1, 1)
        w = (cos - 0.5 * (tau.unsqueeze(-1) + tau.unsqueeze(-2))).clamp_min(0.0)
        w.diagonal(dim1=-2, dim2=-1).zero_()
        row = w.sum(dim=-1, keepdim=True)
        rel = torch.bmm(w, xf)
        rel = torch.where(row < eps, xf, rel / row.clamp_min(eps))
        nnz = (w > 0).float().sum()
        n_off = float(w.shape[0] * n_tokens * (n_tokens - 1))
        isolated = (row.squeeze(-1) < eps).float().mean()
        stats = {
            "mean_tau": float(tau.mean().item()),
            "std_tau": float(tau.std(unbiased=False).item()),
            "edge_keep": float((nnz / max(n_off, 1.0)).item()),
            "isolated": float(isolated.item()),
        }
        return rel.to(dtype=x.dtype), stats


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
        half = 0.5 * global_x + 0.5 * n8_x
        center_x, cst = row_centered_relu_cosine_level(xb)
        center_x = center_x[0]
        feats = {
            "raw": raw,
            "n8": n8_x,
            "half": half,
            "center": center_x,
            "global": global_x,
        }
        rec = {
            "dataset": name,
            "clip": n_clips - 1,
            "n_obj": n_obj,
            "grid": grid,
            "mean_tau": cst["mean_tau"],
            "std_tau": cst["std_tau"],
            "edge_keep": cst["edge_keep"],
            "isolated": cst["isolated"],
            "cos_raw_n8": mean_token_cos(raw, n8_x),
            "cos_raw_half": mean_token_cos(raw, half),
            "cos_raw_center": mean_token_cos(raw, center_x),
            "cos_raw_global": mean_token_cos(raw, global_x),
            "cos_center_n8": mean_token_cos(center_x, n8_x),
            "cos_center_global": mean_token_cos(center_x, global_x),
            "cos_center_half": mean_token_cos(center_x, half),
        }
        for m in METHODS:
            for k, v in score(feats[m], gt, grid, bg_id).items():
                rec[f"{m}_{k}"] = v
        rows.append(rec)
        print(
            f"  clip {rec['clip']:3d} n_obj={n_obj:2d}  "
            f"τ={rec['mean_tau']:.3f} keep={rec['edge_keep']:.3f} iso={rec['isolated']:.3f}  "
            f"keys raw/n8/half/ctr/glob="
            f"{rec['raw_unique_keys']:.0f}/"
            f"{rec['n8_unique_keys']:.0f}/"
            f"{rec['half_unique_keys']:.0f}/"
            f"{rec['center_unique_keys']:.0f}/"
            f"{rec['global_unique_keys']:.0f}  "
            f"obj-cos "
            f"{rec['raw_obj_mean_cos']:.3f}/"
            f"{rec['n8_obj_mean_cos']:.3f}/"
            f"{rec['half_obj_mean_cos']:.3f}/"
            f"{rec['center_obj_mean_cos']:.3f}/"
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
                    "half": half.cpu(),
                    "center": center_x.cpu(),
                    "global": global_x.cpu(),
                    "gt": gt.cpu(),
                }
            )
    return rows, vis


def summarize(rows: List[dict]) -> Dict[str, float]:
    keys = [
        "mean_tau",
        "std_tau",
        "edge_keep",
        "isolated",
        "cos_raw_n8",
        "cos_raw_half",
        "cos_raw_center",
        "cos_raw_global",
        "cos_center_n8",
        "cos_center_global",
        "cos_center_half",
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
        f"  row-center τ={s['mean_tau']:.3f}±{s['std_tau']:.3f}  "
        f"edge keep={s['edge_keep']:.3f}  isolated={s['isolated']:.3f}"
    )
    print(
        "  token cosine to raw:  "
        f"n8 {s['cos_raw_n8']:.3f}  half {s['cos_raw_half']:.3f}  "
        f"center {s['cos_raw_center']:.3f}  global {s['cos_raw_global']:.3f}"
    )
    print(
        "  center vs:  "
        f"n8 {s['cos_center_n8']:.3f}  half {s['cos_center_half']:.3f}  "
        f"global {s['cos_center_global']:.3f}"
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
        "center should keep more keys than global on uniform scenes "
        "(high τ → sparse W) without a mix knob."
    )


def draw(name: str, frames: List[dict], out_path: str) -> None:
    if not frames:
        return
    cols: List[Tuple[str, str]] = [
        ("rgb", "RGB"),
        ("raw", "raw DINO"),
        ("n8", "n8"),
        ("half", "0.5 glob+n8"),
        ("center", "row-center"),
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
        f"{name}: PCA on raw. Row-centered ReLU-cosine P "
        "(no mix knob) vs n8 / 0.5 glob+n8 / dense global. mix=0.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("wrote", out_path, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/workspace/dataset")
    p.add_argument("--out-dir", default="event_analysis/featcur_rowcenter_vs_global")
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
            os.path.join(args.out_dir, f"{name}_pca_rowcenter.png"),
        )
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(all_summ, f, indent=2)
    print("wrote", os.path.join(args.out_dir, "summary.json"), flush=True)


if __name__ == "__main__":
    main()
