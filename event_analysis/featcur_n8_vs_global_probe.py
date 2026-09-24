"""Compare DINO tokens: raw vs global ncut (v39 FC) vs 8-neighbor n8 FC.

Fully leveled (mix=0). Gate graph is already n8; this asks how much the Key
curriculum actually changes appearance if P is dense W vs 8-nbr R vs identity.

Usage (inside the slotcurri image, a free GPU):
  python event_analysis/featcur_n8_vs_global_probe.py --data-dir /workspace/dataset
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_dino_vis import denorm_frame, pca_basis, pca_map, upsample  # noqa: E402
from featcur_ncut_vis_movi import unique_keys  # noqa: E402
from featcur_strength_probe import (  # noqa: E402
    extract_tokens,
    kmeans2,
    patch_ids_movi,
    patch_ids_ytvis,
    spatial_halves,
)
from slotcurri import configuration, data
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor

DATASETS = {
    "ytvis": ("configs/slotcurri/ytvis2021_attnmass_v39lam1gu.yaml", 2, -1),
    "movi_c": ("configs/slotcurri/movi_c_attnmass_v39lam1gu.yaml", 3, 0),
}
METHODS = ("raw", "global", "n8")


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.float(), dim=0)
    b = F.normalize(b.float(), dim=0)
    return float((a * b).sum().clamp(-1.0, 1.0).item())


def mean_offdiag_cos(x: torch.Tensor) -> float:
    xn = F.normalize(x.float(), dim=-1)
    n = xn.shape[0]
    s = xn @ xn.T
    return float((s.sum() - n) / (n * (n - 1)))


def token_std(x: torch.Tensor) -> float:
    return float(x.float().std(unbiased=False).item())


def mean_token_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    an = F.normalize(a.float(), dim=-1)
    bn = F.normalize(b.float(), dim=-1)
    return float((an * bn).sum(-1).mean().item())


def object_ids(gt: torch.Tensor, bg_id: int) -> List[int]:
    return [int(i) for i in gt.unique().tolist() if int(i) != bg_id]


def within_object_var(x: torch.Tensor, gt: torch.Tensor, bg_id: int, min_n: int = 8) -> float:
    vals = []
    for oid in object_ids(gt, bg_id):
        m = gt == oid
        if int(m.sum()) < min_n:
            continue
        vals.append(float(x[m].float().var(dim=0, unbiased=False).mean().item()))
    return float(np.mean(vals)) if vals else float("nan")


def within_object_cohesion(x: torch.Tensor, gt: torch.Tensor, bg_id: int, min_n: int = 8) -> float:
    """Mean cosine of object patches to that object's mean token."""
    vals = []
    xn = F.normalize(x.float(), dim=-1)
    for oid in object_ids(gt, bg_id):
        m = gt == oid
        if int(m.sum()) < min_n:
            continue
        mu = F.normalize(xn[m].mean(0), dim=0)
        vals.append(float((xn[m] @ mu).mean().item()))
    return float(np.mean(vals)) if vals else float("nan")


def object_mean_cos(x: torch.Tensor, gt: torch.Tensor, bg_id: int, min_n: int = 4) -> float:
    means = []
    for oid in object_ids(gt, bg_id):
        m = gt == oid
        if int(m.sum()) < min_n:
            continue
        means.append(F.normalize(x[m].float().mean(0), dim=0))
    if len(means) < 2:
        return float("nan")
    M = torch.stack(means)
    s = M @ M.T
    k = s.shape[0]
    return float((s.sum() - k) / (k * (k - 1)))


def part_cos_kmeans(x: torch.Tensor, gt: torch.Tensor, bg_id: int, min_n: int = 16) -> float:
    """2-means on RAW assignment? Use labels from raw for fairness — here labels
    from the *current* x so we measure residual part gap after leveling."""
    vals = []
    xn = F.normalize(x.float(), dim=-1)
    for oid in object_ids(gt, bg_id):
        idx = (gt == oid).nonzero(as_tuple=False).squeeze(-1)
        if int(idx.numel()) < min_n:
            continue
        lab = kmeans2(xn[idx])
        if lab is None:
            continue
        a = x[idx[lab == 0]].mean(0)
        b = x[idx[lab == 1]].mean(0)
        vals.append(_cos(a, b))
    return float(np.mean(vals)) if vals else float("nan")


def part_cos_spatial(x: torch.Tensor, gt: torch.Tensor, grid: int, bg_id: int, min_n: int = 16) -> float:
    vals = []
    for oid in object_ids(gt, bg_id):
        m = gt == oid
        if int(m.sum()) < min_n:
            continue
        flat = m.view(grid, grid)
        ys, xs = flat.nonzero(as_tuple=True)
        lab = spatial_halves(ys, xs)
        if lab is None:
            continue
        idx = m.nonzero(as_tuple=False).squeeze(-1)
        a = x[idx[lab == 0]].mean(0)
        b = x[idx[lab == 1]].mean(0)
        vals.append(_cos(a, b))
    return float(np.mean(vals)) if vals else float("nan")


def other_obj_p_mass(x: torch.Tensor, gt: torch.Tensor, bg_id: int) -> float:
    """Dense ReLU-cosine P: share of mass from object patches onto other objects."""
    z = F.normalize(x.float(), dim=-1)
    w = (z @ z.T).clamp_min(0.0)
    w.fill_diagonal_(0.0)
    p = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
    ids = object_ids(gt, bg_id)
    if not ids:
        return float("nan")
    fracs = []
    for oid in ids:
        src = gt == oid
        if int(src.sum()) < 4:
            continue
        other = (gt != oid) & (gt != bg_id)
        if int(other.sum()) == 0:
            continue
        fracs.append(float(p[src][:, other].sum(-1).mean()))
    return float(np.mean(fracs)) if fracs else float("nan")


def score(x: torch.Tensor, gt: torch.Tensor, grid: int, bg_id: int) -> Dict[str, float]:
    return {
        "unique_keys": float(unique_keys(x, 0.95)),
        "std": token_std(x),
        "pair_cos": mean_offdiag_cos(x),
        "within_var": within_object_var(x, gt, bg_id),
        "cohesion": within_object_cohesion(x, gt, bg_id),
        "obj_mean_cos": object_mean_cos(x, gt, bg_id),
        "part_kmeans": part_cos_kmeans(x, gt, bg_id),
        "part_spatial": part_cos_spatial(x, gt, grid, bg_id),
        "p_other": other_obj_p_mass(x, gt, bg_id),
    }


def nanmean(xs: List[float]) -> float:
    v = [x for x in xs if x == x]
    return float(np.mean(v)) if v else float("nan")


@torch.no_grad()
def collect(name, cfg_path, min_obj, bg_id, data_dir, device, max_scan, n_vis):
    cfg = configuration.load_config(cfg_path)
    cfg.dataset.val_batch_size = 1
    cfg.dataset.num_val_workers = 0
    dm = data.build(cfg.dataset, data_dir=data_dir)
    dm.setup("validate")
    bb = cfg.model.encoder.backbone
    kwargs = dict(bb.get("model_kwargs") or {})
    backbone = TimmExtractor(
        model=bb.model,
        pretrained=True,
        frozen=True,
        features=bb.features,
        model_kwargs=kwargs or None,
    ).to(device).eval()
    glob = NcutRelationalLeveling(chunk_size=4, n_iter=16, barrier=False).to(device)
    n8 = NcutRelationalLeveling(chunk_size=8, barrier=False, n8=True).to(device)
    id_fn = patch_ids_ytvis if name == "ytvis" else patch_ids_movi
    rows: List[dict] = []
    vis: List[dict] = []
    n_clips = 0
    print(f"\n=== {name}  scan≤{max_scan}  min_obj={min_obj} ===", flush=True)
    for batch in dm.val_dataloader():
        if n_clips >= max_scan:
            break
        if "batch_padding_mask" in batch and torch.is_tensor(batch["batch_padding_mask"]) and bool(
            batch["batch_padding_mask"].any()
        ):
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
        global_x = glob(raw.unsqueeze(0), 0.0)[0]
        n8_x = n8(raw.unsqueeze(0), 0.0)[0]
        feats = {"raw": raw, "global": global_x, "n8": n8_x}
        rec = {
            "dataset": name,
            "clip": n_clips - 1,
            "n_obj": n_obj,
            "grid": grid,
            "cos_raw_global": mean_token_cos(raw, global_x),
            "cos_raw_n8": mean_token_cos(raw, n8_x),
            "cos_global_n8": mean_token_cos(global_x, n8_x),
        }
        for m in METHODS:
            for k, v in score(feats[m], gt, grid, bg_id).items():
                rec[f"{m}_{k}"] = v
        rows.append(rec)
        print(
            f"  clip {rec['clip']:3d} n_obj={n_obj:2d}  "
            f"keys raw/global/n8={rec['raw_unique_keys']:.0f}/"
            f"{rec['global_unique_keys']:.0f}/{rec['n8_unique_keys']:.0f}  "
            f"obj-cos {rec['raw_obj_mean_cos']:.3f}/"
            f"{rec['global_obj_mean_cos']:.3f}/{rec['n8_obj_mean_cos']:.3f}  "
            f"Δtoken cos(raw,·) glob={rec['cos_raw_global']:.3f} n8={rec['cos_raw_n8']:.3f}  "
            f"glob vs n8={rec['cos_global_n8']:.3f}",
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
                    "global": global_x.cpu(),
                    "n8": n8_x.cpu(),
                    "gt": gt.cpu(),
                }
            )
    return rows, vis


def summarize(rows: List[dict]) -> Dict[str, float]:
    keys = [
        "cos_raw_global",
        "cos_raw_n8",
        "cos_global_n8",
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
        f"  token cosine to raw:   global {s['cos_raw_global']:.3f}   n8 {s['cos_raw_n8']:.3f}"
    )
    print(f"  token cosine global↔n8: {s['cos_global_n8']:.3f}")
    hdr = f"{'':16s} {'raw':>9s} {'global':>9s} {'n8':>9s}"
    print(hdr)
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
        print(
            f"  {label:16s} {s[f'raw_{key}']:9.3f} {s[f'global_{key}']:9.3f} {s[f'n8_{key}']:9.3f}"
        )
    print(
        "  read: higher obj-mean cos = instances look more alike (merge risk). "
        "higher part cos = parts collapsed. lower unique keys = coarser Keys."
    )


def draw(name: str, frames: List[dict], out_path: str) -> None:
    if not frames:
        return
    n_row = len(frames)
    fig, axes = plt.subplots(n_row, 4, figsize=(11.2, 2.6 * n_row))
    if n_row == 1:
        axes = np.array([axes])
    titles = ["RGB", "raw DINO", "global ncut  s=0", "n8 FC  s=0"]
    for r, fr in enumerate(frames):
        rgb, h, w = fr["rgb"], *fr["rgb"].shape[:2]
        grid = fr["grid"]
        raw = fr["raw"]
        basis = pca_basis(raw)
        proj = (raw.float() - raw.mean(0)) @ basis
        lo, hi = proj.quantile(0.02, dim=0), proj.quantile(0.98, dim=0)
        imgs = [
            rgb if rgb.max() <= 1.5 else rgb / 255.0,
            upsample(pca_map(raw, basis, lo, hi, grid), h, w),
            upsample(pca_map(fr["global"], basis, lo, hi, grid), h, w),
            upsample(pca_map(fr["n8"], basis, lo, hi, grid), h, w),
        ]
        for c, im in enumerate(imgs):
            axes[r, c].imshow(np.clip(im, 0, 1))
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10)
        axes[r, 0].set_ylabel(f"clip {fr['clip']}\n{fr['n_obj']} obj", fontsize=8)
    fig.suptitle(
        f"{name}: PCA fit on raw. Global ncut mixes same-appearance patches anywhere; "
        "n8 only 8-neighbors. mix=0 (fully leveled).",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("wrote", out_path, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/workspace/dataset")
    p.add_argument("--out-dir", default="event_analysis/featcur_n8_vs_global")
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
        draw(name, vis, os.path.join(args.out_dir, f"{name}_pca_raw_global_n8.png"))
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(all_summ, f, indent=2)
    print("wrote", os.path.join(args.out_dir, "summary.json"), flush=True)


if __name__ == "__main__":
    main()
