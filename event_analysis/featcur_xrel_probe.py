"""X vs X^rel for the v39lam1gu feature curriculum on three val sets.

Operator is the training one at mix=0 (fully leveled Keys):

    Z = X / ||X||
    W_ij = ReLU(z_i^T z_j),  W_ii = 0
    X^rel = P X,  P = row-normalize(W)

No Fiedler cut. Each dataset uses its own v39lam1gu backbone and val resolution:
YouTube-VIS ViT-B/14 @ 518 (37x37), MOVi-C ViT-S/14 @ 336 (24x24),
MOVi-E ViT-B/14 @ 336 (24x24).

Usage (inside the slotcurri image):
  python event_analysis/featcur_xrel_probe.py --data-dir /workspace/dataset
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
    "ytvis": (
        "configs/slotcurri/ytvis2021_attnmass_v39lam1gu.yaml",
        2,
        -1,
        "YouTube-VIS",
    ),
    "movi_c": (
        "configs/slotcurri/movi_c_attnmass_v39lam1gu.yaml",
        3,
        0,
        "MOVi-C",
    ),
    "movi_e": (
        "configs/slotcurri/movi_e_attnmass_v39lam1gu.yaml",
        3,
        0,
        "MOVi-E",
    ),
}


def unique_keys(tokens: torch.Tensor, cos_thr: float = 0.95) -> int:
    """Greedy count. A later token is a new key only if cos < thr to every kept key."""
    xn = F.normalize(tokens.float(), dim=-1)
    sim = xn @ xn.T
    n = xn.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=xn.device)
    n_keys = 0
    while bool(alive.any()):
        i = int(alive.nonzero(as_tuple=False)[0])
        n_keys += 1
        alive = alive & (sim[i] < cos_thr)
    return n_keys


def object_ids(gt: torch.Tensor, bg_id: int, min_n: int = 1) -> List[int]:
    out = []
    for i in gt.unique().tolist():
        oid = int(i)
        if oid == bg_id:
            continue
        if int((gt == oid).sum()) >= min_n:
            out.append(oid)
    return out


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.float(), dim=0)
    b = F.normalize(b.float(), dim=0)
    return float((a * b).sum().clamp(-1.0, 1.0).item())


def token_std(x: torch.Tensor) -> float:
    return float(x.float().std(unbiased=False).item())


def mean_offdiag_cos(x: torch.Tensor) -> float:
    xn = F.normalize(x.float(), dim=-1)
    n = xn.shape[0]
    s = xn @ xn.T
    return float((s.sum() - n) / (n * (n - 1)))


def mean_token_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    an = F.normalize(a.float(), dim=-1)
    bn = F.normalize(b.float(), dim=-1)
    return float((an * bn).sum(-1).mean().item())


def participation_ratio(x: torch.Tensor) -> float:
    xc = x.float() - x.float().mean(0, keepdim=True)
    sv = torch.linalg.svdvals(xc)
    energy = sv.square()
    tot = energy.sum().clamp_min(1e-12)
    return float((tot.square() / energy.square().sum().clamp_min(1e-12)).item())


def within_var(x: torch.Tensor, gt: torch.Tensor, ids: List[int]) -> float:
    vals = []
    for oid in ids:
        m = gt == oid
        if int(m.sum()) < 8:
            continue
        vals.append(float(x[m].float().var(dim=0, unbiased=False).mean().item()))
    return float(np.mean(vals)) if vals else float("nan")


def cohesion(x: torch.Tensor, gt: torch.Tensor, ids: List[int]) -> float:
    vals = []
    xn = F.normalize(x.float(), dim=-1)
    for oid in ids:
        m = gt == oid
        if int(m.sum()) < 8:
            continue
        mu = F.normalize(xn[m].mean(0), dim=0)
        vals.append(float((xn[m] @ mu).mean().item()))
    return float(np.mean(vals)) if vals else float("nan")


def object_mean_stats(x: torch.Tensor, gt: torch.Tensor, ids: List[int], merge_thr: float = 0.90):
    means = []
    for oid in ids:
        m = gt == oid
        if int(m.sum()) < 4:
            continue
        means.append(F.normalize(x[m].float().mean(0), dim=0))
    if len(means) < 2:
        return float("nan"), float("nan")
    s = torch.stack(means) @ torch.stack(means).T
    k = s.shape[0]
    iu = torch.triu_indices(k, k, offset=1, device=s.device)
    pairs = s[iu[0], iu[1]]
    return float(pairs.mean().item()), float((pairs > merge_thr).float().mean().item())


def part_spatial(x: torch.Tensor, gt: torch.Tensor, grid: int, ids: List[int]) -> float:
    vals = []
    for oid in ids:
        m = gt == oid
        if int(m.sum()) < 16:
            continue
        flat = m.view(grid, grid)
        ys, xs = flat.nonzero(as_tuple=True)
        lab = spatial_halves(ys, xs)
        if lab is None:
            continue
        idx = m.nonzero(as_tuple=False).squeeze(-1)
        vals.append(_cos(x[idx[lab == 0]].mean(0), x[idx[lab == 1]].mean(0)))
    return float(np.mean(vals)) if vals else float("nan")


def part_kmeans_frozen(raw: torch.Tensor, other: torch.Tensor, gt: torch.Tensor, ids: List[int]) -> float:
    """2-means labels from raw tokens; cosine of those two group means in `other`."""
    vals = []
    xn = F.normalize(raw.float(), dim=-1)
    for oid in ids:
        idx = (gt == oid).nonzero(as_tuple=False).squeeze(-1)
        if int(idx.numel()) < 16:
            continue
        lab = kmeans2(xn[idx])
        if lab is None:
            continue
        vals.append(_cos(other[idx[lab == 0]].mean(0), other[idx[lab == 1]].mean(0)))
    return float(np.mean(vals)) if vals else float("nan")


def graph_stats(x: torch.Tensor, gt: torch.Tensor, grid: int, bg_id: int, ids: List[int]) -> Dict[str, float]:
    """P is the same row-normalized ReLU-cosine used to build X^rel."""
    z = F.normalize(x.float(), dim=-1)
    w = (z @ z.T).clamp_min(0.0)
    w.fill_diagonal_(0.0)
    row = w.sum(-1, keepdim=True).clamp_min(1e-6)
    p = w / row
    n = z.shape[0]
    ys = torch.arange(n, device=x.device) // grid
    xs = torch.arange(n, device=x.device) % grid
    cheb = torch.maximum((ys[:, None] - ys[None, :]).abs(), (xs[:, None] - xs[None, :]).abs())
    ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
    ent = ent / math.log(max(n, 2))
    pos = w > 0
    pos_cos = (z @ z.T).clamp_min(0.0)
    mean_pos = float(pos_cos[pos].mean().item()) if int(pos.sum()) else float("nan")

    self_m, other_m, bg_m = [], [], []
    for oid in ids:
        src = gt == oid
        if int(src.sum()) < 4:
            continue
        self_m.append(float(p[src][:, src].sum(-1).mean()))
        other = (gt != oid) & (gt != bg_id)
        bg = gt == bg_id
        other_m.append(float(p[src][:, other].sum(-1).mean()) if int(other.sum()) else 0.0)
        bg_m.append(float(p[src][:, bg].sum(-1).mean()) if int(bg.sum()) else 0.0)

    def _m(xs):
        return float(np.mean(xs)) if xs else float("nan")

    quarter = max(grid // 4, 2)
    return {
        "p_self": _m(self_m),
        "p_other": _m(other_m),
        "p_bg": _m(bg_m),
        "p_beyond_n8": float((p * (cheb > 1).float()).sum(-1).mean().item()),
        "p_beyond_quarter": float((p * (cheb > quarter).float()).sum(-1).mean().item()),
        "p_entropy": float(ent.mean().item()),
        "pos_degree_frac": float(pos.float().mean().item()),
        "mean_pos_cos": mean_pos,
        "quarter_radius": float(quarter),
    }


def score_frame(raw: torch.Tensor, rel: torch.Tensor, gt: torch.Tensor, grid: int, bg_id: int) -> dict:
    ids = object_ids(gt, bg_id, min_n=4)
    raw_var = within_var(raw, gt, ids)
    rel_var = within_var(rel, gt, ids)
    raw_obj_cos, raw_merge = object_mean_stats(raw, gt, ids)
    rel_obj_cos, rel_merge = object_mean_stats(rel, gt, ids)
    rec = {
        "cos_x_xrel": mean_token_cos(raw, rel),
        "bg_frac": float((gt == bg_id).float().mean().item()),
        "raw_unique_keys": float(unique_keys(raw)),
        "rel_unique_keys": float(unique_keys(rel)),
        "raw_std": token_std(raw),
        "rel_std": token_std(rel),
        "raw_pair_cos": mean_offdiag_cos(raw),
        "rel_pair_cos": mean_offdiag_cos(rel),
        "raw_within_var": raw_var,
        "rel_within_var": rel_var,
        "within_var_ratio": (rel_var / raw_var) if raw_var and raw_var == raw_var and raw_var > 0 else float("nan"),
        "raw_cohesion": cohesion(raw, gt, ids),
        "rel_cohesion": cohesion(rel, gt, ids),
        "raw_obj_mean_cos": raw_obj_cos,
        "rel_obj_mean_cos": rel_obj_cos,
        "raw_merge_frac": raw_merge,
        "rel_merge_frac": rel_merge,
        "raw_part_spatial": part_spatial(raw, gt, grid, ids),
        "rel_part_spatial": part_spatial(rel, gt, grid, ids),
        "raw_part_kmeans": part_kmeans_frozen(raw, raw, gt, ids),
        "rel_part_kmeans": part_kmeans_frozen(raw, rel, gt, ids),
        "raw_prank": participation_ratio(raw),
        "rel_prank": participation_ratio(rel),
    }
    rec.update(graph_stats(raw, gt, grid, bg_id, ids))
    return rec


def nanmean(xs: List[float]) -> float:
    v = [float(x) for x in xs if x == x]
    return float(np.mean(v)) if v else float("nan")


def nanmedian(xs: List[float]) -> float:
    v = [float(x) for x in xs if x == x]
    return float(np.median(v)) if v else float("nan")


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
    level = NcutRelationalLeveling(chunk_size=1, n_iter=16, barrier=False).to(device)
    id_fn = patch_ids_ytvis if name == "ytvis" else patch_ids_movi
    rows: List[dict] = []
    vis: List[dict] = []
    n_clips = 0
    meta = {
        "dataset": name,
        "label": DATASETS[name][3],
        "config": cfg_path,
        "backbone": str(bb.model),
        "features": str(bb.features),
        "input_size": int(cfg.dataset.val_pipeline.transforms.input_size),
        "anneal": str(cfg.model.feature_curriculum.anneal),
        "barrier": bool(cfg.model.feature_curriculum.barrier),
        "anneal_steps": int(cfg.model.feature_curriculum.anneal_steps),
        "max_steps": int(cfg.trainer.max_steps),
    }
    print(f"\n=== {name}  {meta['backbone']}  input {meta['input_size']} ===", flush=True)
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
        if grid * grid != raw.shape[0]:
            raise RuntimeError(f"{name}: non-square tokens {tuple(raw.shape)}")
        gt = id_fn(seg[ti].to(device), grid)
        n_obj = len(object_ids(gt, bg_id, min_n=1))
        n_clips += 1
        if n_obj < min_obj:
            print(f"  skip clip {n_clips - 1} n_obj={n_obj}", flush=True)
            continue
        rel = level(raw.unsqueeze(0), 0.0)[0]
        rec = {
            "dataset": name,
            "clip": n_clips - 1,
            "frame": ti,
            "n_obj": n_obj,
            "grid": grid,
            "feat_dim": int(raw.shape[-1]),
            "n_patches": int(raw.shape[0]),
        }
        rec.update(score_frame(raw, rel, gt, grid, bg_id))
        rows.append(rec)
        print(
            f"  clip {rec['clip']:3d} n_obj={n_obj:2d}  "
            f"keys {rec['raw_unique_keys']:.0f}->{rec['rel_unique_keys']:.0f}  "
            f"cos(X,Xrel)={rec['cos_x_xrel']:.3f}  "
            f"obj-cos {rec['raw_obj_mean_cos']:.3f}->{rec['rel_obj_mean_cos']:.3f}  "
            f"P self/other/bg={rec['p_self']:.2f}/{rec['p_other']:.2f}/{rec['p_bg']:.2f}  "
            f"beyond8={rec['p_beyond_n8']:.2f}",
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
                    "rel": rel.cpu(),
                    "gt": gt.cpu(),
                }
            )
    meta["grid"] = int(rows[0]["grid"]) if rows else None
    meta["feat_dim"] = int(rows[0]["feat_dim"]) if rows else None
    meta["n_patches"] = int(rows[0]["n_patches"]) if rows else None
    return rows, vis, meta


def summarize(rows: List[dict]) -> Dict[str, float]:
    keys = [
        "n_obj",
        "bg_frac",
        "cos_x_xrel",
        "raw_unique_keys",
        "rel_unique_keys",
        "raw_std",
        "rel_std",
        "raw_pair_cos",
        "rel_pair_cos",
        "raw_within_var",
        "rel_within_var",
        "within_var_ratio",
        "raw_cohesion",
        "rel_cohesion",
        "raw_obj_mean_cos",
        "rel_obj_mean_cos",
        "raw_merge_frac",
        "rel_merge_frac",
        "raw_part_spatial",
        "rel_part_spatial",
        "raw_part_kmeans",
        "rel_part_kmeans",
        "raw_prank",
        "rel_prank",
        "p_self",
        "p_other",
        "p_bg",
        "p_beyond_n8",
        "p_beyond_quarter",
        "p_entropy",
        "pos_degree_frac",
        "mean_pos_cos",
    ]
    out = {"n_frames": float(len(rows))}
    for k in keys:
        vals = [float(r[k]) for r in rows]
        out[k] = nanmean(vals)
        out[k + "_median"] = nanmedian(vals)
    return out


def draw(name: str, label: str, frames: List[dict], out_path: str) -> None:
    if not frames:
        return
    n_row = len(frames)
    fig, axes = plt.subplots(n_row, 4, figsize=(12.4, 2.7 * n_row))
    if n_row == 1:
        axes = np.array([axes])
    titles = ["RGB", "X  raw DINO", "X^rel = P X", "|X - X^rel|"]
    for r, fr in enumerate(frames):
        rgb = fr["rgb"]
        h, w = rgb.shape[:2]
        grid = fr["grid"]
        raw = fr["raw"]
        basis = pca_basis(raw)
        proj = (raw.float() - raw.mean(0)) @ basis
        lo, hi = proj.quantile(0.02, dim=0), proj.quantile(0.98, dim=0)
        delta = (fr["rel"].float() - raw.float()).norm(dim=-1)
        delta = delta / delta.max().clamp_min(1e-6)
        delta_rgb = plt.get_cmap("magma")(delta.view(grid, grid).numpy())[..., :3]
        imgs = [
            rgb if rgb.max() <= 1.5 else rgb / 255.0,
            upsample(pca_map(raw, basis, lo, hi, grid), h, w),
            upsample(pca_map(fr["rel"], basis, lo, hi, grid), h, w),
            upsample(delta_rgb, h, w),
        ]
        for c, im in enumerate(imgs):
            axes[r, c].imshow(np.clip(im, 0, 1))
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=11)
        axes[r, 0].set_ylabel(f"clip {fr['clip']}\n{fr['n_obj']} obj", fontsize=8)
    fig.suptitle(
        f"{label}: PCA fit on raw X, shared color scale. "
        "X^rel is fully leveled (s=0). Bright residual means the curriculum moved that patch.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("wrote", out_path, flush=True)


def draw_compare(summaries: Dict[str, dict], metas: Dict[str, dict], out_path: str) -> None:
    order = [k for k in ("ytvis", "movi_c", "movi_e") if k in summaries]
    labels = [metas[k]["label"] for k in order]
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.4))

    def bars(ax, pairs, title, ylabel):
        x = np.arange(len(order))
        width = 0.36
        for i, (key, lab) in enumerate(pairs):
            vals = [summaries[k][key] for k in order]
            ax.bar(x + (i - 0.5) * width, vals, width, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=8, frameon=False)

    bars(
        axes[0, 0],
        [("raw_unique_keys", "X"), ("rel_unique_keys", "X^rel")],
        "Distinct keys (cos < 0.95)",
        "mean count",
    )
    bars(
        axes[0, 1],
        [("raw_obj_mean_cos", "X"), ("rel_obj_mean_cos", "X^rel")],
        "Cosine between object means",
        "mean pairwise cosine",
    )
    bars(
        axes[0, 2],
        [("raw_part_spatial", "X"), ("rel_part_spatial", "X^rel")],
        "Spatial-half cosine inside an object",
        "mean cosine",
    )
    x = np.arange(len(order))
    width = 0.25
    for i, (key, lab) in enumerate(
        (("p_self", "same object"), ("p_other", "other object"), ("p_bg", "background"))
    ):
        axes[1, 0].bar(x + (i - 1) * width, [summaries[k][key] for k in order], width, label=lab)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Where P sends an object patch", fontsize=10)
    axes[1, 0].set_ylabel("mean affinity mass", fontsize=8)
    axes[1, 0].legend(fontsize=8, frameon=False)

    axes[1, 1].bar(x - 0.18, [summaries[k]["p_beyond_n8"] for k in order], 0.36, label="beyond 8-neigh")
    axes[1, 1].bar(
        x + 0.18,
        [summaries[k]["p_beyond_quarter"] for k in order],
        0.36,
        label="beyond grid/4",
    )
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Long-range mass in P", fontsize=10)
    axes[1, 1].set_ylabel("mean row mass", fontsize=8)
    axes[1, 1].legend(fontsize=8, frameon=False)

    axes[1, 2].bar(x, [summaries[k]["cos_x_xrel"] for k in order], 0.5, color="#4c78a8")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(labels)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title("Token cosine  X vs X^rel", fontsize=10)
    axes[1, 2].set_ylabel("mean cosine", fontsize=8)

    fig.suptitle("v39lam1gu feature curriculum at s=0. Same P, different DINO scenes.", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("wrote", out_path, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/workspace/dataset")
    p.add_argument("--out-dir", default="event_analysis/featcur_xrel")
    p.add_argument("--max-scan", type=int, default=36)
    p.add_argument("--n-vis", type=int, default=4)
    p.add_argument("--datasets", nargs="+", default=["ytvis", "movi_c", "movi_e"])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    all_summ = {}
    metas = {}
    for name in args.datasets:
        cfg, min_obj, bg_id, _label = DATASETS[name]
        rows, vis, meta = collect(
            name, cfg, min_obj, bg_id, args.data_dir, device, args.max_scan, args.n_vis
        )
        metas[name] = meta
        csv_path = os.path.join(args.out_dir, f"{name}_per_frame.csv")
        if rows:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print("wrote", csv_path, flush=True)
        summ = summarize(rows)
        all_summ[name] = {"meta": meta, "summary": summ}
        draw(name, meta["label"], vis, os.path.join(args.out_dir, f"{name}_x_xrel.png"))
    draw_compare(
        {k: v["summary"] for k, v in all_summ.items()},
        metas,
        os.path.join(args.out_dir, "compare_bars.png"),
    )
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(all_summ, f, indent=2)
    print("wrote", os.path.join(args.out_dir, "summary.json"), flush=True)


if __name__ == "__main__":
    main()
