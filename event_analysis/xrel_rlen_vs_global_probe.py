"""Compare v39lam1gu global X^rel with residual correlation-length X^rel.

Old:
    Z = X/||X||, W = ReLU(Z Z^T), W_ii = 0, X^rel = row-normalize(W) X

New:
    r = X - mean(X)
    rho(d) = sum_{cheb=d} r_i·r_j / sum ||r_i|| ||r_j||
    ell = first d with rho(d) <= 0, else G/2
    edge if 0 < cheb <= ell and r_i·r_j > 0
    X^rel = mean of X inside each connected component (singletons stay)

Usage (slotcurri image):
  python event_analysis/xrel_rlen_vs_global_probe.py --data-dir /workspace/dataset
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_strength_probe import extract_tokens, patch_ids_movi, patch_ids_ytvis  # noqa: E402
from featcur_xrel_probe import (  # noqa: E402
    DATASETS,
    cohesion,
    mean_offdiag_cos,
    mean_token_cos,
    object_ids,
    object_mean_stats,
    participation_ratio,
    part_spatial,
    unique_keys,
)
from slotcurri import configuration, data
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor


def _rho_and_ell(gram: torch.Tensor, nrm: torch.Tensor, cheb: torch.Tensor, grid: int):
    denom_w = nrm[:, None] * nrm[None, :]
    rhos = []
    ell = grid // 2
    found = False
    for dist in range(1, grid // 2 + 1):
        m = cheb == dist
        den = denom_w[m].sum().clamp_min(1e-12)
        rho = float((gram[m].sum() / den).item())
        rhos.append(rho)
        if not found and rho <= 0.0:
            ell = dist
            found = True
            break
    return ell, rhos


def xrel_rlen(x: torch.Tensor):
    """Component-mean leveling. x: (N, D) -> rel, ell, sizes, rho0."""
    n = x.shape[0]
    grid = int(math.sqrt(n))
    r = x - x.mean(0, keepdim=True)
    nrm = r.norm(dim=-1).clamp_min(1e-12)
    gram = r @ r.T
    idx = torch.arange(n, device=x.device)
    ys, xs = idx // grid, idx % grid
    cheb = torch.maximum((ys[:, None] - ys[None, :]).abs(), (xs[:, None] - xs[None, :]).abs())
    ell, rhos = _rho_and_ell(gram, nrm, cheb, grid)
    edge = (gram > 0) & (cheb > 0) & (cheb <= ell)
    ii, jj = torch.nonzero(torch.triu(edge, diagonal=1), as_tuple=True)
    parent = np.arange(n, dtype=np.int32)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(ii.tolist(), jj.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = [find(i) for i in range(n)]
    groups = defaultdict(list)
    for i, rt in enumerate(roots):
        groups[rt].append(i)
    out = x.clone()
    sizes = []
    labels = torch.empty(n, dtype=torch.long, device=x.device)
    for lab, members in enumerate(groups.values()):
        sizes.append(len(members))
        sel = torch.tensor(members, device=x.device, dtype=torch.long)
        labels[sel] = lab
        if len(members) == 1:
            continue
        out[sel] = x[sel].mean(0)
    return out, ell, sizes, rhos[0] if rhos else float("nan"), labels


def pack_stats(x: torch.Tensor, gt: torch.Tensor, grid: int, bg_id: int) -> dict:
    ids = object_ids(gt, bg_id, min_n=4)
    obj_cos, merge = object_mean_stats(x, gt, ids)
    return {
        "keys": float(unique_keys(x)),
        "pr": participation_ratio(x),
        "pair_cos": mean_offdiag_cos(x),
        "obj_cos": obj_cos,
        "merge": merge,
        "cohesion": cohesion(x, gt, ids),
        "part": part_spatial(x, gt, grid, ids),
    }


def residual_cos(a: torch.Tensor, b: torch.Tensor, mu: torch.Tensor) -> float:
    return mean_token_cos(a - mu, b - mu)


def region_cos(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum()) < 4:
        return float("nan")
    return mean_token_cos(a[mask], b[mask])


def component_gt(labels: torch.Tensor, gt: torch.Tensor, bg_id: int) -> dict:
    """How components sit on GT. labels is the connected-component id."""
    ids = object_ids(gt, bg_id, min_n=4)
    frag = []
    pure = []
    for oid in ids:
        m = gt == oid
        if int(m.sum()) < 4:
            continue
        frag.append(float(labels[m].unique().numel()))
    for lab in labels.unique().tolist():
        m = labels == lab
        if int(m.sum()) < 4:
            continue
        vals, counts = gt[m].unique(return_counts=True)
        pure.append(float(counts.max().item()) / float(m.sum().item()))
    same = []
    for a_i, oid_a in enumerate(ids):
        for oid_b in ids[a_i + 1 :]:
            la = labels[gt == oid_a]
            lb = labels[gt == oid_b]
            if la.numel() == 0 or lb.numel() == 0:
                continue
            # share a component if any label overlaps
            same.append(float(len(set(la.tolist()) & set(lb.tolist())) > 0))
    return {
        "frag": float(np.mean(frag)) if frag else float("nan"),
        "purity": float(np.mean(pure)) if pure else float("nan"),
        "obj_linked": float(np.mean(same)) if same else float("nan"),
    }


def summarize(rows):
    keys = [k for k in rows[0].keys() if k not in ("dataset",)]
    out = {"n": len(rows)}
    for k in keys:
        vals = [float(r[k]) for r in rows if r[k] == r[k]]
        if not vals:
            out[k] = None
            continue
        a = np.asarray(vals, dtype=np.float64)
        out[k] = {
            "median": float(np.median(a)),
            "mean": float(a.mean()),
            "p25": float(np.quantile(a, 0.25)),
            "p75": float(np.quantile(a, 0.75)),
        }
    return out


@torch.no_grad()
def collect(name, cfg_path, min_obj, bg_id, data_dir, device, max_scan, max_frames):
    cfg = configuration.load_config(cfg_path)
    cfg.dataset.val_batch_size = 1
    cfg.dataset.num_val_workers = 0
    dm = data.build(cfg.dataset, data_dir=data_dir)
    dm.setup("validate")
    bb = cfg.model.encoder.backbone
    backbone = (
        TimmExtractor(
            model=bb.model,
            pretrained=True,
            frozen=True,
            features=bb.features,
            model_kwargs=dict(bb.get("model_kwargs") or {}) or None,
        )
        .to(device)
        .eval()
    )
    level = NcutRelationalLeveling(chunk_size=1, n_iter=16, barrier=False).to(device)
    id_fn = patch_ids_ytvis if name == "ytvis" else patch_ids_movi
    rows = []
    n_clips = 0
    for batch in dm.val_dataloader():
        if len(rows) >= max_frames or n_clips >= max_scan:
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
        n_obj = len(object_ids(gt, bg_id, min_n=1))
        n_clips += 1
        if n_obj < min_obj:
            continue
        old = level(raw.unsqueeze(0), 0.0)[0]
        new, ell, sizes, rho1, labels = xrel_rlen(raw)
        mu = raw.mean(0, keepdim=True)
        fg = gt != bg_id
        st_old = pack_stats(old, gt, grid, bg_id)
        st_new = pack_stats(new, gt, grid, bg_id)
        gt_new = component_gt(labels, gt, bg_id)
        sizes_a = np.asarray(sizes, dtype=np.float64)
        multi = sizes_a[sizes_a >= 2]
        rec = {
            "dataset": name,
            "n_obj": float(n_obj),
            "ell": float(ell),
            "ell_frac": float(ell) / float(grid),
            "rho1": float(rho1),
            "n_comp": float(len(sizes)),
            "n_comp_multi": float(multi.size),
            "singleton_frac": float(np.mean(sizes_a == 1)),
            "multi_size_med": float(np.median(multi)) if multi.size else 0.0,
            "cos_old_new": mean_token_cos(old, new),
            "cos_old_new_fg": region_cos(old, new, fg),
            "cos_old_new_bg": region_cos(old, new, ~fg),
            "cos_old_new_res": residual_cos(old, new, mu),
            "cos_x_old": mean_token_cos(raw, old),
            "cos_x_new": mean_token_cos(raw, new),
            "cos_x_old_fg": region_cos(raw, old, fg),
            "cos_x_new_fg": region_cos(raw, new, fg),
            "cos_x_old_bg": region_cos(raw, old, ~fg),
            "cos_x_new_bg": region_cos(raw, new, ~fg),
            "frag": gt_new["frag"],
            "purity": gt_new["purity"],
            "obj_linked": gt_new["obj_linked"],
        }
        for prefix, st in (("old", st_old), ("new", st_new)):
            for k, v in st.items():
                rec[f"{prefix}_{k}"] = v
        rows.append(rec)
        print(
            f"  {name} n_obj={n_obj:2d} ell={ell:2d}/{grid}  "
            f"cos(old,new)={rec['cos_old_new']:.3f} res={rec['cos_old_new_res']:.3f}  "
            f"keys {st_old['keys']:.0f}->{st_new['keys']:.0f}  "
            f"obj-cos {st_old['obj_cos']:.3f}->{st_new['obj_cos']:.3f}  "
            f"part {st_old['part']:.3f}->{st_new['part']:.3f}  "
            f"linked={rec['obj_linked']:.2f} frag={rec['frag']:.1f} pure={rec['purity']:.2f}",
            flush=True,
        )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/workspace/dataset")
    p.add_argument("--max-scan", type=int, default=80)
    p.add_argument("--max-frames", type=int, default=12)
    p.add_argument("--out", default="event_analysis/featcur_xrel/rlen_vs_global.json")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = {}
    for name, (cfg, min_obj, bg_id, _label) in DATASETS.items():
        print(f"\n=== {name} ===", flush=True)
        rows = collect(name, cfg, min_obj, bg_id, args.data_dir, device, args.max_scan, args.max_frames)
        summary[name] = summarize(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
