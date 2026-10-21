"""Same-frame comparison of v36 2-way Ncut and the v39lam1gu feature curriculum.

Both operators share ReLU-cosine W with a zero diagonal. The difference is the
support of P:

    2-cut (barrier=True): Fiedler of the normalized affinity, median bipartition,
        then row-normalize W only inside each region.
    FC (barrier=False): row-normalize W on the full graph. This is the operator
        v39lam1gu trains at mix=0.

Usage (slotcurri image):
  python event_analysis/featcur_2cut_vs_fc_probe.py --data-dir /workspace/dataset
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_strength_probe import extract_tokens, patch_ids_movi, patch_ids_ytvis  # noqa: E402
from featcur_xrel_probe import (  # noqa: E402
    DATASETS,
    cohesion,
    mean_token_cos,
    object_ids,
    object_mean_stats,
    participation_ratio,
    unique_keys,
    within_var,
)
from slotcurri import configuration, data
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor


def row_p(w: torch.Tensor, region: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    """Row-normalize W. Optional region mask is the 2-cut barrier."""
    ww = w
    if region is not None:
        same = region.unsqueeze(0) == region.unsqueeze(1)
        ww = w.masked_fill(~same, 0.0)
    row = ww.sum(-1)
    p = ww / row.clamp_min(eps).unsqueeze(-1)
    isolated = row < eps
    if bool(isolated.any()):
        idx = isolated.nonzero(as_tuple=False).squeeze(-1)
        p = p.clone()
        p[idx] = 0.0
        p[idx, idx] = 1.0
    return p


def mass_split(p: torch.Tensor, gt: torch.Tensor, bg_id: int, ids: List[int]) -> Dict[str, float]:
    self_m, other_m, bg_m = [], [], []
    for oid in ids:
        src = gt == oid
        if int(src.sum()) < 4:
            continue
        other = (gt != oid) & (gt != bg_id)
        bg = gt == bg_id
        self_m.append(float(p[src][:, src].sum(-1).mean()))
        other_m.append(float(p[src][:, other].sum(-1).mean()) if int(other.sum()) else 0.0)
        bg_m.append(float(p[src][:, bg].sum(-1).mean()) if int(bg.sum()) else 0.0)

    def _m(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return {"p_self": _m(self_m), "p_other": _m(other_m), "p_bg": _m(bg_m)}


def cut_geometry(gt: torch.Tensor, region: torch.Tensor, bg_id: int) -> Dict[str, float]:
    """Median bipartition vs GT. region is bool, True = Fiedler >= median."""
    obj = gt != bg_id
    n_obj_patches = int(obj.sum())
    side1 = region.bool()
    side0 = ~side1
    ids = [int(i) for i in gt.unique().tolist() if int(i) != bg_id]
    maj = []
    n_cut = 0
    pures = []
    for oid in ids:
        m = gt == oid
        if int(m.sum()) == 0:
            continue
        frac = float(region[m].float().mean())
        pures.append(max(frac, 1.0 - frac))
        maj.append(1 if frac >= 0.5 else 0)
        if 0.15 < frac < 0.85:
            n_cut += 1
    n0 = sum(1 for x in maj if x == 0)
    n1 = sum(1 for x in maj if x == 1)
    n_obj = len(maj)
    one_side = float(n_obj > 0 and (n0 == 0 or n1 == 0))
    # Object side is the side holding more object patches. Median forces ~50% of
    # patches onto each side, so this side is objects plus whatever background
    # fills the quota.
    obj0 = int((obj & side0).sum())
    obj1 = int((obj & side1).sum())
    obj_side = side0 if obj0 >= obj1 else side1
    empty_side = ~obj_side
    obj_side_n = int(obj_side.sum())
    empty_n = int(empty_side.sum())
    obj_on_obj_side = int((obj & obj_side).sum())
    return {
        "n_obj_cut": float(n_obj),
        "n_cut": float(n_cut),
        "pure": float(np.mean(pures)) if pures else float("nan"),
        "all_one_side": one_side,
        "obj_patch_on_obj_side": (
            obj_on_obj_side / n_obj_patches if n_obj_patches else float("nan")
        ),
        "obj_side_bg_frac": (
            int((~obj & obj_side).sum()) / obj_side_n if obj_side_n else float("nan")
        ),
        "empty_side_obj_frac": (
            int((obj & empty_side).sum()) / empty_n if empty_n else float("nan")
        ),
        "side_balance": float(side1.float().mean().item()),
    }


def cross_cut_mass(p: torch.Tensor, region: torch.Tensor, gt: torch.Tensor, bg_id: int) -> float:
    """Mean FC mass that the 2-cut barrier would zero, on foreground tokens."""
    src = gt != bg_id
    if int(src.sum()) == 0:
        return float("nan")
    cross = region.unsqueeze(0) != region.unsqueeze(1)
    return float((p[src] * cross[src].float()).sum(-1).mean().item())


def token_pack(x: torch.Tensor, gt: torch.Tensor, bg_id: int, ids: List[int]) -> Dict[str, float]:
    obj_cos, merge = object_mean_stats(x, gt, ids)
    return {
        "keys": float(unique_keys(x)),
        "within_var": within_var(x, gt, ids),
        "obj_cos": obj_cos,
        "merge": merge,
        "cohesion": cohesion(x, gt, ids),
        "prank": participation_ratio(x),
    }


def nanmean(xs: List[float]) -> float:
    v = [float(x) for x in xs if x == x]
    return float(np.mean(v)) if v else float("nan")


def nanmedian(xs: List[float]) -> float:
    v = [float(x) for x in xs if x == x]
    return float(np.median(v)) if v else float("nan")


@torch.no_grad()
def collect(name, cfg_path, min_obj, bg_id, data_dir, device, max_scan, n_keep):
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
    two = NcutRelationalLeveling(chunk_size=1, n_iter=16, barrier=True).to(device)
    fc = NcutRelationalLeveling(chunk_size=1, n_iter=16, barrier=False).to(device)
    id_fn = patch_ids_ytvis if name == "ytvis" else patch_ids_movi
    rows: List[dict] = []
    n_clips = 0
    print(f"\n=== {name} ===", flush=True)
    for batch in dm.val_dataloader():
        if n_clips >= max_scan or len(rows) >= n_keep:
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
            print(f"  skip clip {n_clips - 1} n_obj={n_obj}", flush=True)
            continue
        exp = two.explain(raw.unsqueeze(0))
        rel_cut = exp["rel"][0]
        region = exp["region"][0]
        rel_fc = fc(raw.unsqueeze(0), 0.0)[0]
        ids = object_ids(gt, bg_id, min_n=4)
        z = F.normalize(raw.float(), dim=-1)
        w = (z @ z.T).clamp_min(0.0)
        w.fill_diagonal_(0.0)
        p_fc = row_p(w, None, 1e-6)
        p_cut = row_p(w, region, 1e-6)
        raw_pack = token_pack(raw, gt, bg_id, ids)
        cut_pack = token_pack(rel_cut, gt, bg_id, ids)
        fc_pack = token_pack(rel_fc, gt, bg_id, ids)
        geo = cut_geometry(gt, region, bg_id)
        rec = {
            "dataset": name,
            "clip": n_clips - 1,
            "n_obj": n_obj,
            "grid": grid,
            "bg_frac": float((gt == bg_id).float().mean().item()),
            "cos_raw_cut": mean_token_cos(raw, rel_cut),
            "cos_raw_fc": mean_token_cos(raw, rel_fc),
            "cos_cut_fc": mean_token_cos(rel_cut, rel_fc),
            "fc_cross_cut": cross_cut_mass(p_fc, region, gt, bg_id),
        }
        for prefix, pack in (("raw", raw_pack), ("cut", cut_pack), ("fc", fc_pack)):
            for k, v in pack.items():
                rec[f"{prefix}_{k}"] = v
        for prefix, p in (("cut", p_cut), ("fc", p_fc)):
            for k, v in mass_split(p, gt, bg_id, ids).items():
                rec[f"{prefix}_{k}"] = v
        rec.update(geo)
        rows.append(rec)
        print(
            f"  clip {rec['clip']:3d} n_obj={n_obj:2d}  "
            f"keys {rec['raw_keys']:.0f} / cut {rec['cut_keys']:.0f} / fc {rec['fc_keys']:.0f}  "
            f"merge cut {rec['cut_merge']:.2f} fc {rec['fc_merge']:.2f}  "
            f"one_side {rec['all_one_side']:.0f} obj_side_bg {rec['obj_side_bg_frac']:.2f}  "
            f"Pbg cut {rec['cut_p_bg']:.2f} fc {rec['fc_p_bg']:.2f}  "
            f"cross {rec['fc_cross_cut']:.2f}",
            flush=True,
        )
    del backbone
    torch.cuda.empty_cache()
    return rows


def summarize(rows: List[dict]) -> dict:
    keys = [k for k in rows[0].keys() if k not in ("dataset", "clip")]
    out = {"n_frames": float(len(rows))}
    for k in keys:
        xs = [float(r[k]) for r in rows]
        out[k] = nanmean(xs)
        out[k + "_median"] = nanmedian(xs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--max-scan", type=int, default=48)
    ap.add_argument("--n-keep", type=int, default=16)
    ap.add_argument("--datasets", default="ytvis,movi_c,movi_e")
    ap.add_argument("--out-dir", default="event_analysis/featcur_2cut_vs_fc")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}
    all_rows: List[dict] = []
    for name in args.datasets.split(","):
        cfg_path, min_obj, bg_id, _label = DATASETS[name]
        rows = collect(
            name, cfg_path, min_obj, bg_id, args.data_dir, device, args.max_scan, args.n_keep
        )
        summary[name] = summarize(rows)
        all_rows.extend(rows)
        print(json.dumps({name: summary[name]}, indent=2), flush=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if all_rows:
        csv_path = os.path.join(args.out_dir, "per_frame.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print("wrote", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
