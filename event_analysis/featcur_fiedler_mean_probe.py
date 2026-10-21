"""Median vs mean threshold on the same Fiedler vector.

v36 uses the median, so each side is half the patches. The Self-Guided MAE
paper thresholds the Fiedler at its mean. This keeps that eigenvector and only
changes the threshold, then row-normalizes W inside each side.

Usage (inside the slotcurri image):
  python event_analysis/featcur_fiedler_mean_probe.py --dataset movi_c
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_2cut_vs_fc_probe import cut_geometry, mass_split, row_p  # noqa: E402
from featcur_strength_probe import extract_tokens, patch_ids_movi, patch_ids_ytvis  # noqa: E402
from featcur_xrel_probe import DATASETS, object_ids  # noqa: E402
from slotcurri import configuration, data  # noqa: E402
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor  # noqa: E402


def side_means_cos(rel: torch.Tensor, region: torch.Tensor) -> float:
    a = rel[region]
    b = rel[~region]
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    return float(
        F.cosine_similarity(a.mean(0).float(), b.mean(0).float(), dim=0).item()
    )


def summarize(rows):
    keys = [k for k in rows[0] if k not in ("clip",)]
    out = {"n_frames": len(rows)}
    for key in keys:
        vals = [float(row[key]) for row in rows if row[key] == row[key]]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--data-dir", default="/workspace/dataset")
    parser.add_argument("--out-dir", default="event_analysis/featcur_fiedler_mean")
    parser.add_argument("--max-scan", type=int, default=48)
    parser.add_argument("--n-keep", type=int, default=16)
    args = parser.parse_args()

    cfg_path, min_obj, bg_id, _label = DATASETS[args.dataset]
    cfg = configuration.load_config(cfg_path)
    cfg.dataset.val_batch_size = 1
    cfg.dataset.num_val_workers = 0
    dm = data.build(cfg.dataset, data_dir=args.data_dir)
    dm.setup("validate")
    bb = cfg.model.encoder.backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = TimmExtractor(
        model=bb.model,
        pretrained=True,
        frozen=True,
        features=bb.features,
        model_kwargs=dict(bb.get("model_kwargs") or {}) or None,
    ).to(device).eval()
    two = NcutRelationalLeveling(chunk_size=1, n_iter=16, barrier=True).to(device)
    rw_cut = NcutRelationalLeveling(
        chunk_size=1, n_iter=16, barrier=True, fiedler_cut="rw_mean"
    ).to(device)
    id_fn = patch_ids_ytvis if args.dataset == "ytvis" else patch_ids_movi

    rows = []
    n_clips = 0
    print(f"=== {args.dataset} ===", flush=True)
    for batch in dm.val_dataloader():
        if n_clips >= args.max_scan or len(rows) >= args.n_keep:
            break
        if "batch_padding_mask" in batch and torch.is_tensor(batch["batch_padding_mask"]) and bool(
            batch["batch_padding_mask"].any()
        ):
            n_clips += 1
            continue
        video, seg = batch["video"][0], batch["segmentations"][0]
        frame_i = int(video.shape[0] // 2)
        frame = video[frame_i].to(device)
        raw = extract_tokens(backbone, frame.unsqueeze(0), bb.features).float()[0]
        grid = int(round(math.sqrt(raw.shape[0])))
        gt = id_fn(seg[frame_i].to(device), grid)
        n_obj = len(object_ids(gt, bg_id, min_n=1))
        n_clips += 1
        if n_obj < min_obj:
            continue
        exp = two.explain(raw.unsqueeze(0))
        v2 = exp["v2"][0]
        rw = rw_cut.explain(raw.unsqueeze(0))
        z = F.normalize(raw, dim=-1)
        weight = (z @ z.T).clamp_min(0)
        weight.fill_diagonal_(0)
        ids = object_ids(gt, bg_id, min_n=4)
        rec = {"clip": n_clips - 1, "n_obj": n_obj}
        regions = {
            "med": v2 >= v2.median(),
            "mean": v2 >= v2.mean(),
            "rw": rw["region"][0],
        }
        for name, region in regions.items():
            geo = cut_geometry(gt, region, bg_id)
            affinity = row_p(weight, region, 1e-6)
            mixed = affinity @ raw
            mass = mass_split(affinity, gt, bg_id, ids)
            rec[f"{name}_side"] = float(region.float().mean())
            rec[f"{name}_obj_side_bg"] = geo["obj_side_bg_frac"]
            rec[f"{name}_empty_obj"] = geo["empty_side_obj_frac"]
            rec[f"{name}_one_side"] = geo["all_one_side"]
            rec[f"{name}_pure"] = geo["pure"]
            rec[f"{name}_between"] = side_means_cos(mixed, region)
            rec[f"{name}_p_self"] = mass["p_self"]
            rec[f"{name}_p_other"] = mass["p_other"]
            rec[f"{name}_p_bg"] = mass["p_bg"]
        rec["rw_disagree"] = float((regions["med"] != regions["rw"]).float().mean())
        rows.append(rec)
        print(
            f"  clip {rec['clip']:3d} n_obj={n_obj:2d}  "
            f"side med/rw {rec['med_side']:.2f}/{rec['rw_side']:.2f}  "
            f"obj_bg {rec['med_obj_side_bg']:.2f}->{rec['rw_obj_side_bg']:.2f}  "
            f"Pbg {rec['med_p_bg']:.2f}->{rec['rw_p_bg']:.2f}  "
            f"between {rec['med_between']:.3f}->{rec['rw_between']:.3f}  "
            f"flip {rec['rw_disagree']:.2f}",
            flush=True,
        )

    summary = summarize(rows)
    print(f"\n{args.dataset} n={summary['n_frames']}", flush=True)
    print(
        f"{'thr':6s} {'side':7s} {'obj_bg':7s} {'empty_obj':9s} {'one_side':8s} "
        f"{'pure':7s} {'Pself':7s} {'Pother':7s} {'Pbg':7s} {'between':8s}",
        flush=True,
    )
    for name in ("med", "mean", "rw"):
        print(
            f"{name:6s} {summary[f'{name}_side']:7.3f} {summary[f'{name}_obj_side_bg']:7.3f} "
            f"{summary[f'{name}_empty_obj']:9.3f} {summary[f'{name}_one_side']:8.3f} "
            f"{summary[f'{name}_pure']:7.3f} {summary[f'{name}_p_self']:7.3f} "
            f"{summary[f'{name}_p_other']:7.3f} {summary[f'{name}_p_bg']:7.3f} "
            f"{summary[f'{name}_between']:8.3f}",
            flush=True,
        )
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.dataset}.json")
    with open(path, "w") as handle:
        json.dump({"dataset": args.dataset, "summary": summary}, handle, indent=2)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
