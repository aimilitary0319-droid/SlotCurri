"""Within-group and between-group cosine of the two Ncut partitions.

z = X / ||X||. Within a group, self-pairs are excluded. Between groups is the
mean of every cross pair, which equals the dot product of the two group means.
rw is y = D^{-1/2} v cut at mean(y). med is v cut at its median.

Usage (inside the slotcurri image):
  python event_analysis/featcur_group_cos.py --dataset ytvis
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_strength_probe import extract_tokens, patch_ids_movi, patch_ids_ytvis  # noqa: E402
from featcur_xrel_probe import DATASETS, object_ids  # noqa: E402
from slotcurri import configuration, data  # noqa: E402
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor  # noqa: E402


def within(z: torch.Tensor, mask: torch.Tensor) -> float:
    members = z[mask]
    count = members.shape[0]
    if count < 2:
        return float("nan")
    mean = members.mean(0)
    return (count * float(mean.square().sum()) - 1.0) / (count - 1)


def between(z: torch.Tensor, region: torch.Tensor) -> float:
    left = z[region]
    right = z[~region]
    if left.numel() == 0 or right.numel() == 0:
        return float("nan")
    return float((left.mean(0) * right.mean(0)).sum().item())


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


def group_stats(z: torch.Tensor, region: torch.Tensor) -> dict:
    n0 = int(region.sum())
    n1 = int((~region).sum())
    w0 = within(z, region)
    w1 = within(z, ~region)
    if n0 >= 2 and n1 >= 2:
        inside = (w0 * n0 + w1 * n1) / (n0 + n1)
    else:
        inside = w0 if n0 >= 2 else w1
    return {
        "within": inside,
        "within0": w0,
        "within1": w1,
        "between": between(z, region),
        "side": n0 / (n0 + n1),
    }


def mean_of(rows, key):
    vals = [row[key] for row in rows if row[key] == row[key]]
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--data-dir", default="/workspace/dataset")
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
    med_cut = NcutRelationalLeveling(chunk_size=1, n_iter=16, barrier=True).to(device)
    rw_cut = NcutRelationalLeveling(
        chunk_size=1, n_iter=16, barrier=True, fiedler_cut="rw_mean"
    ).to(device)
    id_fn = patch_ids_ytvis if args.dataset == "ytvis" else patch_ids_movi

    rows = []
    n_clips = 0
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
        n_clips += 1
        if len(object_ids(gt, bg_id, min_n=1)) < min_obj:
            continue
        med = med_cut.explain(raw.unsqueeze(0))["region"][0]
        rw = rw_cut.explain(raw.unsqueeze(0))["region"][0]
        rec = {}
        for name, region in (("med", med), ("rw", rw)):
            mixed = F.normalize(within_group_px(raw, region), dim=-1)
            stats = group_stats(mixed, region)
            for key, value in stats.items():
                rec[f"{name}_{key}"] = value
        rows.append(rec)

    print(
        f"\n{args.dataset} n={len(rows)}  "
        "PX inside each group, then L2 cosine. Self excluded inside a group",
        flush=True,
    )
    print(
        f"{'cut':6s} {'within':8s} {'between':8s} {'gap':8s} {'side':7s}",
        flush=True,
    )
    for name in ("med", "rw"):
        inside = mean_of(rows, f"{name}_within")
        across = mean_of(rows, f"{name}_between")
        side = mean_of(rows, f"{name}_side")
        print(
            f"{name:6s} {inside:8.3f} {across:8.3f} {inside - across:8.3f} {side:7.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
