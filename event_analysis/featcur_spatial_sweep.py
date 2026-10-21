"""How wide a self-excluded filter must be before frame-to-frame cosine matches dense P.

Same pairs and cosine as featcur_spatial_probe.py. One backbone pass, many mixes.
Gaussian / bilateral use every other patch (no radius cutoff). Diagonal is 0.

  gauss_s{σ}              spatial Gaussian
  pool_c{k} / pool_all    leave-one-out block mean
  bila_s{σ}_t{τ}          spatial Gaussian times exp((cos-1)/τ)

Usage (inside the slotcurri image):
  python event_analysis/featcur_spatial_sweep.py --dataset ytvis
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_spatial_probe import (  # noqa: E402
    CONFIGS,
    _empty_stats,
    _grid,
    _update,
    configuration,
    encode,
    encode_batch_size,
    load_pairs,
    pool_exclude_self,
    summarize,
    unit_tokens,
)
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor  # noqa: E402

SIGMAS = (2, 4, 8, 16, 32)
CELLS = (4, 8, 16)
TAUS = (0.1, 0.5)
_SPATIAL = {}


def spatial_weight(grid: int, sigma: float, device, dtype) -> torch.Tensor:
    key = (grid, float(sigma), device.type, device.index, str(dtype))
    cached = _SPATIAL.get(key)
    if cached is not None:
        return cached
    axis = torch.arange(grid, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coord = torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=1)
    delta = coord[:, None, :] - coord[None, :, :]
    dist2 = (delta * delta).sum(-1)
    weight = torch.exp(-dist2 / (2.0 * float(sigma) * float(sigma)))
    weight.fill_diagonal_(0)
    _SPATIAL[key] = weight
    return weight


def mix_rows(weight: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    if weight.ndim == 2:
        mixed = torch.einsum("ij,bjd->bid", weight, tokens)
        row = weight.sum(-1)
        keep = row < 1e-8
        mixed = mixed / row.clamp_min(1e-8).view(1, -1, 1)
        return torch.where(keep.view(1, -1, 1), tokens, mixed)
    mixed = torch.bmm(weight, tokens)
    row = weight.sum(-1)
    keep = row < 1e-8
    mixed = mixed / row.clamp_min(1e-8).unsqueeze(-1)
    return torch.where(keep.unsqueeze(-1), tokens, mixed)


def variant_names(grid: int):
    names = ["raw", "dense"]
    names += [f"gauss_s{sigma:g}" for sigma in SIGMAS]
    names += [f"pool_c{cell}" for cell in CELLS]
    names.append("pool_all")
    for tau in TAUS:
        for sigma in SIGMAS:
            names.append(f"bila_s{sigma:g}_t{tau:g}")
    return names


def apply_variants(tokens, dense):
    grid = _grid(tokens.shape[1])
    out = {"raw": tokens, "dense": dense(tokens, 0.0)}
    for sigma in SIGMAS:
        weight = spatial_weight(grid, sigma, tokens.device, tokens.dtype)
        out[f"gauss_s{sigma:g}"] = mix_rows(weight, tokens)
    for cell in CELLS:
        out[f"pool_c{cell}"] = pool_exclude_self(tokens, cell=cell)
    out["pool_all"] = pool_exclude_self(tokens, cell=grid)
    feat_n = F.normalize(tokens, dim=-1)
    cosine = torch.bmm(feat_n, feat_n.transpose(1, 2))
    for tau in TAUS:
        affinity = torch.exp((cosine - 1.0) / tau)
        for sigma in SIGMAS:
            weight = spatial_weight(grid, sigma, tokens.device, tokens.dtype)
            out[f"bila_s{sigma:g}_t{tau:g}"] = mix_rows(weight.unsqueeze(0) * affinity, tokens)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(CONFIGS))
    parser.add_argument("--data-dir", default="/workspace/dataset")
    parser.add_argument("--out-dir", default="event_analysis/featcur_spatial")
    parser.add_argument("--n-videos", type=int, default=128)
    parser.add_argument("--per-shard", type=int, default=6)
    parser.add_argument("--batch-videos", type=int, default=8)
    args = parser.parse_args()

    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    cfg = configuration.load_config(CONFIGS[args.dataset])
    backbone_cfg = cfg.model.encoder.backbone
    input_size = int(cfg.dataset.val_pipeline.transforms.input_size)
    frame_batch = encode_batch_size(backbone_cfg.model, input_size)
    backbone = TimmExtractor(
        model=backbone_cfg.model,
        pretrained=True,
        frozen=True,
        features=backbone_cfg.features,
        model_kwargs=dict(backbone_cfg.get("model_kwargs") or {}) or None,
    ).to(device).eval()
    dense = NcutRelationalLeveling(chunk_size=4, barrier=False).to(device)

    pairs = load_pairs(cfg, args.data_dir, args.n_videos, args.per_shard, rng)
    print(f"{args.dataset}: {len(pairs)} consecutive pairs", flush=True)
    names = None
    stats = None
    tok0 = None
    with torch.no_grad():
        for start in range(0, len(pairs), args.batch_videos):
            batch = pairs[start : start + args.batch_videos]
            if len(batch) < 2:
                continue
            clip = torch.stack(batch, 0).to(device)
            tok0 = encode(backbone, clip[:, 0], backbone_cfg.features, frame_batch)
            tok1 = encode(backbone, clip[:, 1], backbone_cfg.features, frame_batch)
            filt0 = apply_variants(tok0, dense)
            filt1 = apply_variants(tok1, dense)
            if names is None:
                names = variant_names(_grid(tok0.shape[1]))
                stats = {name: _empty_stats() for name in names}
            raw0 = unit_tokens(filt0["raw"])
            raw1 = unit_tokens(filt1["raw"])
            for name in names:
                _update(
                    stats,
                    name,
                    raw0,
                    unit_tokens(filt0[name]),
                    raw1,
                    unit_tokens(filt1[name]),
                    cross_frame=True,
                )
            print(f"  pairs {start + len(batch)}/{len(pairs)}", flush=True)

    rows = summarize(stats)
    grid = _grid(tok0.shape[1])
    dense_t = rows["dense"]["temporal_patch"]
    print(
        f"\n{args.dataset} grid={grid} videos={stats['raw']['videos']}  "
        f"dense t_patch={dense_t:.3f}",
        flush=True,
    )
    print(
        f"{'variant':16s} {'within':8s} {'cross':8s} {'t_patch':8s} "
        f"{'gap':8s} {'vs_raw':8s}",
        flush=True,
    )
    for name in names:
        row = rows[name]
        gap = row["temporal_patch"] - dense_t
        print(
            f"{name:16s} {row['within']:8.3f} {row['cross']:8.3f} "
            f"{row['temporal_patch']:8.3f} {gap:8.3f} {row['vs_raw']:8.3f}",
            flush=True,
        )
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.dataset}_sweep.json")
    with open(path, "w") as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "n_videos": int(stats["raw"]["videos"]),
                "grid": grid,
                "sigmas": list(SIGMAS),
                "cells": list(CELLS),
                "taus": list(TAUS),
                "exclude_self": True,
                "variants": rows,
            },
            handle,
            indent=2,
        )
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
