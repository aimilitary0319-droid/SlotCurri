"""raw / dense P / 8-neighbor P, self excluded on both graphs.

n8 is NcutRelationalLeveling(n8=True): R_ij = ReLU(z_i^T z_j) for Chebyshev-1
neighbors only, diagonal absent, then row-normalize. A zero row keeps X.

Usage (inside the slotcurri image):
  python event_analysis/featcur_n8_probe.py --dataset movi_e
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

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
    summarize,
    unit_tokens,
)
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor  # noqa: E402

NAMES = ("raw", "dense", "n8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(CONFIGS))
    parser.add_argument("--data-dir", default="/workspace/dataset")
    parser.add_argument("--out-dir", default="event_analysis/featcur_spatial")
    parser.add_argument("--n-videos", type=int, default=128)
    parser.add_argument("--per-shard", type=int, default=6)
    parser.add_argument("--batch-videos", type=int, default=8)
    args = parser.parse_args()

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
    n8 = NcutRelationalLeveling(chunk_size=8, barrier=False, n8=True).to(device)

    pairs = load_pairs(cfg, args.data_dir, args.n_videos, args.per_shard, rng)
    print(f"{args.dataset}: {len(pairs)} consecutive pairs", flush=True)
    stats = {name: _empty_stats() for name in NAMES}
    tok0 = None
    with torch.no_grad():
        for start in range(0, len(pairs), args.batch_videos):
            batch = pairs[start : start + args.batch_videos]
            if len(batch) < 2:
                continue
            clip = torch.stack(batch, 0).to(device)
            tok0 = encode(backbone, clip[:, 0], backbone_cfg.features, frame_batch)
            tok1 = encode(backbone, clip[:, 1], backbone_cfg.features, frame_batch)
            filt0 = {"raw": tok0, "dense": dense(tok0, 0.0), "n8": n8(tok0, 0.0)}
            filt1 = {"raw": tok1, "dense": dense(tok1, 0.0), "n8": n8(tok1, 0.0)}
            raw0 = unit_tokens(filt0["raw"])
            raw1 = unit_tokens(filt1["raw"])
            for name in NAMES:
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
    print(
        f"\n{args.dataset} grid={grid} videos={stats['raw']['videos']}",
        flush=True,
    )
    print(
        f"{'variant':8s} {'within':8s} {'cross':8s} {'ratio':7s} "
        f"{'t_patch':8s} {'vs_raw':8s}",
        flush=True,
    )
    for name in NAMES:
        row = rows[name]
        print(
            f"{name:8s} {row['within']:8.3f} {row['cross']:8.3f} {row['ratio']:7.3f} "
            f"{row['temporal_patch']:8.3f} {row['vs_raw']:8.3f}",
            flush=True,
        )
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.dataset}_n8.json")
    with open(path, "w") as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "n_videos": int(stats["raw"]["videos"]),
                "grid": grid,
                "exclude_self": True,
                "variants": rows,
            },
            handle,
            indent=2,
        )
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
