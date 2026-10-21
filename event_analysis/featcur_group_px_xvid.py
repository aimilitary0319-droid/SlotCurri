"""Cross-video cosine after PX is restricted to one Ncut group.

Same 128 training videos and LN+L2 statistic as featcur_spatial_probe.py.
  dense  row-normalize ReLU-cosine over the whole frame
  med    same, but only inside the median cut of v
  rw     same, but only inside y = D^{-1/2} v cut at mean(y)

Usage (inside the slotcurri image):
  python event_analysis/featcur_group_px_xvid.py --dataset ytvis
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_spatial_probe import (  # noqa: E402
    CONFIGS,
    _empty_stats,
    _update,
    configuration,
    encode,
    encode_batch_size,
    load_pairs,
    summarize,
    unit_tokens,
)
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor  # noqa: E402

NAMES = ("raw", "dense", "med", "rw")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(CONFIGS))
    parser.add_argument("--data-dir", default="/workspace/dataset")
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
    dense = NcutRelationalLeveling(chunk_size=4, n_iter=16, barrier=False).to(device)
    med = NcutRelationalLeveling(
        chunk_size=4, n_iter=16, barrier=True, fiedler_cut="sym_median"
    ).to(device)
    rw = NcutRelationalLeveling(
        chunk_size=4, n_iter=16, barrier=True, fiedler_cut="rw_mean"
    ).to(device)

    pairs = load_pairs(cfg, args.data_dir, args.n_videos, args.per_shard, rng)
    print(f"{args.dataset}: {len(pairs)} pairs", flush=True)
    stats = {name: _empty_stats() for name in NAMES}
    with torch.no_grad():
        for start in range(0, len(pairs), args.batch_videos):
            batch = pairs[start : start + args.batch_videos]
            if len(batch) < 2:
                continue
            clip = torch.stack(batch, 0).to(device)
            tok0 = encode(backbone, clip[:, 0], backbone_cfg.features, frame_batch)
            tok1 = encode(backbone, clip[:, 1], backbone_cfg.features, frame_batch)
            raw0 = unit_tokens(tok0)
            raw1 = unit_tokens(tok1)
            outputs0 = {
                "raw": tok0,
                "dense": dense(tok0, 0.0),
                "med": med(tok0, 0.0),
                "rw": rw(tok0, 0.0),
            }
            outputs1 = {
                "raw": tok1,
                "dense": dense(tok1, 0.0),
                "med": med(tok1, 0.0),
                "rw": rw(tok1, 0.0),
            }
            for name in NAMES:
                _update(
                    stats,
                    name,
                    raw0,
                    unit_tokens(outputs0[name]),
                    raw1,
                    unit_tokens(outputs1[name]),
                    cross_frame=True,
                )
            print(f"  pairs {start + len(batch)}/{len(pairs)}", flush=True)

    rows = summarize(stats)
    print(f"\n{args.dataset} videos={stats['raw']['videos']}  LN+L2", flush=True)
    print(f"{'variant':8s} {'within':8s} {'cross':8s} {'ratio':7s}", flush=True)
    for name in NAMES:
        row = rows[name]
        print(
            f"{name:8s} {row['within']:8.3f} {row['cross']:8.3f} {row['ratio']:7.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
