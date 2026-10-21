"""Mean-shift feature groups vs dense P.

Uses FeatureModeCollapse on every patch (probe=0): blurring mean-shift at
bandwidth h, then each patch is replaced by the mean of the original tokens
in its mode. Self is not kept as its own weight; a mode mean excludes the
idea of a one-shot global mix.

Per h, besides the usual within / cross / frame-to-frame cosine:
  C        mean number of modes
  dom      fraction of patches in the largest mode
  in_mode  size-weighted mean pairwise cosine of the raw tokens inside a mode
  hold     fraction of patches whose mode mean stays within cosine 0.98
           across the next frame

Usage (inside the slotcurri image):
  python event_analysis/featcur_modes_probe.py --dataset movi_e
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
from slotcurri.modules.encoders import (  # noqa: E402
    FeatureModeCollapse,
    NcutRelationalLeveling,
    TimmExtractor,
)

BANDWIDTHS = (0.05, 0.1, 0.2, 0.5, 1.0)
HOLD_COS = 0.98


def _mode_labels(covered: torch.Tensor):
    labels = []
    counts = []
    for row in covered:
        _, inverse = torch.unique(row, dim=0, return_inverse=True)
        labels.append(inverse)
        counts.append(int(inverse.max().item()) + 1)
    return labels, counts


def _group_purity(raw_u: torch.Tensor, labels) -> tuple:
    """Size-weighted within-mode cosine of raw unit tokens, and dominant share."""
    numer = 0.0
    denom = 0
    dom = 0.0
    for row, inverse in zip(raw_u, labels):
        tally = torch.bincount(inverse)
        dom += float(tally.max()) / float(inverse.numel())
        for mode, count in enumerate(tally.tolist()):
            if count < 2:
                continue
            members = row[inverse == mode]
            mean = members.mean(0)
            cosine = (count * float(mean.square().sum()) - 1.0) / (count - 1)
            numer += cosine * count
            denom += count
    purity = numer / denom if denom else 1.0
    return purity, dom / len(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(CONFIGS))
    parser.add_argument("--data-dir", default="/workspace/dataset")
    parser.add_argument("--out-dir", default="event_analysis/featcur_spatial")
    parser.add_argument("--n-videos", type=int, default=128)
    parser.add_argument("--per-shard", type=int, default=6)
    parser.add_argument("--batch-videos", type=int, default=4)
    parser.add_argument(
        "--bandwidths",
        default=",".join(str(h) for h in BANDWIDTHS),
        help="comma-separated mean-shift bandwidths",
    )
    args = parser.parse_args()
    bandwidths = tuple(float(part) for part in args.bandwidths.split(",") if part)

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
    modes = FeatureModeCollapse(
        n_iter=8, delta=0.02, probe=0, h_min=0.02, chunk_size=2
    ).to(device)

    names = ["raw", "dense"] + [f"h{h:g}" for h in bandwidths]
    pairs = load_pairs(cfg, args.data_dir, args.n_videos, args.per_shard, rng)
    print(f"{args.dataset}: {len(pairs)} consecutive pairs", flush=True)
    stats = {name: _empty_stats() for name in names}
    extra = {
        name: {"c": 0.0, "dom": 0.0, "in_mode": 0.0, "hold": 0.0, "n": 0}
        for name in names
        if name not in ("raw", "dense")
    }
    tok0 = None
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
            outputs0 = {"raw": tok0, "dense": dense(tok0, 0.0)}
            outputs1 = {"raw": tok1, "dense": dense(tok1, 0.0)}
            for bandwidth in bandwidths:
                modes.bandwidth = bandwidth
                covered0 = modes(tok0, 0.0)
                covered1 = modes(tok1, 0.0)
                name = f"h{bandwidth:g}"
                outputs0[name] = covered0
                outputs1[name] = covered1
                labels0, counts0 = _mode_labels(covered0)
                labels1, counts1 = _mode_labels(covered1)
                purity0, dom0 = _group_purity(raw0, labels0)
                purity1, dom1 = _group_purity(raw1, labels1)
                hold = float(
                    ((unit_tokens(covered0) * unit_tokens(covered1)).sum(-1) > HOLD_COS)
                    .float()
                    .mean()
                )
                bucket = extra[name]
                frames = covered0.shape[0]
                bucket["c"] += sum(counts0) + sum(counts1)
                bucket["dom"] += (dom0 + dom1) * frames
                bucket["in_mode"] += (purity0 + purity1) * frames
                bucket["hold"] += hold * frames
                bucket["n"] += frames * 2
            for name in names:
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
    grid = _grid(tok0.shape[1])
    print(f"\n{args.dataset} grid={grid} videos={stats['raw']['videos']}", flush=True)
    print(
        f"{'variant':8s} {'within':8s} {'cross':8s} {'ratio':7s} "
        f"{'t_patch':8s} {'vs_raw':8s} {'C':7s} {'dom':7s} {'in_mode':8s} {'hold':7s}",
        flush=True,
    )
    recorded = {}
    for name in names:
        row = dict(rows[name])
        if name in extra:
            bucket = extra[name]
            row["C"] = bucket["c"] / bucket["n"]
            row["dom"] = bucket["dom"] / bucket["n"]
            row["in_mode"] = bucket["in_mode"] / bucket["n"]
            row["hold"] = bucket["hold"] / (bucket["n"] / 2)
        recorded[name] = row
        c_text = f"{row['C']:7.1f}" if "C" in row else f"{'':7s}"
        dom_text = f"{row['dom']:7.3f}" if "dom" in row else f"{'':7s}"
        in_text = f"{row['in_mode']:8.3f}" if "in_mode" in row else f"{'':8s}"
        hold_text = f"{row['hold']:7.3f}" if "hold" in row else f"{'':7s}"
        print(
            f"{name:8s} {row['within']:8.3f} {row['cross']:8.3f} {row['ratio']:7.3f} "
            f"{row['temporal_patch']:8.3f} {row['vs_raw']:8.3f} "
            f"{c_text} {dom_text} {in_text} {hold_text}",
            flush=True,
        )
    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "modes" if list(bandwidths) == list(BANDWIDTHS) else "modes_fine"
    path = os.path.join(args.out_dir, f"{args.dataset}_{suffix}.json")
    with open(path, "w") as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "n_videos": int(stats["raw"]["videos"]),
                "grid": grid,
                "bandwidths": list(bandwidths),
                "probe": 0,
                "hold_cos": HOLD_COS,
                "variants": recorded,
            },
            handle,
            indent=2,
        )
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
