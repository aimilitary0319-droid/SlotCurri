"""Dense P with the other-videos mean removed from the affinity.

  z = X / ||X||
  mu_{-v} = mean of z over every patch in the batch that is not video v
  b_i = [z_i^T mu_{-v}]_+
  W_ij = [z_i^T z_j - b_i]_+,  W_ii = 0
  X_rel = row-normalize(W) X     (a zero row keeps X)

Batch size is the training per-GPU video batch. Both frames of the other
videos enter mu. Cosine stats match featcur_spatial_probe.py.

  kept   fraction of patches whose row is empty, so they stay X
  edges  fraction of off-diagonal affinities that stay positive

Usage (inside the slotcurri image):
  python event_analysis/featcur_wsub_probe.py --dataset ytvis
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
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
    summarize,
    unit_tokens,
)
from slotcurri.modules.encoders import TimmExtractor  # noqa: E402

EPS = 1e-6
NAMES = ("raw", "dense", "wsub")


def leave_video_mean(z: torch.Tensor) -> torch.Tensor:
    """z (V, T, N, D) -> mu_{-v} (V, D). Mean of normalized patches in other videos."""
    videos, frames, n_tokens, _ = z.shape
    if videos < 2:
        raise ValueError("leave-one-video mean needs at least 2 videos")
    total = z.sum(dim=(0, 1, 2))
    video_sum = z.sum(dim=(1, 2))
    denom = (videos - 1) * frames * n_tokens
    return (total - video_sum) / denom


def mix_row(x: torch.Tensor, z: torch.Tensor, bias: Optional[torch.Tensor]):
    """x, z (N, D). bias (N,) is subtracted inside the ReLU. Returns mix, kept, edges."""
    weight = z @ z.transpose(0, 1)
    if bias is not None:
        weight = weight - bias[:, None]
    weight = weight.clamp_min(0)
    weight.fill_diagonal_(0)
    row = weight.sum(dim=-1, keepdim=True)
    kept = row.squeeze(-1) < EPS
    mixed = torch.where(kept[:, None], x, weight @ x / row.clamp_min(EPS))
    n_tokens = x.shape[0]
    edges = (weight > 0).sum().float() / (n_tokens * (n_tokens - 1))
    return mixed, kept.float().mean(), edges


def apply_batch(tokens0, tokens1):
    """tokens (V, N, D) at two times. Returns dense and wsub for each time, plus graph stats."""
    z0 = F.normalize(tokens0, dim=-1)
    z1 = F.normalize(tokens1, dim=-1)
    stacked = torch.stack((z0, z1), dim=1)
    mu = leave_video_mean(stacked)
    videos = tokens0.shape[0]
    out0 = {"dense": [], "wsub": []}
    out1 = {"dense": [], "wsub": []}
    graph = {name: {"kept": 0.0, "edges": 0.0, "bias": 0.0, "n": 0} for name in ("dense", "wsub")}
    for video in range(videos):
        for tokens, z, dest in (
            (tokens0[video], z0[video], out0),
            (tokens1[video], z1[video], out1),
        ):
            dense, kept_d, edges_d = mix_row(tokens, z, None)
            bias = (z @ mu[video]).clamp_min(0)
            wsub, kept_w, edges_w = mix_row(tokens, z, bias)
            dest["dense"].append(dense)
            dest["wsub"].append(wsub)
            graph["dense"]["kept"] += float(kept_d)
            graph["dense"]["edges"] += float(edges_d)
            graph["wsub"]["kept"] += float(kept_w)
            graph["wsub"]["edges"] += float(edges_w)
            graph["wsub"]["bias"] += float(bias.mean())
            graph["dense"]["n"] += 1
            graph["wsub"]["n"] += 1
    packed = tuple(
        {name: torch.stack(dest[name], 0) for name in ("dense", "wsub")}
        for dest in (out0, out1)
    )
    return packed, graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(CONFIGS))
    parser.add_argument("--data-dir", default="/workspace/dataset")
    parser.add_argument("--out-dir", default="event_analysis/featcur_spatial")
    parser.add_argument("--n-videos", type=int, default=128)
    parser.add_argument("--per-shard", type=int, default=6)
    parser.add_argument("--batch-videos", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    cfg = configuration.load_config(CONFIGS[args.dataset])
    backbone_cfg = cfg.model.encoder.backbone
    input_size = int(cfg.dataset.val_pipeline.transforms.input_size)
    frame_batch = encode_batch_size(backbone_cfg.model, input_size)
    video_batch = int(args.batch_videos) or int(cfg.globals.BATCH_SIZE_PER_GPU)
    backbone = TimmExtractor(
        model=backbone_cfg.model,
        pretrained=True,
        frozen=True,
        features=backbone_cfg.features,
        model_kwargs=dict(backbone_cfg.get("model_kwargs") or {}) or None,
    ).to(device).eval()

    pairs = load_pairs(cfg, args.data_dir, args.n_videos, args.per_shard, rng)
    print(
        f"{args.dataset}: {len(pairs)} pairs, video batch {video_batch}",
        flush=True,
    )
    stats = {name: _empty_stats() for name in NAMES}
    graph_acc = {name: {"kept": 0.0, "edges": 0.0, "bias": 0.0, "n": 0} for name in ("dense", "wsub")}
    tok0 = None
    with torch.no_grad():
        for start in range(0, len(pairs), video_batch):
            batch = pairs[start : start + video_batch]
            if len(batch) < 2:
                continue
            clip = torch.stack(batch, 0).to(device)
            tok0 = encode(backbone, clip[:, 0], backbone_cfg.features, frame_batch)
            tok1 = encode(backbone, clip[:, 1], backbone_cfg.features, frame_batch)
            (filt0, filt1), graph = apply_batch(tok0, tok1)
            for name in ("dense", "wsub"):
                for key in ("kept", "edges", "bias", "n"):
                    graph_acc[name][key] += graph[name][key]
            raw0 = unit_tokens(tok0)
            raw1 = unit_tokens(tok1)
            outputs0 = {"raw": tok0, "dense": filt0["dense"], "wsub": filt0["wsub"]}
            outputs1 = {"raw": tok1, "dense": filt1["dense"], "wsub": filt1["wsub"]}
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
    for name in ("dense", "wsub"):
        count = graph_acc[name]["n"]
        rows[name]["kept"] = graph_acc[name]["kept"] / count
        rows[name]["edges"] = graph_acc[name]["edges"] / count
        rows[name]["bias"] = graph_acc[name]["bias"] / count
    grid = _grid(tok0.shape[1])
    print(
        f"\n{args.dataset} grid={grid} videos={stats['raw']['videos']} batch={video_batch}",
        flush=True,
    )
    print(
        f"{'variant':8s} {'within':8s} {'cross':8s} {'ratio':7s} "
        f"{'t_patch':8s} {'vs_raw':8s} {'kept':7s} {'edges':7s} {'bias':7s}",
        flush=True,
    )
    for name in NAMES:
        row = rows[name]
        kept = f"{row['kept']:7.3f}" if "kept" in row else f"{'':7s}"
        edges = f"{row['edges']:7.3f}" if "edges" in row else f"{'':7s}"
        bias = f"{row['bias']:7.3f}" if "bias" in row else f"{'':7s}"
        print(
            f"{name:8s} {row['within']:8.3f} {row['cross']:8.3f} {row['ratio']:7.3f} "
            f"{row['temporal_patch']:8.3f} {row['vs_raw']:8.3f} {kept} {edges} {bias}",
            flush=True,
        )
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.dataset}_wsub.json")
    with open(path, "w") as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "n_videos": int(stats["raw"]["videos"]),
                "grid": grid,
                "video_batch": video_batch,
                "frames_in_mu": 2,
                "variants": rows,
            },
            handle,
            indent=2,
        )
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
