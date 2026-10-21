"""Spatial FC filters with the center removed, vs dense ReLU-cosine P.

Each filter replaces a token by a weighted average of other tokens only.
A row whose remaining weight is ~0 keeps the original token.

  raw        X
  dense      current FC at s=0: P = row-normalize(ReLU(z z^T)), diagonal 0
  gaussian   spatial Gaussian, sigma=2 patches, support |dy|,|dx| <= 6, center 0
  pool       ~4x4 blocks; each token is the mean of the other tokens in its block
  bilateral  same spatial kernel times exp((cos(z_i,z_j)-1)/tau), tau=0.1, center 0

Out-of-grid neighbors are dropped and the remaining weights are renormalized.
No reflect pad, so a border token cannot meet itself through the padding.

Cosine matches event_analysis/featcur_xvid_probe.py part A: LayerNorm, then L2.
within  = mean pairwise cosine inside a frame
cross   = mean dot product of per-frame mean tokens, one frame per video
temporal_patch = mean same-location cosine between consecutive frames
temporal_mean  = cosine of those two frames' mean tokens
vs_raw  = mean same-location cosine between the filter output and X

Usage (inside the slotcurri image):
  python event_analysis/featcur_spatial_probe.py --dataset movi_e
  python event_analysis/featcur_spatial_probe.py --self-check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from global_winner_probe import CONFIGS  # noqa: E402
from slotcurri import configuration  # noqa: E402
from slotcurri.data import transforms  # noqa: E402
from slotcurri.featcur_start import _shard_paths  # noqa: E402
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor  # noqa: E402

SIGMA = 2.0
RADIUS = 6
TAU = 0.1
CELL = 4
EPS = 1e-8
VARIANTS = ("raw", "dense", "gaussian", "pool", "bilateral")


def _grid(n_tokens: int) -> int:
    grid = int(round(n_tokens ** 0.5))
    if grid * grid != n_tokens:
        raise ValueError(f"expected a square token grid, got N={n_tokens}")
    return grid


def block_edges(n: int, cell: int) -> list:
    """Partition 0..n into about n/cell contiguous blocks."""
    n_cells = max(1, int(round(n / float(cell))))
    edges = [int(round(i * n / n_cells)) for i in range(n_cells + 1)]
    edges[0] = 0
    edges[-1] = n
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = min(n, edges[i - 1] + 1)
    edges[-1] = n
    return edges


def gaussian_bilateral(
    x: torch.Tensor,
    sigma: float = SIGMA,
    radius: int = RADIUS,
    tau: Optional[float] = None,
) -> torch.Tensor:
    """x (B, N, D) -> mix of other tokens. tau=None is spatial Gaussian."""
    batch, n_tokens, dim = x.shape
    grid = _grid(n_tokens)
    feat = x.view(batch, grid, grid, dim)
    feat_n = F.normalize(feat, dim=-1) if tau is not None else None
    acc = torch.zeros_like(feat)
    weight_sum = feat.new_zeros(batch, grid, grid, 1)
    denom = 2.0 * sigma * sigma
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            y0 = max(0, -dy)
            y1 = grid - max(0, dy)
            x0 = max(0, -dx)
            x1 = grid - max(0, dx)
            if y0 >= y1 or x0 >= x1:
                continue
            spatial = math.exp(-(dy * dy + dx * dx) / denom)
            src = feat[:, y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            if feat_n is None:
                weight = feat.new_full((1, 1, 1, 1), spatial)
            else:
                cos = (
                    feat_n[:, y0:y1, x0:x1]
                    * feat_n[:, y0 + dy : y1 + dy, x0 + dx : x1 + dx]
                ).sum(-1, keepdim=True)
                weight = spatial * torch.exp((cos - 1.0) / tau)
            acc[:, y0:y1, x0:x1] = acc[:, y0:y1, x0:x1] + weight * src
            weight_sum[:, y0:y1, x0:x1] = weight_sum[:, y0:y1, x0:x1] + weight
    mixed = torch.where(
        weight_sum < EPS, feat, acc / weight_sum.clamp_min(EPS)
    )
    return mixed.view(batch, n_tokens, dim)


def pool_exclude_self(x: torch.Tensor, cell: int = CELL) -> torch.Tensor:
    """Each token becomes the mean of the other tokens in its block."""
    batch, n_tokens, dim = x.shape
    grid = _grid(n_tokens)
    feat = x.view(batch, grid, grid, dim)
    edges = block_edges(grid, cell)
    out = feat.clone()
    for y0, y1 in zip(edges[:-1], edges[1:]):
        for x0, x1 in zip(edges[:-1], edges[1:]):
            if y1 <= y0 or x1 <= x0:
                continue
            block = feat[:, y0:y1, x0:x1]
            height, width = block.shape[1], block.shape[2]
            count = height * width
            if count <= 1:
                continue
            flat = block.reshape(batch, count, dim)
            mixed = (flat.sum(1, keepdim=True) - flat) / (count - 1)
            out[:, y0:y1, x0:x1] = mixed.view(batch, height, width, dim)
    return out.view(batch, n_tokens, dim)


def _brute_local(x: torch.Tensor, tau: Optional[float]) -> torch.Tensor:
    """Reference mix on a single (H, W, D) map. Used by the self-check only."""
    grid, _, dim = x.shape
    out = torch.empty_like(x)
    denom = 2.0 * SIGMA * SIGMA
    z = F.normalize(x, dim=-1) if tau is not None else None
    for y in range(grid):
        for x_i in range(grid):
            acc = x.new_zeros(dim)
            weight_sum = x.new_zeros(())
            for dy in range(-RADIUS, RADIUS + 1):
                for dx in range(-RADIUS, RADIUS + 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = y + dy, x_i + dx
                    if not (0 <= yy < grid and 0 <= xx < grid):
                        continue
                    weight = math.exp(-(dy * dy + dx * dx) / denom)
                    if z is not None:
                        weight *= math.exp((float((z[y, x_i] * z[yy, xx]).sum()) - 1.0) / tau)
                    acc = acc + weight * x[yy, xx]
                    weight_sum = weight_sum + weight
            out[y, x_i] = x[y, x_i] if float(weight_sum) < EPS else acc / weight_sum
    return out


def self_check() -> None:
    torch.manual_seed(0)
    grid = 8
    x = torch.randn(2, grid * grid, 4)
    constant = torch.ones_like(x)
    for name, fn in (
        ("gaussian", lambda t: gaussian_bilateral(t)),
        ("bilateral", lambda t: gaussian_bilateral(t, tau=TAU)),
        ("pool", pool_exclude_self),
    ):
        got = fn(constant)
        if not torch.allclose(got, constant, atol=1e-5):
            raise AssertionError(f"{name} changed a constant field")

    gauss = gaussian_bilateral(x)[0].view(grid, grid, 4)
    bila = gaussian_bilateral(x, tau=TAU)[0].view(grid, grid, 4)
    ref_x = x[0].view(grid, grid, 4)
    if not torch.allclose(gauss, _brute_local(ref_x, None), atol=1e-5):
        raise AssertionError("gaussian disagrees with the direct sum")
    if not torch.allclose(bila, _brute_local(ref_x, TAU), atol=1e-5):
        raise AssertionError("bilateral disagrees with the direct sum")

    impulse = torch.zeros(1, grid * grid, 1)
    impulse[0, (grid // 2) * grid + grid // 2, 0] = 1.0
    center = gaussian_bilateral(impulse).view(grid, grid, 1)[grid // 2, grid // 2, 0]
    if float(center.abs()) > 1e-6:
        raise AssertionError(f"gaussian kept the center impulse ({float(center)})")

    pooled = pool_exclude_self(x, cell=4)[0].view(grid, grid, 4)
    block = ref_x[:4, :4].reshape(-1, 4)
    expect = (block.sum(0, keepdim=True) - block) / (block.shape[0] - 1)
    if not torch.allclose(pooled[:4, :4].reshape(-1, 4), expect, atol=1e-5):
        raise AssertionError("pool is not the leave-one-out block mean")

    edges24 = block_edges(24, CELL)
    edges37 = block_edges(37, CELL)
    if edges24 != [0, 4, 8, 12, 16, 20, 24]:
        raise AssertionError(f"MOVi block edges {edges24}")
    if edges37[0] != 0 or edges37[-1] != 37 or any(
        b <= a for a, b in zip(edges37, edges37[1:])
    ):
        raise AssertionError(f"YTVIS block edges {edges37}")
    print("self-check ok", flush=True)
    print(f"  movi blocks {edges24}", flush=True)
    print(f"  ytvis blocks {edges37}", flush=True)


def _frame_hwc(video: np.ndarray, index: int) -> np.ndarray:
    frame = video[index]
    if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    return frame


def load_pairs(cfg, data_dir, n_videos, per_shard, rng):
    """Two consecutive frames, transformed together so the crop matches."""
    tf = transforms.build(cfg.dataset.val_pipeline.transforms)["video"]
    paths = list(_shard_paths(str(cfg.dataset.train_shards), data_dir))
    paths = [paths[i] for i in rng.permutation(len(paths))]
    pairs = []
    for path in paths:
        ds = wds.WebDataset([path], shardshuffle=False, nodesplitter=lambda urls: urls)
        ds = ds.map(
            lambda sample: {
                k: v for k, v in sample.items() if "video" in k or str(k).startswith("__")
            }
        ).decode("rgb")
        taken = 0
        for sample in ds:
            video = next(
                v for k, v in sample.items() if "video" in k and not str(k).startswith("__")
            )
            video = np.asarray(video)
            if video.ndim != 4 or video.shape[0] < 2:
                continue
            index = int(rng.integers(0, video.shape[0] - 1))
            clip = np.stack(
                [_frame_hwc(video, index), _frame_hwc(video, index + 1)], axis=0
            )
            transformed = tf(clip)
            pairs.append(transformed)
            taken += 1
            if taken >= per_shard or len(pairs) >= n_videos:
                break
        if len(pairs) >= n_videos:
            break
    return pairs


def encode(backbone, images, feature_key, batch_size):
    pieces = []
    for start in range(0, images.shape[0], batch_size):
        tokens = backbone(images[start : start + batch_size])
        if isinstance(tokens, dict):
            tokens = tokens[feature_key]
        pieces.append(tokens.float())
    return torch.cat(pieces, 0)


def unit_tokens(tokens: torch.Tensor) -> torch.Tensor:
    return F.normalize(F.layer_norm(tokens, (tokens.shape[-1],)), dim=-1)


def apply_filters(tokens, dense):
    return {
        "raw": tokens,
        "dense": dense(tokens, 0.0),
        "gaussian": gaussian_bilateral(tokens),
        "pool": pool_exclude_self(tokens),
        "bilateral": gaussian_bilateral(tokens, tau=TAU),
    }


def _empty_stats():
    return {
        "within_sum": 0.0,
        "within_n": 0,
        "mean_sum": None,
        "mean_sq": 0.0,
        "videos": 0,
        "temporal_patch": 0.0,
        "temporal_mean": 0.0,
        "temporal_n": 0,
        "vs_raw": 0.0,
        "vs_raw_n": 0,
    }


def _update(stats, name, raw_u, filt_u, raw_u1, filt_u1, cross_frame: bool):
    st = stats[name]
    for frame_u in (filt_u, filt_u1):
        mean = frame_u.mean(1)
        n_tokens = frame_u.shape[1]
        within = (n_tokens * mean.square().sum(-1) - 1.0) / (n_tokens - 1)
        st["within_sum"] += float(within.sum())
        st["within_n"] += int(within.shape[0])
    if cross_frame:
        mean0 = filt_u.mean(1)
        if st["mean_sum"] is None:
            st["mean_sum"] = mean0.sum(0)
        else:
            st["mean_sum"] = st["mean_sum"] + mean0.sum(0)
        st["mean_sq"] += float(mean0.square().sum())
        st["videos"] += int(mean0.shape[0])
    patch = (filt_u * filt_u1).sum(-1).mean(-1)
    mean_cos = F.cosine_similarity(filt_u.mean(1), filt_u1.mean(1), dim=-1)
    st["temporal_patch"] += float(patch.sum())
    st["temporal_mean"] += float(mean_cos.sum())
    st["temporal_n"] += int(patch.shape[0])
    vs = (filt_u * raw_u).sum(-1).mean(-1)
    vs1 = (filt_u1 * raw_u1).sum(-1).mean(-1)
    st["vs_raw"] += float(vs.sum() + vs1.sum())
    st["vs_raw_n"] += int(vs.shape[0] + vs1.shape[0])


def summarize(stats):
    rows = {}
    for name, st in stats.items():
        videos = st["videos"]
        cross = (float(st["mean_sum"].square().sum()) - st["mean_sq"]) / (
            videos * (videos - 1)
        )
        within = st["within_sum"] / st["within_n"]
        rows[name] = {
            "within": within,
            "cross": cross,
            "ratio": cross / within,
            "temporal_patch": st["temporal_patch"] / st["temporal_n"],
            "temporal_mean": st["temporal_mean"] / st["temporal_n"],
            "vs_raw": st["vs_raw"] / st["vs_raw_n"],
        }
    return rows


def encode_batch_size(model_name: str, input_size: int) -> int:
    name = str(model_name)
    if "vit_base" in name and int(input_size) >= 500:
        return 4
    if "vit_base" in name:
        return 8
    return 16


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(CONFIGS))
    parser.add_argument("--data-dir", default="/workspace/dataset")
    parser.add_argument("--out-dir", default="event_analysis/featcur_spatial")
    parser.add_argument("--n-videos", type=int, default=128)
    parser.add_argument("--per-shard", type=int, default=6)
    parser.add_argument("--batch-videos", type=int, default=8)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    self_check()
    if args.self_check:
        return
    if args.dataset is None:
        raise SystemExit("--dataset is required unless --self-check")

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

    started = time.time()
    pairs = load_pairs(cfg, args.data_dir, args.n_videos, args.per_shard, rng)
    print(
        f"{args.dataset}: {len(pairs)} consecutive pairs "
        f"({time.time() - started:.0f}s)",
        flush=True,
    )
    if len(pairs) < 4:
        raise RuntimeError(f"need at least 4 videos, got {len(pairs)}")

    stats = {name: _empty_stats() for name in VARIANTS}
    for start in range(0, len(pairs), args.batch_videos):
        batch = pairs[start : start + args.batch_videos]
        if len(batch) < 2:
            continue
        clip = torch.stack(batch, 0).to(device)
        images0 = clip[:, 0]
        images1 = clip[:, 1]
        tok0 = encode(backbone, images0, backbone_cfg.features, frame_batch)
        tok1 = encode(backbone, images1, backbone_cfg.features, frame_batch)
        filt0 = apply_filters(tok0, dense)
        filt1 = apply_filters(tok1, dense)
        raw0 = unit_tokens(filt0["raw"])
        raw1 = unit_tokens(filt1["raw"])
        for name in VARIANTS:
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
    header = (
        f"{'variant':10s} {'within':8s} {'cross':8s} {'ratio':7s} "
        f"{'t_patch':8s} {'t_mean':8s} {'vs_raw':8s}"
    )
    print(
        f"\n{args.dataset}  grid={grid}  videos={rows and stats['raw']['videos']}  "
        f"sigma={SIGMA} tau={TAU} cell={CELL} radius={RADIUS} exclude_self=1",
        flush=True,
    )
    print(header, flush=True)
    for name in VARIANTS:
        row = rows[name]
        print(
            f"{name:10s} {row['within']:8.3f} {row['cross']:8.3f} {row['ratio']:7.3f} "
            f"{row['temporal_patch']:8.3f} {row['temporal_mean']:8.3f} {row['vs_raw']:8.3f}",
            flush=True,
        )

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.dataset}.json")
    payload = {
        "dataset": args.dataset,
        "n_videos": int(stats["raw"]["videos"]),
        "grid": grid,
        "block_edges": block_edges(grid, CELL),
        "sigma": SIGMA,
        "tau": TAU,
        "cell": CELL,
        "radius": RADIUS,
        "exclude_self": True,
        "variants": rows,
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
