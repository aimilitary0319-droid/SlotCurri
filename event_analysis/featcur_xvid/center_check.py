"""How far is the batch mean of P X from mu, and does it matter for xvid_energy?

P is row-stochastic, so mean(P X) != mu = mean(X). xvid_energy rescales P X - mu,
so the offset mean(P X) - mu is amplified by g and lands on every token.

  xvid    mu + g  (P X - mu),         g^2  = E|X - mu|^2 / E|P X - mu|^2   (implemented)
  xvid_c  mu + g' (P X - mean(P X)),  g'^2 = E|X - mu|^2 / E|P X - mean(P X)|^2

Reports the offset share of E|P X - mu|^2, the LN + L2 key cosine (within frame /
across videos, as featcur_xvid_probe A), and the mean 8-neighbor raw cosine that
the lambda1 gate graph reads from X^bind.

Usage (inside the slotcurri image):
  python event_analysis/featcur_xvid/center_check.py --dataset movi_e
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "event_analysis"))
from global_winner_probe import CONFIGS, load_frames  # noqa: E402
from slotcurri import configuration  # noqa: E402
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor  # noqa: E402
from slotcurri.modules.encoders import _N8_OFFSETS, _square_grid  # noqa: E402


def key_cosine(tok):
    u = F.normalize(F.layer_norm(tok, (tok.shape[-1],)), dim=-1)
    m = u.mean(1)
    n, v = tok.shape[1], m.shape[0]
    within = ((n * m.square().sum(-1) - 1.0) / (n - 1)).mean().item()
    s = m.sum(0)
    across = ((s.square().sum() - m.square().sum()) / (v * (v - 1))).item()
    return within, across


def n8_cosine(tok):
    b, n, d = tok.shape
    grid = _square_grid(n)
    z = F.normalize(tok, dim=-1).view(b, grid, grid, d)
    zp = F.pad(z, (0, 0, 1, 1, 1, 1))
    tot, cnt = 0.0, 0
    for dy, dx in _N8_OFFSETS:
        nb = zp[:, 1 - dy : 1 - dy + grid, 1 - dx : 1 - dx + grid]
        valid = nb.norm(dim=-1) > 0
        c = (z * nb).sum(-1).clamp_min(0.0)
        tot += float(c[valid].sum())
        cnt += int(valid.sum())
    return tot / cnt


def energy(t, c):
    return (t - c).square().sum(-1).mean()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(CONFIGS))
    p.add_argument("--data-dir", default="/workspace/dataset")
    p.add_argument("--n-videos", type=int, default=64)
    p.add_argument("--per-shard", type=int, default=4)
    p.add_argument("--batch-videos", type=int, default=32)
    args = p.parse_args()

    torch.set_num_threads(min(32, os.cpu_count() or 1))
    rng = np.random.default_rng(0)
    cfg = configuration.load_config(CONFIGS[args.dataset])
    bb = cfg.model.encoder.backbone
    backbone = TimmExtractor(
        model=bb.model, pretrained=True, frozen=True, features=bb.features,
        model_kwargs=dict(bb.get("model_kwargs") or {}) or None,
    ).eval()
    level = NcutRelationalLeveling(chunk_size=8, barrier=False)
    frames = load_frames(cfg, args.data_dir, args.n_videos, args.per_shard, rng)

    rows = {k: [] for k in ("raw", "dense", "xvid", "xvid_c")}
    stats = {"offset_share": [], "offset_after_g": [], "g": [], "g_c": []}
    for b0 in range(0, len(frames), args.batch_videos):
        x_img = torch.stack(frames[b0 : b0 + args.batch_videos], 0)
        if x_img.shape[0] < 8:
            continue
        tok = []
        for c0 in range(0, x_img.shape[0], 8):
            t = backbone(x_img[c0 : c0 + 8])
            tok.append((t[bb.features] if isinstance(t, dict) else t).float())
        x = torch.cat(tok, 0)
        px = level._level(x)
        mu = x.mean(dim=(0, 1), keepdim=True)
        pbar = px.mean(dim=(0, 1), keepdim=True)
        e_raw = energy(x, mu)
        g = (e_raw / energy(px, mu)).sqrt()
        g_c = (e_raw / energy(px, pbar)).sqrt()
        off = (pbar - mu).square().sum()
        stats["offset_share"].append(float(off / energy(px, mu)))
        stats["offset_after_g"].append(float(g * g * off / e_raw))
        stats["g"].append(float(g))
        stats["g_c"].append(float(g_c))
        variants = {
            "raw": x,
            "dense": px,
            "xvid": mu + g * (px - mu),
            "xvid_c": mu + g_c * (px - pbar),
        }
        for k, v in variants.items():
            rows[k].append((*key_cosine(v), n8_cosine(v)))

    print(f"{args.dataset}: {len(frames)} videos, batch {args.batch_videos}")
    for k, v in stats.items():
        print(f"  {k:15s} {np.mean(v):.4f}")
    print(f"  {'variant':8s} {'within':>8s} {'across':>8s} {'n8_raw_cos':>11s}")
    for k, v in rows.items():
        a = np.mean(np.array(v), 0)
        print(f"  {k:8s} {a[0]:8.3f} {a[1]:8.3f} {a[2]:11.3f}")


if __name__ == "__main__":
    main()
