#!/usr/bin/env python3
"""Oracle: does induced n8 π mark two-object merges the way v39 G_s does not?

Prints exclusive / union_touch / union_apart on 0/1 masks for:
  v39 G_s π, v37 Gram π, induced n8 π, induced ρ.

Toy grid always. Real DINO+GT if --data-dir has shards.

Usage:
  python event_analysis/v39_induced_n8_probe.py
  python event_analysis/v39_induced_n8_probe.py --data-dir /mnt/ssd2/hmlee/dataset --dataset movi_c --max-clips 20
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slotcurri.modules.video import (  # noqa: E402
    spectral_graph_n8_induced_impurity,
    spectral_graph_n8_induced_purity,
    spectral_graph_n8_slot_purity,
    spectral_slot_purity,
)


def _mask_xy(grid, ys, ye, xs, xe):
    a = torch.zeros(grid * grid)
    for y in range(ys, ye):
        for x in range(xs, xe):
            a[y * grid + x] = 1.0
    return a


def _row(name, att, feat):
    v39 = float(spectral_graph_n8_slot_purity(att, feat)[0, 0])
    gram = float(spectral_slot_purity(att, feat)[0, 0])
    ind = float(spectral_graph_n8_induced_purity(att, feat, support_rel=0.0)[0, 0])
    rho = float(
        spectral_graph_n8_induced_impurity(
            att, feat, support_rel=0.0, fiedler_tau=0.05
        )[0, 0]
    )
    return name, v39, gram, ind, rho


def toy():
    grid = 8
    n = grid * grid
    z = torch.zeros(1, n, 4)
    z[..., 0] = 1.0
    a_ex = _mask_xy(grid, 1, 4, 1, 4)
    a_touch = a_ex + _mask_xy(grid, 1, 4, 4, 7)
    a_apart = a_ex + _mask_xy(grid, 5, 8, 5, 8)
    att = torch.zeros(1, 1, n)
    print(
        f"{'case':16s}  {'v39 π':>8s}  {'v37 Gram':>8s}  {'ind π':>8s}  {'ind ρ':>8s}",
        flush=True,
    )
    for name, a in (
        ("exclusive", a_ex),
        ("union_touch", a_touch),
        ("union_apart", a_apart),
    ):
        att[0, 0] = a
        lab, v39, gram, ind, rho = _row(name, att, z)
        print(
            f"{lab:16s}  {v39:8.3f}  {gram:8.3f}  {ind:8.3f}  {rho:8.3f}",
            flush=True,
        )


def _touch8(a: torch.Tensor, b: torch.Tensor) -> bool:
    g = int(math.sqrt(a.numel()))
    aa = a.view(g, g) > 0.25
    bb = b.view(g, g) > 0.25
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            a_s = aa[max(dy, 0) : g + min(dy, 0), max(dx, 0) : g + min(dx, 0)]
            b_s = bb[max(-dy, 0) : g + min(-dy, 0), max(-dx, 0) : g + min(-dx, 0)]
            if bool((a_s & b_s).any()):
                return True
    return False


@torch.no_grad()
def real(data_dir, dataset, max_clips):
    from v38_featcur_mix_check import DATASETS, collect

    if dataset == "movi":
        dataset = "movi_c"
    cfg, min_obj, bg_id = DATASETS[dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = collect(
        dataset, cfg, min_obj, bg_id, data_dir, device, n_keep=max_clips, max_scan=max(40, max_clips * 3)
    )
    buckets = {
        k: {"v39": [], "gram": [], "ind": [], "rho": []}
        for k in ("excl", "union_touch", "union_apart")
    }
    for fr in frames:
        feat = fr["raw"].unsqueeze(0)
        gt = fr["gt"]
        grid = int(fr["grid"])
        ids = [int(i) for i in gt.unique().tolist() if int(i) != bg_id]
        sizes = sorted(((i, int((gt == i).sum())) for i in ids), key=lambda t: -t[1])
        sizes = [(i, n) for i, n in sizes if n >= 8]
        if len(sizes) < 2:
            continue
        oa, ob = sizes[0][0], sizes[1][0]
        a_ex = (gt == oa).float().view(1, 1, -1)
        a_un = ((gt == oa) | (gt == ob)).float().view(1, 1, -1)
        tag = "union_touch" if _touch8(gt == oa, gt == ob) else "union_apart"
        for name, a in (("excl", a_ex), (tag, a_un)):
            _, v39, gram, ind, rho = _row(name, a, feat)
            buckets[name]["v39"].append(v39)
            buckets[name]["gram"].append(gram)
            buckets[name]["ind"].append(ind)
            buckets[name]["rho"].append(rho)
    print(
        f"\n=== {dataset}  clips={len(frames)}  "
        f"(mean on GT 0/1 masks) ===",
        flush=True,
    )
    print(
        f"{'tag':16s}  n  {'v39 π':>8s}  {'v37 Gram':>8s}  {'ind π':>8s}  {'ind ρ':>8s}",
        flush=True,
    )
    for tag, rec in buckets.items():
        n = len(rec["ind"])
        if n == 0:
            print(f"{tag:16s}  0", flush=True)
            continue
        mean = {k: sum(v) / n for k, v in rec.items()}
        print(
            f"{tag:16s}  {n:d}  {mean['v39']:8.3f}  {mean['gram']:8.3f}  "
            f"{mean['ind']:8.3f}  {mean['rho']:8.3f}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--dataset", default="movi_c", choices=("ytvis", "movi_c", "movi"))
    ap.add_argument("--max-clips", type=int, default=20)
    args = ap.parse_args()
    print("======== TOY 8x8 identical features ========", flush=True)
    toy()
    if args.data_dir:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        real(args.data_dir, args.dataset, args.max_clips)


if __name__ == "__main__":
    main()
