"""8-neighbor R for purity only: does merge get a real λ2?

Curriculum P stays global. This graph is only for G_s = diag(a) S diag(a).
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "event_analysis")
from v38_featcur_mix_check import DATASETS, collect
from v38_pi_mechanism_check import spatial_halves
from slotcurri.modules.video import _relation_graph_S


def n8_R(z, grid):
    n, _ = z.shape
    z = F.normalize(z.float(), dim=-1)
    R = z.new_zeros(n, n)
    nbr = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y in range(grid):
        for x in range(grid):
            i = y * grid + x
            for dy, dx in nbr:
                yy, xx = y + dy, x + dx
                if 0 <= yy < grid and 0 <= xx < grid:
                    j = yy * grid + xx
                    R[i, j] = (z[i] * z[j]).sum().clamp_min(0.0)
    return R


def S_from_R(R, eps=1e-6):
    d = R.sum(-1).clamp_min(eps)
    inv = d.rsqrt()
    return inv[:, None] * R * inv[None, :]


def eigs_G(S, a):
    m = a > 0.5
    n = int(m.sum())
    if n < 2:
        return n, 0.0, 0.0, 0.0
    G = S[m][:, m]
    G = 0.5 * (G + G.T)
    ev = torch.linalg.eigvalsh(G)
    l1, l2 = float(ev[-1]), float(ev[-2])
    pi = max(l1 - max(l2, 0.0), 0.0)
    return n, l1, l2, pi


def ab_touch(gt, oa, ob, grid):
    a = (gt == oa).reshape(grid, grid)
    b = (gt == ob).reshape(grid, grid)
    n8 = 0
    n4 = 0
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        aa = a[max(dy, 0) : grid + min(dy, 0), max(dx, 0) : grid + min(dx, 0)]
        bb = b[max(-dy, 0) : grid + min(-dy, 0), max(-dx, 0) : grid + min(-dx, 0)]
        n8 += int((aa & bb).sum())
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        aa = a[max(dy, 0) : grid + min(dy, 0), max(dx, 0) : grid + min(dx, 0)]
        bb = b[max(-dy, 0) : grid + min(-dy, 0), max(-dx, 0) : grid + min(-dx, 0)]
        n4 += int((aa & bb).sum())
    return n8, n4


def one_frame(name, feat, gt, grid, bg_id, tag):
    ids = [int(i) for i in gt.unique().tolist() if int(i) != bg_id]
    sizes = sorted([(oid, int((gt == oid).sum())) for oid in ids], key=lambda t: -t[1])
    sizes = [(o, n) for o, n in sizes if n >= 8]
    if len(sizes) < 2:
        return
    oa, ob = sizes[0][0], sizes[1][0]
    n8, n4 = ab_touch(gt, oa, ob, grid)
    print(
        f"\n==== {name} {tag}  A n={sizes[0][1]} B n={sizes[1][1]}  "
        f"AB 8-nbr contacts={n8}  4-nbr={n4} ====",
        flush=True,
    )
    z = F.normalize(feat.float(), dim=-1)
    S_dense = _relation_graph_S(z.unsqueeze(0), 1e-6)[0]
    S_n8 = S_from_R(n8_R(z, grid))
    h1, _ = spatial_halves(gt, oa, grid)
    cases = [
        ("exclusive A", (gt == oa).float()),
        ("exclusive B", (gt == ob).float()),
        ("split A", h1),
        ("A∪B", ((gt == oa) | (gt == ob)).float()),
    ]
    print(f"{'case':16s} {'dense λ1/λ2/π  π/λ1':>32s}   {'n8 λ1/λ2/π  π/λ1':>32s}", flush=True)
    for lab, a in cases:
        nd, l1d, l2d, pid = eigs_G(S_dense, a)
        nn, l1n, l2n, pin = eigs_G(S_n8, a)
        print(
            f"{lab:16s}  {l1d:5.3f}/{l2d:5.3f}/{pid:5.3f} {pid/max(l1d,1e-8):5.2f}"
            f"     {l1n:5.3f}/{l2n:5.3f}/{pin:5.3f} {pin/max(l1n,1e-8):5.2f}",
            flush=True,
        )


def main():
    device = torch.device("cpu")
    for name, (cfg, min_obj, bg_id) in DATASETS.items():
        fr = collect(name, cfg, min_obj, bg_id, "/workspace/dataset", device, n_keep=1, max_scan=30)[0]
        one_frame(name, fr["raw"], fr["gt"], fr["grid"], bg_id, "raw")


if __name__ == "__main__":
    main()
