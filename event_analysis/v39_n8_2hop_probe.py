"""v39 gate graphs: 8-nbr 1-hop vs two-step propagate vs dense.

  n8:     R = ReLU-cos on N8, S = D^{-1/2} R D^{-1/2}     (v39)
  n8^2:   use S@S as the relation in G (two propagations on the same graph)
  R8^2:   R2 = R8 @ R8, S = normalize(R2)                  (2-hop affinity)
  n24:    ReLU-cos on Chebyshev d<=2 (5x5 stencil)         (wider neighborhood)
  dense:  global ReLU-cos                                  (v38)

π is the locked v39 formula on 0/1 GT masks, exact eigh on support.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "event_analysis")
from v38_featcur_mix_check import DATASETS, collect
from v38_n8_purity_probe import S_from_R, ab_touch, n8_R
from v38_pi_mechanism_check import spatial_halves
from slotcurri.modules.video import _relation_graph_S

EPS = 1e-6


def stencil_R(z, grid, radius):
    n, _ = z.shape
    z = F.normalize(z.float(), dim=-1)
    R = z.new_zeros(n, n)
    for y in range(grid):
        for x in range(grid):
            i = y * grid + x
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < grid and 0 <= xx < grid:
                        j = yy * grid + xx
                        R[i, j] = (z[i] * z[j]).sum().clamp_min(0.0)
    return R


def eigs_G(S, a, square_induced=False):
    m = a > 0.5
    n = int(m.sum())
    if n < 2:
        return n, 0.0, 0.0
    G = 0.5 * (S[m][:, m] + S[m][:, m].T)
    if square_induced:
        G = G @ G
    ev = torch.linalg.eigvalsh(G)
    return n, float(ev[-1]), float(ev[-2])


def pi_u(l1, l2, u=1.0):
    l2p = max(l2, 0.0)
    gap = max(l1 - l2p, 0.0)
    return l1 * gap / (u - l2p + EPS), l2p / max(l1, 1e-12)


def between_mass(S, gt, oa, ob):
    a = gt == oa
    b = gt == ob
    return float(S[a][:, b].sum()), float((S[a][:, b] > 1e-8).sum())


def run(name, feat, gt, grid, bg_id):
    ids = [int(i) for i in gt.unique().tolist() if int(i) != bg_id]
    sizes = sorted([(oid, int((gt == oid).sum())) for oid in ids], key=lambda t: -t[1])
    sizes = [(o, k) for o, k in sizes if k >= 8]
    oa, ob = sizes[0][0], sizes[1][0]
    n8c, n4c = ab_touch(gt, oa, ob, grid)
    z = F.normalize(feat.float(), dim=-1)
    R8 = n8_R(z, grid)
    Sn8 = S_from_R(R8)
    graphs = [
        ("dense", _relation_graph_S(z.unsqueeze(0), EPS)[0], False),
        ("n8", Sn8, False),
        ("n8^2", Sn8 @ Sn8, False),  # full-image 2-hop; may bridge via background
        ("n8^2in", Sn8, True),  # 2-hop only inside the slot
        ("n24", S_from_R(stencil_R(z, grid, 2)), False),
    ]
    h1, _ = spatial_halves(gt, oa, grid)
    cases = [
        ("excl A", (gt == oa).float()),
        ("split A", h1),
        ("A∪B", ((gt == oa) | (gt == ob)).float()),
    ]
    print(
        f"\n======== {name}  A n={sizes[0][1]} B n={sizes[1][1]}  "
        f"AB 8-nbr contacts={n8c}  4-nbr={n4c} ========",
        flush=True,
    )
    for gname, S, _ in graphs:
        mass, nnz = between_mass(S, gt, oa, ob)
        print(f"  {gname:6s}  S_AB sum={mass:.4f}  nnz={nnz:.0f}", flush=True)
    print("  n8^2in uses induced S then S@S; AB nnz follows n8", flush=True)
    names = [g[0] for g in graphs]
    hdr = f"{'case':8s}" + "".join(f"  {g:>22s}" for g in names)
    print(hdr, flush=True)
    rows = {g: {} for g in names}
    for lab, a in cases:
        bits = [f"{lab:8s}"]
        for gname, S, sq in graphs:
            _, l1, l2 = eigs_G(S, a, square_induced=sq)
            pi, _r = pi_u(l1, l2)
            rows[gname][lab] = pi
            bits.append(f"  λ={l1:.3f}/{l2:.3f} π={pi:.3f}")
        print("".join(bits), flush=True)
    print("  ratios split/excl  and  merge/excl  (v39 π; <1 means exclusive wins)", flush=True)
    for gname in names:
        e, s, m = rows[gname]["excl A"], rows[gname]["split A"], rows[gname]["A∪B"]
        print(
            f"  {gname:6s}  split/excl={s/max(e,1e-12):.3f}  merge/excl={m/max(e,1e-12):.3f}",
            flush=True,
        )


def main():
    device = torch.device("cpu")
    for name, (cfg, min_obj, bg_id) in DATASETS.items():
        fr = collect(
            name, cfg, min_obj, bg_id, "/workspace/dataset", device, n_keep=1, max_scan=30
        )[0]
        run(name, fr["raw"], fr["gt"], fr["grid"], bg_id)


if __name__ == "__main__":
    main()
