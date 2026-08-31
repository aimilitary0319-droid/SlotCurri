"""Can any f(λ1, λ2, ...) of G_s = diag(a) S diag(a) rank exclusive > merge?"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "event_analysis")
from v38_featcur_mix_check import DATASETS, collect
from v38_pi_mechanism_check import spatial_halves
from slotcurri.modules.video import _relation_graph_S


def topk_G(S, a, k=6):
    m = a > 0.5
    n = int(m.sum())
    if n < k + 1:
        k = max(n - 1, 1)
    G = S[m][:, m]
    G = 0.5 * (G + G.T)
    ev = torch.linalg.eigvalsh(G)
    lam = ev.flip(0)[:k].clamp_min(-1e-8)
    return n, lam


def show(title, S, cases):
    print(f"\n==== {title} ====", flush=True)
    hdr = f"{'case':22s} n  " + " ".join(f"λ{i+1:>7}" for i in range(4))
    hdr += f"  {'λ2/λ1':>7} {'λ3/λ1':>7} {'gap/λ1':>7} {'λ2-λ3':>7}"
    print(hdr, flush=True)
    rows = {}
    for name, a in cases:
        n, lam = topk_G(S, a, 6)
        l = [float(x) for x in lam]
        while len(l) < 4:
            l.append(0.0)
        r21 = l[1] / max(l[0], 1e-8)
        r31 = l[2] / max(l[0], 1e-8)
        gap = (l[0] - max(l[1], 0.0)) / max(l[0], 1e-8)
        d23 = l[1] - l[2]
        print(
            f"{name:22s} {n:3d} "
            f"{l[0]:7.3f} {l[1]:7.3f} {l[2]:7.3f} {l[3]:7.3f}  "
            f"{r21:7.3f} {r31:7.3f} {gap:7.3f} {d23:7.3f}",
            flush=True,
        )
        rows[name] = l
    return rows


def run_frame(name, feat, gt, grid, bg_id, tag):
    ids = [int(i) for i in gt.unique().tolist() if int(i) != bg_id]
    sizes = sorted([(oid, int((gt == oid).sum())) for oid in ids], key=lambda t: -t[1])
    sizes = [(o, n) for o, n in sizes if n >= 8]
    if len(sizes) < 2:
        return
    oa, ob = sizes[0][0], sizes[1][0]
    S = _relation_graph_S(F.normalize(feat.float(), dim=-1).unsqueeze(0), 1e-6)[0]
    h1, _ = spatial_halves(gt, oa, grid)
    show(f"{name} {tag}", S, [
        ("exclusive A", (gt == oa).float()),
        ("exclusive B", (gt == ob).float()),
        ("split A", h1),
        ("A∪B", ((gt == oa) | (gt == ob)).float()),
    ])


def main():
    # toy: two orthogonal cliques — λ2 of merge should match λ1
    z = torch.zeros(16, 4)
    z[:8, 0] = 1.0
    z[8:, 1] = 1.0
    S = _relation_graph_S(F.normalize(z, dim=-1).unsqueeze(0), 1e-6)[0]
    aA = torch.zeros(16); aA[:8] = 1
    aM = torch.ones(16)
    show("TOY", S, [("exclusive A", aA), ("A∪B", aM)])

    device = torch.device("cpu")
    for name, (cfg, min_obj, bg_id) in DATASETS.items():
        fr = collect(name, cfg, min_obj, bg_id, "/workspace/dataset", device, n_keep=1, max_scan=30)[0]
        run_frame(name, fr["raw"], fr["gt"], fr["grid"], bg_id, "raw")


if __name__ == "__main__":
    main()
