"""Fiedler μ2 of L_s = D_s - W_s, W_s = diag(a) R diag(a).

Compares exclusive / split / merge on toy cliques and real DINO+GT.
Laplacian is formed on the support {i: a_i > 0} so isolated zeros do not
force μ2=0 (softmax-on-all-N would).
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "event_analysis")
from v38_featcur_mix_check import DATASETS, collect
from v38_pi_mechanism_check import spatial_halves


def relu_R(feat):
    z = F.normalize(feat.float(), dim=-1)
    R = (z @ z.T).clamp_min(0.0)
    R.fill_diagonal_(0.0)
    return R


def fiedler(R, a, eps=1e-8):
    """μ2 of unnormalized L and of symmetric-normalized L, on support of a."""
    m = a > 1e-6
    n = int(m.sum())
    if n < 2:
        return float("nan"), float("nan"), n
    aa = a[m]
    W = aa[:, None] * R[m][:, m] * aa[None, :]
    d = W.sum(-1)
    L = torch.diag(d) - W
    ev = torch.linalg.eigvalsh(0.5 * (L + L.T))
    mu2 = float(ev[1].clamp_min(0.0))
    d_inv = d.clamp_min(eps).rsqrt()
    Ln = d_inv[:, None] * L * d_inv[None, :]
    evn = torch.linalg.eigvalsh(0.5 * (Ln + Ln.T))
    mu2n = float(evn[1].clamp_min(0.0))
    return mu2, mu2n, n


def report(title, R, cases):
    print(f"\n==== {title} ====", flush=True)
    print(f"{'case':28s}  n  {'μ2(L)':>10s}  {'μ2(norm L)':>10s}", flush=True)
    out = {}
    for name, a in cases:
        mu2, mu2n, n = fiedler(R, a)
        print(f"{name:28s} {n:3d}  {mu2:10.4f}  {mu2n:10.4f}", flush=True)
        out[name] = (mu2, mu2n, n)
    return out


def toy():
    n, dim = 16, 4
    z = torch.zeros(n, dim)
    z[:8, 0] = 1.0
    z[8:, 1] = 1.0
    R = relu_R(z)
    aA = torch.zeros(n); aA[:8] = 1
    aB = torch.zeros(n); aB[8:] = 1
    aS = torch.zeros(n); aS[:4] = 1
    aM = torch.ones(n)
    return report("TOY orthogonal cliques", R, [
        ("exclusive A", aA),
        ("split half A", aS),
        ("A∪B", aM),
    ])


def real_one(name, feat, gt, grid, bg_id, tag):
    ids = [int(i) for i in gt.unique().tolist() if int(i) != bg_id]
    sizes = sorted([(oid, int((gt == oid).sum())) for oid in ids], key=lambda t: -t[1])
    sizes = [(o, n) for o, n in sizes if n >= 8]
    if len(sizes) < 2:
        return
    oa, ob = sizes[0][0], sizes[1][0]
    R = relu_R(feat)
    h1, h2 = spatial_halves(gt, oa, grid)
    report(f"{name} {tag} A n={sizes[0][1]} B n={sizes[1][1]}", R, [
        (f"exclusive A n={sizes[0][1]}", (gt == oa).float()),
        (f"exclusive B n={sizes[1][1]}", (gt == ob).float()),
        ("split A half", h1),
        ("A∪B", ((gt == oa) | (gt == ob)).float()),
    ])


def main():
    toy()
    device = torch.device("cpu")
    for name, (cfg, min_obj, bg_id) in DATASETS.items():
        frames = collect(name, cfg, min_obj, bg_id, "/workspace/dataset", device, n_keep=1, max_scan=30)
        fr = frames[0]
        real_one(name, fr["raw"], fr["gt"], fr["grid"], bg_id, "raw mix=1")
        real_one(name, fr["rel"], fr["gt"], fr["grid"], bg_id, "rel mix=0")


if __name__ == "__main__":
    main()
