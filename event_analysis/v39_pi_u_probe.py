"""Empirically check v39 π before implementing.

  G = diag(a) S diag(a)   S = 8-neighbor (and dense, as control)
  u = (max a)^2
  π_u = λ1 (λ1 - λ2+) / (u - λ2+ + eps)
  π_1 = same with 1 in the denominator  (the formula that needs hard a)
  π_gap = λ1 - λ2+                     (v38)

Uses the training top-2 solver on full-N softmax a (no harden).
"""
from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "event_analysis")
from v38_featcur_mix_check import DATASETS, collect
from v38_n8_purity_probe import S_from_R, n8_R
from v38_pi_mechanism_check import spatial_halves
from slotcurri.modules.video import (
    _relation_graph_S,
    _top2_algebraic_diag_s_diag,
)

EPS = 1e-6


def lam12(S, a):
    """a: (N,), S: (N,N) -> λ1, λ2 via v38 solver."""
    l1, l2 = _top2_algebraic_diag_s_diag(
        S.unsqueeze(0), a.view(1, 1, -1), n_iter=16, eps=EPS
    )
    return float(l1[0, 0]), float(l2[0, 0])


def pi_pack(l1, l2, u, eps=EPS):
    l2p = max(l2, 0.0)
    gap = max(l1 - l2p, 0.0)
    pi_u = l1 * gap / (u - l2p + eps)
    pi_1 = l1 * gap / (1.0 - l2p + eps)
    return {
        "l1": l1,
        "l2": l2,
        "l2p": l2p,
        "u": u,
        "gap": gap,
        "pi_u": pi_u,
        "pi_1": pi_1,
        "pi_gap": gap,
        "l1_le_u": l1 <= u + 1e-5,
        "r21": l2p / max(l1, 1e-12),
    }


def make_a(n, owned, peak, n_slots):
    """Synthetic last-iter mask: peak on owned patches, 1/K elsewhere."""
    a = torch.full((n,), 1.0 / n_slots)
    a[owned] = float(peak)
    return a


def fmt(p):
    return (
        f"λ1={p['l1']:.3f} λ2={p['l2']:.3f} u={p['u']:.3f} "
        f"λ1≤u={str(p['l1_le_u']):5s} r={p['r21']:.3f}  "
        f"π_u={p['pi_u']:.4f} π_1={p['pi_1']:.4f} π_gap={p['pi_gap']:.4f}"
    )


def eval_cases(S, gt, oa, ob, grid, n_slots, peaks, tag):
    n = gt.numel()
    owned_a = gt == oa
    owned_b = gt == ob
    owned_m = owned_a | owned_b
    h1, _ = spatial_halves(gt, oa, grid)
    specs = [
        ("exclusive A", owned_a),
        ("exclusive B", owned_b),
        ("split A", h1 > 0.5),
        ("A∪B", owned_m),
    ]
    print(f"\n---- {tag} ----", flush=True)
    for peak in peaks:
        print(f"  peak a={peak:.3f}  (1/K={1.0/n_slots:.3f})", flush=True)
        rows = {}
        for name, owned in specs:
            if peak >= 0.999:
                a = owned.float()
            else:
                a = make_a(n, owned, peak, n_slots)
            u = float(a.max() ** 2)
            l1, l2 = lam12(S, a)
            p = pi_pack(l1, l2, u)
            rows[name] = p
            print(f"    {name:14s} {fmt(p)}", flush=True)
        ea, mer = rows["exclusive A"]["pi_u"], rows["A∪B"]["pi_u"]
        e1, m1 = rows["exclusive A"]["pi_1"], rows["A∪B"]["pi_1"]
        ok_u = mer < ea - 1e-6
        ok_1 = m1 < e1 - 1e-6
        ratio_u = mer / max(ea, 1e-12)
        print(
            f"    rank π_u merge/excl={ratio_u:.3f}  "
            f"{'PASS' if ok_u else 'FAIL (merge not lower)'}   "
            f"π_1 merge<excl: {'PASS' if ok_1 else 'FAIL'}",
            flush=True,
        )


def main():
    device = torch.device("cpu")
    for name, (cfg, min_obj, bg_id) in DATASETS.items():
        n_slots = 7 if name == "ytvis" else 11
        fr = collect(
            name, cfg, min_obj, bg_id, "/workspace/dataset", device, n_keep=1, max_scan=30
        )[0]
        gt, grid = fr["gt"], fr["grid"]
        z = F.normalize(fr["raw"].float(), dim=-1)
        ids = [int(i) for i in gt.unique().tolist() if int(i) != bg_id]
        sizes = sorted(
            [(oid, int((gt == oid).sum())) for oid in ids], key=lambda t: -t[1]
        )
        sizes = [(o, k) for o, k in sizes if k >= 8]
        oa, ob = sizes[0][0], sizes[1][0]
        print(
            f"\n======== {name} raw  A n={sizes[0][1]} B n={sizes[1][1]}  "
            f"K={n_slots} N={gt.numel()} ========",
            flush=True,
        )
        S_n8 = S_from_R(n8_R(z, grid))
        S_den = _relation_graph_S(z.unsqueeze(0), EPS)[0]
        peaks = [1.0, 0.85, 0.55, 1.0 / n_slots]
        eval_cases(S_n8, gt, oa, ob, grid, n_slots, peaks, f"{name} 8-neighbor S")
        eval_cases(
            S_den, gt, oa, ob, grid, n_slots, [1.0, 0.85], f"{name} dense S (control)"
        )


if __name__ == "__main__":
    main()
