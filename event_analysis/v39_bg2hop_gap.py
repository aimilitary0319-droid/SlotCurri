"""How often does full-image 8-nbr 2-hop bridge two objects via background?

Chebyshev d=1: already 8-adjacent (v39 1-hop merges them).
d=2: 1-hop disconnected; S^2 through one intermediate patch connects them.
d>=3: still disconnected after 2-hop.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "event_analysis")
from v38_featcur_mix_check import DATASETS, collect
from v38_n8_purity_probe import S_from_R, n8_R
from v39_n8_2hop_probe import eigs_G, pi_u
from slotcurri.modules.video import _relation_graph_S

EPS = 1e-6


def obj_ids(gt, bg_id, min_n=8):
    sizes = []
    for i in gt.unique().tolist():
        i = int(i)
        if i == bg_id:
            continue
        n = int((gt == i).sum())
        if n >= min_n:
            sizes.append((i, n))
    sizes.sort(key=lambda t: -t[1])
    return sizes


def min_cheb(gt, oa, ob, grid):
    a = (gt == oa).reshape(grid, grid)
    b = (gt == ob).reshape(grid, grid)
    ya, xa = torch.where(a)
    yb, xb = torch.where(b)
    # (Na,1) vs (1,Nb)
    d = torch.maximum(
        (ya[:, None] - yb[None, :]).abs(),
        (xa[:, None] - xb[None, :]).abs(),
    )
    return int(d.min())


def buckets(d):
    if d <= 1:
        return "d=1 touch"
    if d == 2:
        return "d=2 2hop-bridge"
    return "d>=3 far"


def eval_pi(z, gt, oa, ob, grid):
    R8 = n8_R(z, grid)
    Sn8 = S_from_R(R8)
    a_e = (gt == oa).float()
    a_m = ((gt == oa) | (gt == ob)).float()
    out = {}
    for name, S in (("n8", Sn8), ("n8^2", Sn8 @ Sn8), ("dense", _relation_graph_S(z.unsqueeze(0), EPS)[0])):
        _, l1e, l2e = eigs_G(S, a_e)
        _, l1m, l2m = eigs_G(S, a_m)
        pe, _ = pi_u(l1e, l2e)
        pm, _ = pi_u(l1m, l2m)
        nnz = float((S[gt == oa][:, gt == ob] > 1e-8).sum())
        out[name] = (pe, pm, pm / max(pe, 1e-12), nnz)
    return out


def main():
    device = torch.device("cpu")
    for name, (cfg, min_obj, bg_id) in DATASETS.items():
        frames = collect(
            name, cfg, min_obj, bg_id, "/workspace/dataset", device, n_keep=20, max_scan=50
        )
        counts = {"d=1 touch": 0, "d=2 2hop-bridge": 0, "d>=3 far": 0}
        d2_examples = []
        print(f"\n======== {name}  {len(frames)} clips  (two largest objects) ========", flush=True)
        for fr in frames:
            gt, grid = fr["gt"], fr["grid"]
            sizes = obj_ids(gt, bg_id)
            if len(sizes) < 2:
                continue
            oa, ob = sizes[0][0], sizes[1][0]
            d = min_cheb(gt, oa, ob, grid)
            b = buckets(d)
            counts[b] += 1
            extra = ""
            if d == 2:
                z = F.normalize(fr["raw"].float(), dim=-1)
                pis = eval_pi(z, gt, oa, ob, grid)
                extra = (
                    f"  n8 merge/excl={pis['n8'][2]:.2f} nnz={pis['n8'][3]:.0f}"
                    f"  n8^2 merge/excl={pis['n8^2'][2]:.2f} nnz={pis['n8^2'][3]:.0f}"
                    f"  dense={pis['dense'][2]:.2f}"
                )
                d2_examples.append(pis)
            print(
                f"  clip {fr['clip']:3d}  A={sizes[0][1]:3d} B={sizes[1][1]:3d}  "
                f"d={d:2d} {b}{extra}",
                flush=True,
            )
        n = sum(counts.values())
        print("  totals:", {k: f"{v}/{n}" for k, v in counts.items()}, flush=True)
        if d2_examples:
            r8 = sum(p["n8"][2] for p in d2_examples) / len(d2_examples)
            r2 = sum(p["n8^2"][2] for p in d2_examples) / len(d2_examples)
            rd = sum(p["dense"][2] for p in d2_examples) / len(d2_examples)
            print(
                f"  d=2 mean merge/excl  n8={r8:.2f}  n8^2={r2:.2f}  dense={rd:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
