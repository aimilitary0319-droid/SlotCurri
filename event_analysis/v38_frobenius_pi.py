"""π = λ1^2 / ||G||_F^2  (and /||G||_F) on dense S vs 8-neighbor S."""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "event_analysis")
from v38_featcur_mix_check import DATASETS, collect
from v38_n8_purity_probe import S_from_R, n8_R
from v38_pi_mechanism_check import spatial_halves as spatial_halves2
from slotcurri.modules.video import _relation_graph_S


def stats(S, a):
    m = a > 0.5
    n = int(m.sum())
    if n < 2:
        return n, float("nan"), float("nan"), float("nan"), float("nan")
    G = 0.5 * (S[m][:, m] + S[m][:, m].T)
    ev = torch.linalg.eigvalsh(G)
    l1 = float(ev[-1])
    l2 = float(ev[-2])
    f2 = float((G * G).sum())
    f = f2 ** 0.5
    return n, l1, l2, (l1 * l1) / max(f2, 1e-12), (l1 * l1) / max(f, 1e-12)


def run(name, feat, gt, grid, bg_id):
    ids = [int(i) for i in gt.unique().tolist() if int(i) != bg_id]
    sizes = sorted([(oid, int((gt == oid).sum())) for oid in ids], key=lambda t: -t[1])
    sizes = [(o, n) for o, n in sizes if n >= 8]
    oa, ob = sizes[0][0], sizes[1][0]
    z = F.normalize(feat.float(), dim=-1)
    S_d = _relation_graph_S(z.unsqueeze(0), 1e-6)[0]
    S_n = S_from_R(n8_R(z, grid))
    h1, _ = spatial_halves2(gt, oa, grid)
    cases = [
        ("exclusive A", (gt == oa).float()),
        ("split A", h1),
        ("A∪B", ((gt == oa) | (gt == ob)).float()),
    ]
    print(f"\n==== {name} A n={sizes[0][1]} B n={sizes[1][1]} ====", flush=True)
    print(
        f"{'case':14s}  {'dense λ1²/||G||F²':>16s} {'n8 λ1²/||G||F²':>14s}  "
        f"{'dense λ1-λ2':>12s} {'n8 λ1-λ2':>10s}",
        flush=True,
    )
    for lab, a in cases:
        nd, l1d, l2d, e_d, _ = stats(S_d, a)
        nn, l1n, l2n, e_n, _ = stats(S_n, a)
        print(
            f"{lab:14s}  {e_d:16.3f} {e_n:14.3f}  "
            f"{max(l1d-max(l2d,0),0):12.3f} {max(l1n-max(l2n,0),0):10.3f}",
            flush=True,
        )


def main():
    z = torch.zeros(16, 4)
    z[:8, 0] = 1.0
    z[8:, 1] = 1.0
    S = _relation_graph_S(F.normalize(z, dim=-1).unsqueeze(0), 1e-6)[0]
    aA = torch.zeros(16); aA[:8] = 1
    aM = torch.ones(16)
    print("TOY dense-clique  exclusive λ1²/F2=%.3f  merge=%.3f" % (
        stats(S, aA)[3], stats(S, aM)[3]), flush=True)
    device = torch.device("cpu")
    for name, (cfg, min_obj, bg_id) in DATASETS.items():
        fr = collect(name, cfg, min_obj, bg_id, "/workspace/dataset", device, n_keep=1, max_scan=30)[0]
        run(name, fr["raw"], fr["gt"], fr["grid"], bg_id)


if __name__ == "__main__":
    main()
