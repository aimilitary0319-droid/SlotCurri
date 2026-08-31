"""Intra vs inter ReLU-cosine / S for the two largest GT objects."""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "event_analysis")
from v38_featcur_mix_check import DATASETS, collect
from slotcurri.modules.video import _relation_graph_S


def intra(M, m):
    n = int(m.sum())
    return float(M[m][:, m].sum()) / max(n * (n - 1), 1)


def inter(M, m1, m2):
    return float(M[m1][:, m2].mean())


def block_stats(feat, gt, oa, ob):
    z = F.normalize(feat.float(), dim=-1)
    R = (z @ z.T).clamp_min(0.0)
    R.fill_diagonal_(0.0)
    S = _relation_graph_S(z.unsqueeze(0), 1e-6)[0]
    a, b = gt == oa, gt == ob
    r_aa, r_bb, r_ab = intra(R, a), intra(R, b), inter(R, a, b)
    s_aa, s_bb, s_ab = intra(S, a), intra(S, b), inter(S, a, b)
    ma = F.normalize(z[a].mean(0), dim=0)
    mb = F.normalize(z[b].mean(0), dim=0)
    return {
        "mean_cos_AB": float(ma @ mb),
        "R_AA": r_aa,
        "R_BB": r_bb,
        "R_AB": r_ab,
        "R_ratio": r_ab / max(0.5 * (r_aa + r_bb), 1e-8),
        "S_ratio": s_ab / max(0.5 * (s_aa + s_bb), 1e-8),
        "frac_pos_AB": float((R[a][:, b] > 1e-6).float().mean()),
    }


def main():
    device = torch.device("cpu")
    for name, (cfg, min_obj, bg_id) in DATASETS.items():
        frames = collect(name, cfg, min_obj, bg_id, "/workspace/dataset", device, n_keep=1, max_scan=30)
        fr = frames[0]
        gt = fr["gt"]
        ids = [int(i) for i in gt.unique().tolist() if int(i) != bg_id]
        sizes = sorted([(oid, int((gt == oid).sum())) for oid in ids], key=lambda t: -t[1])
        oa, ob = sizes[0][0], sizes[1][0]
        print(f"\n#### {name} A n={sizes[0][1]} B n={sizes[1][1]}", flush=True)
        for tag, feat in [("raw mix=1", fr["raw"]), ("rel mix=0", fr["rel"])]:
            st = block_stats(feat, gt, oa, ob)
            print(
                f"  {tag:12s}  centroid-cos AB={st['mean_cos_AB']:.3f}  "
                f"R AA/BB/AB={st['R_AA']:.3f}/{st['R_BB']:.3f}/{st['R_AB']:.3f}  "
                f"R_AB/intra={st['R_ratio']:.3f}  S_AB/intra={st['S_ratio']:.3f}  "
                f"frac R>0 on AB={st['frac_pos_AB']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
