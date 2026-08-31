"""Verify the three v38 π claims on toy cliques and real DINO+GT.

  1. exclusive object: large λ1, λ2≈0 → high π
  2. split a related object: λ1 drops → low π
  3. two communities in one slot: λ2 rises → low π

Usage (slotcurri image):
  python event_analysis/v38_pi_mechanism_check.py
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v38_featcur_mix_check import DATASETS, collect
from slotcurri.modules.video import (
    _relation_graph_S,
    _top2_algebraic_diag_s_diag,
)


def eigs(feat: torch.Tensor, att: torch.Tensor, n_iter: int = 16):
    """feat (N,D) or (1,N,D); att (S,N) or (1,S,N) -> λ1, λ2, π each (S,)."""
    if feat.ndim == 2:
        feat = feat.unsqueeze(0)
    if att.ndim == 2:
        att = att.unsqueeze(0)
    z = F.normalize(feat.float(), dim=-1)
    s_mat = _relation_graph_S(z, 1e-6)
    lam1, lam2 = _top2_algebraic_diag_s_diag(s_mat, att.float(), n_iter, 1e-6)
    pi = (lam1 - lam2.clamp_min(0.0)).clamp_min(0.0)
    return lam1[0], lam2[0], pi[0]


def eigs_full(feat: torch.Tensor, att: torch.Tensor):
    """Exact eigvalsh of G_s. feat (N,D), att (S,N)."""
    z = F.normalize(feat.float(), dim=-1)
    w = (z @ z.T).clamp_min(0.0)
    w.fill_diagonal_(0.0)
    d = w.sum(-1).clamp_min(1e-6)
    s_mat = d.rsqrt()[:, None] * w * d.rsqrt()[None, :]
    lam1, lam2, pi = [], [], []
    for s in range(att.shape[0]):
        a = att[s].float()
        g = a[:, None] * s_mat * a[None, :]
        g = 0.5 * (g + g.T)
        ev = torch.linalg.eigvalsh(g)
        l1, l2 = float(ev[-1]), float(ev[-2]) if ev.numel() > 1 else 0.0
        lam1.append(l1)
        lam2.append(l2)
        pi.append(max(l1 - max(l2, 0.0), 0.0))
    return (
        torch.tensor(lam1),
        torch.tensor(lam2),
        torch.tensor(pi),
    )


def row(name, l1, l2, pi):
    return f"{name:28s}  λ1={l1:7.3f}  λ2={l2:7.3f}  π={pi:7.3f}  π/λ1={pi/max(l1,1e-8):6.3f}"


def toy():
    print("\n======== TOY (exact eigh) ========", flush=True)
    # two orthogonal 8-cliques
    n, dim = 16, 4
    z = torch.zeros(n, dim)
    z[:8, 0] = 1.0
    z[8:, 1] = 1.0

    att_ex = torch.zeros(3, n)
    att_ex[0, :8] = 1.0
    att_ex[1, 8:] = 1.0
    att_split = torch.zeros(2, n)
    att_split[0, :4] = 1.0
    att_split[1, 4:8] = 1.0
    att_merge = torch.zeros(1, n)
    att_merge[0, :] = 1.0

    cases = [
        ("1 exclusive clique A", att_ex[:1]),
        ("1 exclusive clique B", att_ex[1:2]),
        ("2 split half of A", att_split[:1]),
        ("2 split other half of A", att_split[1:2]),
        ("3 A∪B in one slot", att_merge),
    ]
    results = {}
    for name, att in cases:
        l1, l2, pi = eigs_full(z, att)
        print(row(name, float(l1[0]), float(l2[0]), float(pi[0])), flush=True)
        results[name] = (float(l1[0]), float(l2[0]), float(pi[0]))

    l1e, l2e, pie = results["1 exclusive clique A"]
    l1s, l2s, pis = results["2 split half of A"]
    l1m, l2m, pim = results["3 A∪B in one slot"]
    print(
        f"  check 1: exclusive λ2≈0? {abs(l2e):.4f}  π high? {pie:.3f}",
        flush=True,
    )
    print(
        f"  check 2: split drops λ1? {l1e:.3f} -> {l1s:.3f}  (Δ={l1s-l1e:+.3f})  "
        f"λ2 still ~0? {l2s:.4f}  π {pie:.3f} -> {pis:.3f}",
        flush=True,
    )
    print(
        f"  check 3: merge raises λ2? {l2e:.3f} -> {l2m:.3f}  "
        f"λ1 {l1e:.3f} -> {l1m:.3f}  π {pie:.3f} -> {pim:.3f}",
        flush=True,
    )
    return results


def spatial_halves(gt: torch.Tensor, oid: int, grid: int):
    n = gt.numel()
    xs = torch.arange(n, device=gt.device) % grid
    m = gt == oid
    if int(m.sum()) < 8:
        return None, None
    med = xs[m].float().median()
    a1 = (m & (xs <= med)).float()
    a2 = (m & (xs > med)).float()
    if int(a1.sum()) < 4 or int(a2.sum()) < 4:
        idx = m.nonzero(as_tuple=False).view(-1)
        mid = idx.numel() // 2
        a1 = torch.zeros(n, device=gt.device)
        a2 = torch.zeros(n, device=gt.device)
        a1[idx[:mid]] = 1.0
        a2[idx[mid:]] = 1.0
    return a1, a2


def real_frame(name, feat, gt, grid, bg_id, tag):
    print(f"\n======== {name} {tag} N={feat.shape[0]} ========", flush=True)
    ids = [int(i) for i in gt.unique().tolist() if int(i) != bg_id]
    sizes = [(oid, int((gt == oid).sum())) for oid in ids]
    sizes.sort(key=lambda t: -t[1])
    sizes = [(oid, n) for oid, n in sizes if n >= 8]
    if len(sizes) < 2:
        print("  skip: need 2 objects with >=8 patches", flush=True)
        return None
    (oa, na), (ob, nb) = sizes[0], sizes[1]
    print(f"  objects A={oa} n={na}  B={ob} n={nb}", flush=True)

    att_ex = torch.zeros(2, feat.shape[0])
    att_ex[0] = (gt == oa).float()
    att_ex[1] = (gt == ob).float()
    h1, h2 = spatial_halves(gt, oa, grid)
    att_split = torch.stack([h1, h2], 0)
    att_merge = ((gt == oa) | (gt == ob)).float().unsqueeze(0)

    rows = []
    for lab, att in [
        (f"1 exclusive A (n={na})", att_ex[:1]),
        (f"1 exclusive B (n={nb})", att_ex[1:2]),
        ("2 split A half-1", att_split[:1]),
        ("2 split A half-2", att_split[1:2]),
        ("3 A∪B in one slot", att_merge),
    ]:
        l1, l2, pi = eigs(feat, att)
        print(row(lab, float(l1[0]), float(l2[0]), float(pi[0])), flush=True)
        rows.append((lab, float(l1[0]), float(l2[0]), float(pi[0])))

    l1a, l2a, pia = rows[0][1], rows[0][2], rows[0][3]
    l1s, l2s, pis = rows[2][1], rows[2][2], rows[2][3]
    l1m, l2m, pim = rows[4][1], rows[4][2], rows[4][3]
    ok1 = (abs(l2a) < 0.08 * max(l1a, 1e-6) or l2a < 0.05) and pia > 0.5 * l1a
    ok2_l1 = l1s < l1a - 1e-4
    ok2_pi = pis < pia - 1e-4
    ok3_l2 = l2m > l2a + 1e-4
    ok3_pi = pim < pia - 1e-4
    print(
        f"  verdict 1 exclusive λ2≪λ1: {'PASS' if ok1 else 'WEAK'}  "
        f"λ2/λ1={l2a/max(l1a,1e-8):.3f}",
        flush=True,
    )
    print(
        f"  verdict 2 split drops λ1: {'PASS' if ok2_l1 else 'FAIL'}  "
        f"and π: {'PASS' if ok2_pi else 'FAIL'}  λ1 {l1a:.3f}->{l1s:.3f}  "
        f"π {pia:.3f}->{pis:.3f}  (split λ2={l2s:.3f})",
        flush=True,
    )
    print(
        f"  verdict 3 merge raises λ2: {'PASS' if ok3_l2 else 'FAIL'}  "
        f"and π drops: {'PASS' if ok3_pi else 'FAIL'}  "
        f"λ2 {l2a:.3f}->{l2m:.3f}  λ1 {l1a:.3f}->{l1m:.3f}  π {pia:.3f}->{pim:.3f}",
        flush=True,
    )
    return {
        "exclusive": rows[0],
        "exclusive_B": rows[1],
        "split": rows[2],
        "merge": rows[4],
        "ok": (ok1, ok2_l1 and ok2_pi, ok3_l2),
    }


def draw(toy_r, real, out_path):
    labels = ["exclusive", "split half", "merge two objs"]
    fig, axes = plt.subplots(1, 1 + len(real), figsize=(4.2 * (1 + len(real)), 4.2))
    if not hasattr(axes, "__len__"):
        axes = [axes]
    packs = [("toy cliques", [
        toy_r["1 exclusive clique A"],
        toy_r["2 split half of A"],
        toy_r["3 A∪B in one slot"],
    ])]
    for name, rec in real:
        if rec is None:
            continue
        packs.append((name, [rec["exclusive"][1:], rec["split"][1:], rec["merge"][1:]]))
    for ax, (title, trip) in zip(axes, packs):
        xs = torch.arange(3)
        w = 0.25
        l1 = [t[0] for t in trip]
        l2 = [t[1] for t in trip]
        pi = [t[2] for t in trip]
        ax.bar(xs - w, l1, w, label="λ1", color="#2c7fb8")
        ax.bar(xs, l2, w, label="λ2", color="#f03b20")
        ax.bar(xs + w, pi, w, label="π=λ1−max(λ2,0)", color="#2ca25f")
        ax.set_xticks(list(xs.numpy()))
        ax.set_xticklabels(labels, rotation=15)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylabel("eigenvalue")
    fig.suptitle(
        "Photo claims: exclusive → λ2≈0 high π; split → λ1↓; merge → λ2↑",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("wrote", out_path, flush=True)


def main():
    toy_r = toy()
    data_dir = "/workspace/dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    real = []
    out_dir = "event_analysis/v38_featcur_map"
    os.makedirs(out_dir, exist_ok=True)
    for name, (cfg, min_obj, bg_id) in DATASETS.items():
        frames = collect(name, cfg, min_obj, bg_id, data_dir, device, n_keep=1, max_scan=30)
        if not frames:
            continue
        fr = frames[0]
        rec_raw = real_frame(name, fr["raw"], fr["gt"], fr["grid"], bg_id, "raw DINO (mix=1)")
        rec_rel = real_frame(name, fr["rel"], fr["gt"], fr["grid"], bg_id, "X^rel (mix=0)")
        real.append((f"{name} raw", rec_raw))
        real.append((f"{name} rel", rec_rel))
    draw(toy_r, real, os.path.join(out_dir, "v38_pi_mechanism.png"))


if __name__ == "__main__":
    main()
