"""Calibrate the log-ratio gate temperature tau_g from a trained checkpoint.

Runs the YTVIS validation split through a trained model (default: v20 @ 100k), collects
the last-iteration slot attention, and measures the empirical distributions of
  m  = gamma-sharpened attention mass            (coverage)
  c  = 1 - H/log F over the sharpened attention  (assignment confidence)
  log r = beta log(m+eps) + (1-beta) log(c+eps)  (evidence score, v26 late stage)
Slots are grouped per frame by how many patches they win (argmax over slots):
  ghost: 0 wins   |   small: 1..5% of patches   |   large: > 5%
and the gate g = sigmoid(log(r/p)/tau_g) is evaluated for candidate tau_g values against
the v26 end-of-curriculum threshold p = p_end_mult / S.

v20's late-training attention is a proxy for v26's: same backbone, losses and slot
budget, but a linear-gate curriculum, so treat absolute numbers as approximate.

Usage (inside the slotcurri container):
  python event_analysis/calibrate_tau_g.py \
      --ckpt "logs/_ytvis_attnmass_v20/checkpoints/slotcurri_step=step=100000-v1.ckpt" \
      --data-dir /workspace/dataset --max-clips 200
"""
import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from slotcurri import configuration, data, models

EPS = 1e-6


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(pos > neg). 1.0 = perfectly separable."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort().astype(np.float64) + 1.0
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


@torch.no_grad()
def collect(model, loader, max_clips, device):
    ms, cs, wins, gates_v20 = [], [], [], []
    n_clips = 0
    for batch in loader:
        if "batch_padding_mask" in batch:
            batch = model._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                continue
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model.forward(batch, train=False, cycle=model.cyclic_inference)
        att = outputs["processor"]["state_attn_mask"].float()  # (B, T, S, F)
        gate = outputs["processor"]["active_mask"].float()  # (B, T, S) v20's own gate
        b, t, s, f = att.shape
        att = att.reshape(b * t, s, f)

        att_sharp = att.pow(2.0)
        att_sharp = att_sharp / att_sharp.sum(dim=1, keepdim=True).clamp_min(1e-8)
        m = att_sharp.sum(dim=-1) / f  # (N, S)

        p_feat = att_sharp / att_sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ent = -(p_feat * p_feat.clamp_min(1e-8).log()).sum(dim=-1)
        c = (1.0 - ent / math.log(f)).clamp(0.0, 1.0)  # (N, S)

        winner = att.argmax(dim=1)  # (N, F)
        win_counts = torch.zeros(b * t, s, device=att.device)
        win_counts.scatter_add_(1, winner, torch.ones_like(winner, dtype=win_counts.dtype))

        ms.append(m.cpu())
        cs.append(c.cpu())
        wins.append(win_counts.cpu())
        gates_v20.append(gate.reshape(b * t, s).cpu())

        n_clips += b
        if n_clips >= max_clips:
            break
    return (
        torch.cat(ms).numpy(),
        torch.cat(cs).numpy(),
        torch.cat(wins).numpy(),
        torch.cat(gates_v20).numpy(),
        n_clips,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/slotcurri/ytvis2021_attnmass_v20.yaml")
    ap.add_argument(
        "--ckpt",
        default="logs/_ytvis_attnmass_v20/checkpoints/slotcurri_step=step=100000-v1.ckpt",
        help="checkpoint to analyze (unused when --cache exists)",
    )
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--max-clips", type=int, default=200)
    ap.add_argument("--beta", type=float, default=0.7, help="v26 beta_final (late stage)")
    ap.add_argument("--p-end-mult", type=float, default=0.1, help="v26 p_end multiplier")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--small-frac", type=float, default=0.05,
                    help="win-share boundary between small and large slots")
    ap.add_argument("--p-sweep", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.3],
                    help="p_end_mult candidates for the threshold trade-off table")
    ap.add_argument("--cache", default="event_analysis/tau_g_stats_v20.npz",
                    help="npz cache of collected (m, c, wins); reused if it exists")
    ap.add_argument("--out", default="event_analysis/tau_g_calibration_v20.png")
    args = ap.parse_args()

    if args.cache and os.path.exists(args.cache):
        blob = np.load(args.cache)
        m, c, wins = blob["m"], blob["c"], blob["wins"]
        n_clips = int(blob["n_clips"])
        print(f"loaded cached stats from {args.cache}")
    else:
        config = configuration.load_config(args.config)
        dataset = data.build(config.dataset, data_dir=args.data_dir)
        model = models.build(config.model, config.optimizer, None, None)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)

        dataset.setup("validate")
        loader = dataset.val_dataloader()
        m, c, wins, gate_v20, n_clips = collect(model, loader, args.max_clips, device)
        if args.cache:
            np.savez_compressed(args.cache, m=m, c=c, wins=wins, n_clips=n_clips)
            print(f"cached stats to {args.cache}")
    n_slots = m.shape[1]

    # groups per (frame, slot)
    f_total = wins.sum(axis=1).max()  # patches per frame
    small_hi = args.small_frac * f_total
    ghost = wins == 0
    small = (wins > 0) & (wins <= small_hi)
    large = wins > small_hi

    p = args.p_end_mult / n_slots
    log_r = args.beta * np.log(m + EPS) + (1.0 - args.beta) * np.log(c + EPS)
    log_ratio = log_r - math.log(p + EPS)
    log_ratio_mass = np.log(m + EPS) - math.log(p + EPS)  # beta = 1 baseline

    def grp(x, mask):
        return x[mask]

    print(f"clips={n_clips}  frames*slots={m.size}  patches/frame={int(f_total)}")
    print(f"group sizes: ghost={ghost.sum()}  small={small.sum()}  large={large.sum()}")
    print(f"v26 late stage: beta={args.beta}  p={p:.5f} (p_end_mult={args.p_end_mult}/S={n_slots})")
    print()
    for name, mask in (("ghost", ghost), ("small", small), ("large", large)):
        mm, cc = grp(m, mask), grp(c, mask)
        if mm.size == 0:
            continue
        print(
            f"{name:6s} m: med={np.median(mm):.4f} [{np.quantile(mm, 0.1):.4f},"
            f" {np.quantile(mm, 0.9):.4f}]   c: med={np.median(cc):.3f}"
            f" [{np.quantile(cc, 0.1):.3f}, {np.quantile(cc, 0.9):.3f}]"
        )
    print()
    auc_mass = auc_rank(grp(log_ratio_mass, small), grp(log_ratio_mass, ghost))
    auc_evid = auc_rank(grp(log_ratio, small), grp(log_ratio, ghost))
    print(f"small-vs-ghost separability AUC: mass-only={auc_mass:.4f}  evidence(beta={args.beta})={auc_evid:.4f}")
    print()
    header = f"{'tau_g':>6s} | {'g_ghost med':>11s} {'g>0.5':>6s} | {'g_small med':>11s} {'g<0.5':>6s} | {'g_large med':>11s}"
    print(header)
    print("-" * len(header))
    for tau in args.taus:
        g = 1.0 / (1.0 + np.exp(-log_ratio / tau))
        gg, gs, gl = grp(g, ghost), grp(g, small), grp(g, large)
        print(
            f"{tau:6.2f} | {np.median(gg):11.3f} {float((gg > 0.5).mean()):6.1%}"
            f" | {np.median(gs):11.3f} {float((gs < 0.5).mean()):6.1%}"
            f" | {np.median(gl):11.3f}"
        )

    # ---- p_end_mult trade-off sweep (tau fixed to the first candidate) ----
    tau0 = args.taus[0]
    print()
    print(f"p_end_mult sweep at beta={args.beta}, tau_g={tau0}:")
    header2 = (
        f"{'p_mult':>6s} {'p':>8s} | {'ghost pass (r>p)':>16s} {'g_ghost med':>11s}"
        f" | {'small miss (r<p)':>16s} {'g_small med':>11s} {'g_small p10':>11s}"
        f" | {'g_large med':>11s}"
    )
    print(header2)
    print("-" * len(header2))
    for pm in args.p_sweep:
        p_i = pm / n_slots
        lr_i = log_r - math.log(p_i + EPS)
        g_i = 1.0 / (1.0 + np.exp(-lr_i / tau0))
        gg, gs, gl = grp(g_i, ghost), grp(g_i, small), grp(g_i, large)
        print(
            f"{pm:6.2f} {p_i:8.4f} | {float((grp(lr_i, ghost) > 0).mean()):16.1%}"
            f" {np.median(gg):11.3f} | {float((grp(lr_i, small) < 0).mean()):16.1%}"
            f" {np.median(gs):11.3f} {np.quantile(gs, 0.10):11.3f}"
            f" | {np.median(gl):11.3f}"
        )

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    colors = {"ghost": "tab:red", "small": "tab:green", "large": "tab:blue"}
    masks = {"ghost": ghost, "small": small, "large": large}

    ax = axes[0, 0]
    for name, mask in masks.items():
        vals = np.log10(grp(m, mask) + EPS)
        if vals.size:
            ax.hist(vals, bins=80, alpha=0.5, density=True, label=name, color=colors[name])
    ax.axvline(np.log10(p), color="k", ls="--", label=f"p_end={p:.4f}")
    ax.set_xlabel("log10 m (gamma-sharpened mass)")
    ax.set_title("coverage m by slot group")
    ax.legend()

    ax = axes[0, 1]
    for name, mask in masks.items():
        vals = grp(c, mask)
        if vals.size:
            ax.hist(vals, bins=80, alpha=0.5, density=True, label=name, color=colors[name])
    ax.set_xlabel("c (assignment confidence)")
    ax.set_title("confidence c by slot group")
    ax.legend()

    ax = axes[1, 0]
    idx = np.random.default_rng(0).permutation(m.size)[:20000]
    flat_m, flat_c = m.ravel()[idx], c.ravel()[idx]
    flat_grp = (small.astype(int) + 2 * large.astype(int)).ravel()[idx]
    for gi, name in ((0, "ghost"), (1, "small"), (2, "large")):
        sel = flat_grp == gi
        ax.scatter(np.log10(flat_m[sel] + EPS), flat_c[sel], s=2, alpha=0.3,
                   color=colors[name], label=name)
    ax.axvline(np.log10(p), color="k", ls="--")
    ax.set_xlabel("log10 m")
    ax.set_ylabel("c")
    ax.set_title("joint (m, c): small active vs ghost")
    ax.legend(markerscale=4)

    ax = axes[1, 1]
    for name, mask in masks.items():
        vals = grp(log_ratio, mask)
        if vals.size:
            ax.hist(vals, bins=100, alpha=0.5, density=True, label=name, color=colors[name])
    ax.axvline(0.0, color="k", ls="--", label="r = p (g = 0.5)")
    for tau, ls in zip(args.taus, (":", "-.", "--")):
        ax.axvspan(-2 * tau, 2 * tau, alpha=0.06, color="gray")
        ax.axvline(2 * tau, color="gray", ls=ls, lw=1, label=f"g=0.88 @ tau={tau}")
    ax.set_xlabel(f"log(r / p)   (beta={args.beta}, p={p:.4f})")
    ax.set_title("evidence log-ratio by group; gray bands = sigmoid transition (|logit|<2)")
    ax.legend()

    fig.suptitle(
        f"tau_g calibration from {args.config.split('/')[-1]} @ {n_clips} val clips "
        f"(AUC small-vs-ghost: mass {auc_mass:.3f} -> evidence {auc_evid:.3f})"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nsaved plot: {args.out}")


if __name__ == "__main__":
    main()
