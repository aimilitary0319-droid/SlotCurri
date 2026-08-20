"""Compare the logratio gate's confidence variants on a trained checkpoint (v29 probe).

Same methodology as calibrate_tau_g.py (v20 @ 100k, 200 val clips, slots grouped per
frame by argmax win share: ghost = 0 wins, small = 1..5%, large = > 5%), but collects
three confidence definitions side by side:

  c_ent     = 1 - H/log F over the gamma-sharpened attention   (v26, entropy)
  pur_raw   = sum A^2 / sum A over the RAW attention           (v29 proposal)
  pur_sharp = sum A~^2 / sum A~ over the sharpened attention   (variant)

and answers, for each variant:
  1. does it separate small objects from ghosts better? (conf-only AUC and the
     evidence-score AUC at beta = 0.7)
  2. what do the end-of-curriculum gates look like at tau_g in {0.25, 0.5, 1.0}?
  3. for the winning variant, how do beta_final candidates trade ghost passes
     against small-object misses? (beta sweep at tau_g = 0.5)

Usage (inside the slotcurri container):
  python event_analysis/conf_vs_purity_probe.py \
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
    ms, ents, purs, pur_sharps, wins = [], [], [], [], []
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
        b, t, s, f = att.shape
        att = att.reshape(b * t, s, f)

        att_sharp = att.pow(2.0)
        att_sharp = att_sharp / att_sharp.sum(dim=1, keepdim=True).clamp_min(1e-8)
        m = att_sharp.sum(dim=-1) / f  # (N, S)

        p_feat = att_sharp / att_sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ent = -(p_feat * p_feat.clamp_min(1e-8).log()).sum(dim=-1)
        c_ent = (1.0 - ent / math.log(f)).clamp(0.0, 1.0)  # (N, S)

        pur_raw = (att * att).sum(dim=-1) / att.sum(dim=-1).clamp_min(1e-8)  # (N, S)
        pur_sharp = (att_sharp * att_sharp).sum(dim=-1) / att_sharp.sum(dim=-1).clamp_min(1e-8)

        winner = att.argmax(dim=1)  # (N, F)
        win_counts = torch.zeros(b * t, s, device=att.device)
        win_counts.scatter_add_(1, winner, torch.ones_like(winner, dtype=win_counts.dtype))

        ms.append(m.cpu())
        ents.append(c_ent.cpu())
        purs.append(pur_raw.cpu())
        pur_sharps.append(pur_sharp.cpu())
        wins.append(win_counts.cpu())

        n_clips += b
        if n_clips >= max_clips:
            break
    return (
        torch.cat(ms).numpy(),
        torch.cat(ents).numpy(),
        torch.cat(purs).numpy(),
        torch.cat(pur_sharps).numpy(),
        torch.cat(wins).numpy(),
        n_clips,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/slotcurri/ytvis2021_attnmass_v20.yaml")
    ap.add_argument(
        "--ckpt",
        default="logs/_ytvis_attnmass_v20/checkpoints/slotcurri_step=step=100000-v1.ckpt",
    )
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--max-clips", type=int, default=200)
    ap.add_argument("--beta", type=float, default=0.7, help="v26 beta_final (late stage)")
    ap.add_argument("--p-end-mult", type=float, default=0.1)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--betas", type=float, nargs="+", default=[1.0, 0.8, 0.7, 0.6, 0.5])
    ap.add_argument("--small-frac", type=float, default=0.05)
    ap.add_argument("--cache", default="event_analysis/conf_purity_stats_v20.npz")
    ap.add_argument("--out", default="event_analysis/conf_vs_purity_v20.png")
    args = ap.parse_args()

    if args.cache and os.path.exists(args.cache):
        blob = np.load(args.cache)
        m, c_ent, pur_raw, pur_sharp, wins = (
            blob["m"], blob["c_ent"], blob["pur_raw"], blob["pur_sharp"], blob["wins"]
        )
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
        m, c_ent, pur_raw, pur_sharp, wins, n_clips = collect(
            model, loader, args.max_clips, device
        )
        if args.cache:
            np.savez_compressed(
                args.cache, m=m, c_ent=c_ent, pur_raw=pur_raw, pur_sharp=pur_sharp,
                wins=wins, n_clips=n_clips,
            )
            print(f"cached stats to {args.cache}")

    n_slots = m.shape[1]
    f_total = wins.sum(axis=1).max()
    small_hi = args.small_frac * f_total
    ghost = wins == 0
    small = (wins > 0) & (wins <= small_hi)
    large = wins > small_hi
    p = args.p_end_mult / n_slots

    variants = {"entropy": c_ent, "pur_raw": pur_raw, "pur_sharp": pur_sharp}

    print(f"clips={n_clips}  frames*slots={m.size}  patches/frame={int(f_total)}")
    print(f"group sizes: ghost={ghost.sum()}  small={small.sum()}  large={large.sum()}")
    print(f"late stage: beta={args.beta}  p={p:.5f} (p_end_mult={args.p_end_mult}/S={n_slots})")

    print("\n--- 1. confidence distributions by slot group (median [q10, q90]) ---")
    print(f"{'variant':>9s} | {'ghost':>22s} | {'small':>22s} | {'large':>22s}")
    for name, c in variants.items():
        row = f"{name:>9s} |"
        for mask in (ghost, small, large):
            v = c[mask]
            row += (f" {np.median(v):5.3f} [{np.quantile(v, .1):5.3f},"
                    f" {np.quantile(v, .9):5.3f}] |")
        print(row[:-2])

    print("\n--- 2. small-vs-ghost separability (AUC, higher is better) ---")
    lr_mass = np.log(m + EPS) - math.log(p + EPS)
    print(f"{'mass-only':>9s}  conf-only=  ---    evidence(beta={args.beta})="
          f"{auc_rank(lr_mass[small], lr_mass[ghost]):.4f}")
    for name, c in variants.items():
        auc_conf = auc_rank(c[small], c[ghost])
        log_r = args.beta * np.log(m + EPS) + (1.0 - args.beta) * np.log(c + EPS)
        lr = log_r - math.log(p + EPS)
        auc_ev = auc_rank(lr[small], lr[ghost])
        print(f"{name:>9s}  conf-only={auc_conf:.4f}  evidence(beta={args.beta})={auc_ev:.4f}")

    print("\n--- 3. end-of-curriculum gates by tau_g (at p_end, beta=%.2f) ---" % args.beta)
    header = (f"{'variant':>9s} {'tau_g':>6s} | {'g_ghost med':>11s} {'g>0.5':>6s}"
              f" | {'g_small med':>11s} {'g<0.5':>6s} | {'g_large med':>11s}")
    print(header)
    print("-" * len(header))
    for name, c in variants.items():
        log_r = args.beta * np.log(m + EPS) + (1.0 - args.beta) * np.log(c + EPS)
        lr = log_r - math.log(p + EPS)
        for tau in args.taus:
            g = 1.0 / (1.0 + np.exp(-lr / tau))
            gg, gs, gl = g[ghost], g[small], g[large]
            print(
                f"{name:>9s} {tau:6.2f} | {np.median(gg):11.3f}"
                f" {float((gg > 0.5).mean()):6.1%} | {np.median(gs):11.3f}"
                f" {float((gs < 0.5).mean()):6.1%} | {np.median(gl):11.3f}"
            )
        print("-" * len(header))

    print("\n--- 4. beta sweep for pur_raw (tau_g=0.5, at p_end) ---")
    header = (f"{'beta':>5s} | {'AUC':>6s} | {'ghost pass (r>p)':>16s} {'g_ghost med':>11s}"
              f" | {'small miss (r<p)':>16s} {'g_small med':>11s} {'g_small p10':>11s}")
    print(header)
    print("-" * len(header))
    for beta in args.betas:
        log_r = beta * np.log(m + EPS) + (1.0 - beta) * np.log(pur_raw + EPS)
        lr = log_r - math.log(p + EPS)
        g = 1.0 / (1.0 + np.exp(-lr / 0.5))
        auc = auc_rank(lr[small], lr[ghost])
        print(
            f"{beta:5.2f} | {auc:6.4f} | {float((lr[ghost] > 0).mean()):16.1%}"
            f" {np.median(g[ghost]):11.3f} | {float((lr[small] < 0).mean()):16.1%}"
            f" {np.median(g[small]):11.3f} {np.quantile(g[small], 0.10):11.3f}"
        )

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    colors = {"ghost": "tab:red", "small": "tab:green", "large": "tab:blue"}
    masks = {"ghost": ghost, "small": small, "large": large}
    for ax, (name, c) in zip(axes.flat, variants.items()):
        for gname, mask in masks.items():
            v = c[mask]
            if v.size:
                ax.hist(v, bins=80, alpha=0.5, density=True, label=gname,
                        color=colors[gname])
        ax.set_xlabel(name)
        ax.set_title(f"{name} by slot group (AUC {auc_rank(c[small], c[ghost]):.3f})")
        ax.legend()
    ax = axes[1, 1]
    for name, c, ls in (("entropy", c_ent, "-"), ("pur_raw", pur_raw, "--")):
        log_r = args.beta * np.log(m + EPS) + (1.0 - args.beta) * np.log(c + EPS)
        lr = log_r - math.log(p + EPS)
        for gname, mask in masks.items():
            v = lr[mask]
            if v.size:
                ax.hist(v, bins=100, alpha=0.35, density=True, histtype="step", ls=ls,
                        label=f"{gname} ({name})", color=colors[gname])
    ax.axvline(0.0, color="k", ls="--", lw=1)
    ax.set_xlabel(f"log(r/p)  (beta={args.beta}, p={p:.4f}; solid=entropy, dashed=pur_raw)")
    ax.set_title("evidence log-ratio by group")
    ax.legend(fontsize=7)
    fig.suptitle(f"confidence variants on {args.config.split('/')[-1]} @ {n_clips} val clips")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nsaved plot: {args.out}")


if __name__ == "__main__":
    main()
