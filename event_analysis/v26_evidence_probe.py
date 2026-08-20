"""Probe how much mass (m) and confidence (c) each contribute to the v26 gate.

Loads a mid-training v26 checkpoint, collects per-slot (m, c, wins) on val clips
(same recipe as calibrate_tau_g.py), and answers:
  1. do m and c carry independent information (corr, group medians)?
  2. how different are the gates with vs without the confidence term, at the
     checkpoint's own training stage AND at the eval stage (lambda = 1)?
  3. what share of the within-frame score variance comes from the c-term?
"""
import argparse
import math
import os

import numpy as np
import torch

from slotcurri import configuration, data, models

EPS = 1e-6


@torch.no_grad()
def collect(model, loader, max_clips, device):
    ms, cs, wins = [], [], []
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
        m = att_sharp.sum(dim=-1) / f

        p_feat = att_sharp / att_sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ent = -(p_feat * p_feat.clamp_min(1e-8).log()).sum(dim=-1)
        c = (1.0 - ent / math.log(f)).clamp(0.0, 1.0)

        winner = att.argmax(dim=1)
        win_counts = torch.zeros(b * t, s, device=att.device)
        win_counts.scatter_add_(1, winner, torch.ones_like(winner, dtype=win_counts.dtype))

        ms.append(m.cpu())
        cs.append(c.cpu())
        wins.append(win_counts.cpu())
        n_clips += b
        if n_clips >= max_clips:
            break
    return torch.cat(ms).numpy(), torch.cat(cs).numpy(), torch.cat(wins).numpy(), n_clips


def gate(log_r, p, tau):
    return 1.0 / (1.0 + np.exp(-(log_r - math.log(p + EPS)) / tau))


def stage_report(name, m, c, groups, beta, p_mult, tau, n_slots):
    p = p_mult / n_slots
    log_m = np.log(m + EPS)
    log_c = np.log(c + EPS)
    log_r = beta * log_m + (1.0 - beta) * log_c
    g_ev = gate(log_r, p, tau)
    g_mo = gate(log_m, p, tau)  # mass-only counterpart (beta = 1)
    dg = g_ev - g_mo

    print(f"--- {name}: beta={beta:.3f} p_mult={p_mult:.3f} tau={tau} ---")
    print(f"mean |g_evidence - g_massonly| = {np.abs(dg).mean():.4f}   "
          f"frac |dg|>0.1 = {(np.abs(dg) > 0.1).mean():.1%}   "
          f"frac decision flips (0.5) = {((g_ev > 0.5) != (g_mo > 0.5)).mean():.1%}")
    for gname, mask in groups.items():
        if mask.sum() == 0:
            continue
        print(f"  {gname:6s} g_evidence med={np.median(g_ev[mask]):.3f}  "
              f"g_massonly med={np.median(g_mo[mask]):.3f}  "
              f"dg med={np.median(dg[mask]):+.3f}")
    # within-frame variance decomposition of the score's slot-to-slot differences
    sm = beta * log_m
    sc = (1.0 - beta) * log_c
    vm = (sm - sm.mean(axis=1, keepdims=True)).var(axis=1).mean()
    vc = (sc - sc.mean(axis=1, keepdims=True)).var(axis=1).mean()
    cov = ((sm - sm.mean(axis=1, keepdims=True)) * (sc - sc.mean(axis=1, keepdims=True))).mean()
    vt = (log_r * beta / beta - log_r.mean(axis=1, keepdims=True)).var(axis=1).mean()
    print(f"  within-frame score variance: mass-term={vm:.4f}  conf-term={vc:.4f}  "
          f"2cov={2 * cov:.4f}  total={vt:.4f}  (conf share ~ {(vc + cov) / max(vt, 1e-9):.1%})")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/slotcurri/ytvis2021_attnmass_v26.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--step", type=int, required=True, help="training step of the ckpt")
    ap.add_argument("--anneal-steps", type=int, default=75000)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--max-clips", type=int, default=60)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--beta-final", type=float, default=0.7)
    ap.add_argument("--cache", default="event_analysis/v26_evidence_stats.npz")
    args = ap.parse_args()

    if args.cache and os.path.exists(args.cache):
        blob = np.load(args.cache)
        m, c, wins = blob["m"], blob["c"], blob["wins"]
        n_clips = int(blob["n_clips"])
        print(f"loaded cache {args.cache}")
    else:
        config = configuration.load_config(args.config)
        dataset = data.build(config.dataset, data_dir=args.data_dir)
        model = models.build(config.model, config.optimizer, None, None)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)
        dataset.setup("validate")
        m, c, wins, n_clips = collect(model, dataset.val_dataloader(), args.max_clips, device)
        if args.cache:
            np.savez_compressed(args.cache, m=m, c=c, wins=wins, n_clips=n_clips)

    n_slots = m.shape[1]
    f_total = wins.sum(axis=1).max()
    groups = {
        "ghost": wins == 0,
        "small": (wins > 0) & (wins <= 0.05 * f_total),
        "large": wins > 0.05 * f_total,
    }

    print(f"clips={n_clips}  frames={m.shape[0]}  slots={n_slots}")
    for gname, mask in groups.items():
        if mask.sum() == 0:
            print(f"{gname:6s} n=0")
            continue
        mm, cc = m[mask], c[mask]
        print(f"{gname:6s} n={mask.sum():6d} ({mask.mean():5.1%})  "
              f"m med={np.median(mm):.4f} [{np.quantile(mm, .1):.4f},{np.quantile(mm, .9):.4f}]  "
              f"c med={np.median(cc):.3f} [{np.quantile(cc, .1):.3f},{np.quantile(cc, .9):.3f}]")
    lm, lc = np.log(m + EPS).ravel(), np.log(c + EPS).ravel()
    print(f"corr(log m, log c) = {np.corrcoef(lm, lc)[0, 1]:.3f}")
    print()

    lam = 0.5 * (1.0 - math.cos(math.pi * min(args.step / args.anneal_steps, 1.0)))
    beta_now = 1.0 - lam * (1.0 - args.beta_final)
    p_now = 1.5 + (0.1 - 1.5) * lam
    stage_report(f"train stage @ step {args.step} (lambda={lam:.3f})",
                 m, c, groups, beta_now, p_now, args.tau, n_slots)
    stage_report("eval stage (lambda=1)", m, c, groups, args.beta_final, 0.1, args.tau, n_slots)


if __name__ == "__main__":
    main()
