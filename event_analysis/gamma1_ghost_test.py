"""Does confidence alone suppress ghosts when gamma=1 (no mass sharpening)?

Collects RAW-attention (gamma=1) per-slot mass and confidence from the v20 @ 100k
checkpoint (ghost-rich, late-trained) and evaluates the end-of-curriculum gate
  log r = beta log(m) + (1-beta) log(c),  g = sigmoid((log r - log p)/tau)
for the gamma=1 candidate (tau=0.25) against the current gamma=2 setting (tau=0.5).
"""
import argparse
import math
import os

import numpy as np
import torch

from slotcurri import configuration, data, models

EPS = 1e-6


@torch.no_grad()
def collect(model, loader, max_clips, device, gamma):
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
        att = outputs["processor"]["state_attn_mask"].float()
        b, t, s, f = att.shape
        att = att.reshape(b * t, s, f)

        if gamma != 1.0:
            att_g = att.pow(gamma)
            att_g = att_g / att_g.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            att_g = att
        m = att_g.sum(dim=-1) / f
        p_feat = att_g / att_g.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ent = -(p_feat * p_feat.clamp_min(1e-8).log()).sum(dim=-1)
        c = (1.0 - ent / math.log(f)).clamp(0.0, 1.0)

        winner = att.argmax(dim=1)  # winner defined on raw attention either way
        win_counts = torch.zeros(b * t, s, device=att.device)
        win_counts.scatter_add_(1, winner, torch.ones_like(winner, dtype=win_counts.dtype))

        ms.append(m.cpu()); cs.append(c.cpu()); wins.append(win_counts.cpu())
        n_clips += b
        if n_clips >= max_clips:
            break
    return torch.cat(ms).numpy(), torch.cat(cs).numpy(), torch.cat(wins).numpy(), n_clips


def report(tag, m, c, wins, beta, tau, p_mult):
    S = m.shape[1]
    F = wins.sum(axis=1).max()
    ghost, small = wins == 0, (wins > 0) & (wins <= 0.05 * F)
    large = wins > 0.05 * F
    p = p_mult / S
    lr = beta * np.log(m + EPS) + (1 - beta) * np.log(c + EPS)
    g = 1.0 / (1.0 + np.exp(-(lr - math.log(p + EPS)) / tau))
    print(f"--- {tag} (beta={beta}, tau={tau}, p={p_mult}/S) ---")
    for name, mask in (("ghost", ghost), ("small", small), ("large", large)):
        if mask.sum() == 0:
            print(f"  {name:6s} n=0"); continue
        print(f"  {name:6s} n={mask.sum():6d}  m med={np.median(m[mask]):.4f}"
              f"  c med={np.median(c[mask]):.3f}"
              f"  g med={np.median(g[mask]):.3f}"
              f"  pass(r>p)={float((lr[mask] > math.log(p + EPS)).mean()):.1%}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/slotcurri/ytvis2021_attnmass_v20.yaml")
    ap.add_argument("--ckpt",
                    default="logs/_ytvis_attnmass_v20/checkpoints/slotcurri_step=step=100000-v1.ckpt")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--max-clips", type=int, default=60)
    ap.add_argument("--cache", default="event_analysis/gamma1_stats_v20.npz")
    args = ap.parse_args()

    if args.cache and os.path.exists(args.cache):
        blob = np.load(args.cache)
        m1, c1, m2, c2, wins = blob["m1"], blob["c1"], blob["m2"], blob["c2"], blob["wins"]
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
        # two passes share the loader recipe; clips may differ slightly if the loader
        # shuffles, so collect both gammas from the same pass instead
        loader = dataset.val_dataloader()
        m1, c1, wins = None, None, None
        ms1, cs1, ms2, cs2, wl = [], [], [], [], []
        n_clips = 0
        with torch.no_grad():
            for batch in loader:
                if "batch_padding_mask" in batch:
                    batch = model._remove_padding(batch, batch["batch_padding_mask"])
                    if batch is None:
                        continue
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    outputs = model.forward(batch, train=False, cycle=model.cyclic_inference)
                att = outputs["processor"]["state_attn_mask"].float()
                b, t, s, f = att.shape
                att = att.reshape(b * t, s, f)
                for gamma, msl, csl in ((1.0, ms1, cs1), (2.0, ms2, cs2)):
                    if gamma != 1.0:
                        att_g = att.pow(gamma)
                        att_g = att_g / att_g.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    else:
                        att_g = att
                    m = att_g.sum(dim=-1) / f
                    p_feat = att_g / att_g.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    ent = -(p_feat * p_feat.clamp_min(1e-8).log()).sum(dim=-1)
                    c = (1.0 - ent / math.log(f)).clamp(0.0, 1.0)
                    msl.append(m.cpu()); csl.append(c.cpu())
                winner = att.argmax(dim=1)
                wc = torch.zeros(b * t, s, device=att.device)
                wc.scatter_add_(1, winner, torch.ones_like(winner, dtype=wc.dtype))
                wl.append(wc.cpu())
                n_clips += b
                if n_clips >= args.max_clips:
                    break
        m1 = torch.cat(ms1).numpy(); c1 = torch.cat(cs1).numpy()
        m2 = torch.cat(ms2).numpy(); c2 = torch.cat(cs2).numpy()
        wins = torch.cat(wl).numpy()
        np.savez_compressed(args.cache, m1=m1, c1=c1, m2=m2, c2=c2, wins=wins)
        print(f"cached {args.cache} ({n_clips} clips)")

    report("gamma=1 candidate", m1, c1, wins, beta=0.7, tau=0.25, p_mult=0.1)
    report("gamma=2 current  ", m2, c2, wins, beta=0.7, tau=0.5, p_mult=0.1)
    report("gamma=1, beta pushed to 0.5 (more c weight)", m1, c1, wins, beta=0.5, tau=0.25, p_mult=0.1)


if __name__ == "__main__":
    main()
