"""Measure the real per-slot attention-mass distribution of a trained checkpoint and
evaluate how much the final gate temperature (tau_end) matters at a given p_end.

The gate is g = sigmoid((m - p) / tau). Because the decoder renormalizes masks after
multiplying by g, only the *relative* spread of g across slots affects reconstruction.
This script reports both the effective slot count (sum g) and the relative-weight spread
so that "does tau_end 0.3 vs 0.1 matter" can be answered on measured masses instead of
hypothetical ones.

Usage (inside the container):
  python event_analysis/tau_end_sensitivity.py <config> <checkpoint> [n_batches]
"""
import os
import sys

import numpy as np
import torch

from slotcurri import configuration, data, models

CFG = sys.argv[1]
CKPT = sys.argv[2]
N_BATCHES = int(sys.argv[3]) if len(sys.argv) > 3 else 15
# Measured masses are cached so threshold sweeps can be rerun without a GPU.
CACHE = sys.argv[4] if len(sys.argv) > 4 else "event_analysis/mass_cache.npy"

if os.path.exists(CACHE):
    masses = np.load(CACHE)
    S = masses.shape[1]
    gamma = 2.0
    print("loaded cached masses from %s" % CACHE)
else:
    config = configuration.load_config(CFG)
    dataset = data.build(config.dataset)
    dataset.setup("validate")
    loader = dataset.val_dataloader()

    model = models.build(config.model, config.optimizer, None, None)
    ckpt = torch.load(CKPT, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print("load_state_dict: missing=%d unexpected=%d" % (len(missing), len(unexpected)))
    model = model.cuda().eval()

    gamma = float(model.amc_mass_gamma)
    S = model.n_slots
    print("n_slots=%d  mass_gamma=%.1f  amc_p_end=%.4f  amc_tau_end=%.4f"
          % (S, gamma, model.amc_p_end, model.amc_tau_end))

    buf = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if batch is None:
                continue
            if bi >= N_BATCHES:
                break
            if "batch_padding_mask" in batch:
                batch = model._remove_padding(batch, batch["batch_padding_mask"])
                if batch is None:
                    continue
            batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            out = model.forward(batch, train=False, cycle=False)
            att = out["processor"]["state_attn_mask"]  # (B, T, S, F), softmax over slots
            if gamma != 1.0:
                att = att.pow(gamma)
                att = att / att.sum(dim=2, keepdim=True).clamp_min(1e-8)
            mass = att.sum(-1) / att.shape[-1]  # (B, T, S)
            buf.append(mass.flatten(0, 1).float().cpu())

    masses = torch.cat(buf).numpy()  # (N_frames, S)
    np.save(CACHE, masses)
    print("cached masses to %s" % CACHE)
print("frames measured:", masses.shape[0])

srt = -np.sort(-masses, axis=1)  # per-frame descending: rank-ordered mass
print("\n=== rank-ordered per-slot mass (gamma=%.1f sharpened) ===" % gamma)
print("%6s %8s %8s %8s %8s" % ("rank", "mean", "p10", "p50", "p90"))
for r in range(S):
    c = srt[:, r]
    print("%6d %8.4f %8.4f %8.4f %8.4f"
          % (r, c.mean(), np.percentile(c, 10), np.median(c), np.percentile(c, 90)))

P_MULTS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
T_MULTS = [0.1, 0.2, 0.3]

print("\n=== (p_end, tau_end) sweep: effective slots and gate influence ===")
print("w-spread = mean(max(w) - min(w)) where w = g / mean(g); this is what survives")
print("the decoder mask renormalization (0 = gate is a no-op).")
print("\n%10s %10s %10s %10s %10s %10s" % ("p_mult", "tau_mult", "p", "tau", "sum g", "w-spread"))
for p_mult in P_MULTS:
    p = p_mult / S
    for t_mult in T_MULTS:
        t = t_mult / S
        g = 1.0 / (1.0 + np.exp(-(srt - p) / t))
        w = g / g.mean(axis=1, keepdims=True)
        print("%10.2f %10.1f %10.4f %10.4f %10.3f %10.3f"
              % (p_mult, t_mult, p, t, g.sum(axis=1).mean(),
                 (w.max(axis=1) - w.min(axis=1)).mean()))

print("\n=== per-rank gate values ===")
for p_mult in P_MULTS:
    p = p_mult / S
    for t_mult in T_MULTS:
        t = t_mult / S
        g = 1.0 / (1.0 + np.exp(-(srt - p) / t))
        print("p=%.2f tau=%.1f : %s" % (p_mult, t_mult, " ".join("%.3f" % x for x in g.mean(axis=0))))
