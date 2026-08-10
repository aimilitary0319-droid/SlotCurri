"""Probe the attention-centroid repulsion loss on a trained checkpoint.

Reports the distribution of pairwise slot-centroid distances and, for a few candidate
margins, how much gate-weighted pair mass falls inside the margin plus the resulting
loss value. This says whether `gate_rep` would be an inert term, a gentle nudge, or a
dominant force before spending a training run on it.

Usage (inside the container):
  python event_analysis/repulsion_probe.py <config> <checkpoint> [n_batches]
"""
import sys

import numpy as np
import torch

from slotcurri import configuration, data, models

CFG = sys.argv[1]
CKPT = sys.argv[2]
N_BATCHES = int(sys.argv[3]) if len(sys.argv) > 3 else 10

config = configuration.load_config(CFG)
dataset = data.build(config.dataset)
dataset.setup("validate")
loader = dataset.val_dataloader()

model = models.build(config.model, config.optimizer, None, None)
ckpt = torch.load(CKPT, map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=False)
model = model.cuda().eval()

gamma = float(model.amc_mass_gamma)
p_end = model.amc_p_end
tau_end = model.amc_tau_end
print("mass_gamma=%.1f p_end=%.4f tau_end=%.4f" % (gamma, p_end, tau_end))

dists, pair_gates = [], []
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
        att = out["processor"]["state_attn_mask"]  # (B, T, S, F)
        B, T, S, F = att.shape
        H = W = int(F**0.5)

        ys, xs = torch.meshgrid(
            torch.linspace(0, 1, H, device=att.device),
            torch.linspace(0, 1, W, device=att.device),
            indexing="ij",
        )
        pos = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1)  # (F, 2)

        mass_raw = att.sum(-1).clamp_min(1e-8)
        centroids = torch.einsum("btsf,fd->btsd", att, pos) / mass_raw.unsqueeze(-1)
        diff = centroids.unsqueeze(3) - centroids.unsqueeze(2)
        dist = diff.pow(2).sum(-1).clamp_min(0).sqrt()  # (B, T, S, S)

        att_g = att.pow(gamma) if gamma != 1.0 else att
        att_g = att_g / att_g.sum(dim=2, keepdim=True).clamp_min(1e-8)
        mass = att_g.sum(-1) / F
        g = torch.sigmoid((mass - p_end) / tau_end)  # gate at inference thresholds
        pg = g.unsqueeze(3) * g.unsqueeze(2)

        iu = torch.triu(torch.ones(S, S, device=att.device, dtype=torch.bool), diagonal=1)
        dists.append(dist[:, :, iu].reshape(-1).cpu())
        pair_gates.append(pg[:, :, iu].reshape(-1).cpu())

d = torch.cat(dists).numpy()
pg = torch.cat(pair_gates).numpy()
print("slot pairs measured: %d" % d.size)
print("\n=== pairwise centroid distance (patch coords in [0,1]^2, max = 1.414) ===")
for q in [5, 10, 25, 50, 75, 90]:
    print("  p%02d: %.4f" % (q, np.percentile(d, q)))
print("  mean: %.4f" % d.mean())

print("\n=== repulsion at candidate margins ===")
print("%10s %14s %16s %14s" % ("margin", "pairs inside", "gate-wt inside", "L_rep"))
for m in [0.1, 0.15, 0.2, 0.3, 0.4]:
    hinge = np.clip(m - d, 0, None) ** 2
    L = (pg * hinge).sum() / max(pg.sum(), 1e-8)
    print("%10.2f %13.1f%% %15.1f%% %14.5f"
          % (m, 100 * (d < m).mean(), 100 * (pg * (d < m)).sum() / max(pg.sum(), 1e-8), L))
print("\nweighted contribution to total loss at gate_rep=0.05:")
for m in [0.2]:
    hinge = np.clip(m - d, 0, None) ** 2
    L = (pg * hinge).sum() / max(pg.sum(), 1e-8)
    print("  margin %.2f -> 0.05 * %.5f = %.6f  (compare loss_featrec ~1.7)" % (m, L, 0.05 * L))
