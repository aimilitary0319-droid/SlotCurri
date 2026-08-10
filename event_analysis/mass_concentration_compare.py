"""Compare mass concentration and per-object fragmentation between two trained checkpoints.

Motivating question: the attention-mass gate anneals to a near-uniform state by the end of
training (contrast g(1.5x)/g(0.5x) falls 14.5x -> 1.3x), so at eval it barely reweights the
decoder. Does the strong early phase nevertheless leave a *persistent* bias toward one slot
covering a whole object, or does the late near-uniform phase let featrec re-fragment?

Two families of measurements, both at eval settings (cycle=False, train=False):

1. Mass concentration: rank-ordered per-slot attention mass, reported at gamma=1 (plain) and
   gamma=2 (sharpened) for BOTH models, so a model whose config sets mass_gamma=1 is never
   compared against a gamma=2 measurement.

2. Per-object fragmentation, the quantity the "one slot covers the object" claim is really
   about: for each GT object, how many decoder slots claim a meaningful share of it, and how
   much of it the single best slot captures.
"""
import sys

import numpy as np
import torch

from slotcurri import configuration, data, models

# (label, settings.yaml, checkpoint)
RUNS = [
    ("baseline", "logs/_ytvis/settings/slotcurri/settings.yaml",
     "logs/_ytvis/checkpoints/slotcurri_step=step=100000-v1.ckpt"),
    ("v10", "logs/_ytvis_attnmass_v10/settings/slotcurri/settings.yaml",
     "logs/_ytvis_attnmass_v10/checkpoints/slotcurri_step=step=100000-v1.ckpt"),
]
N_CLIPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
# A slot "claims" an object if it owns at least this fraction of the object's pixels.
CLAIM_FRAC = 0.1


def rank_stats(masses: np.ndarray) -> dict:
    srt = -np.sort(-masses, axis=1)
    q = srt / srt.sum(axis=1, keepdims=True).clip(1e-9)
    ent = -(q * np.log(q.clip(1e-9))).sum(axis=1)
    return {
        "ranked": srt.mean(axis=0),
        "top1": srt[:, 0].mean(),
        "top2": srt[:, :2].sum(axis=1).mean(),
        "entropy": ent.mean(),
    }


def measure(label: str, cfg_path: str, ckpt_path: str, n_clips: int) -> dict:
    config = configuration.load_config(cfg_path)
    dataset = data.build(config.dataset)
    dataset.setup("validate")
    loader = dataset.val_dataloader()

    model = models.build(config.model, config.optimizer, None, None)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    model = model.cuda().eval()
    print(f"[{label}] load_state_dict missing={len(missing)} unexpected={len(unexpected)} "
          f"attn_mass={model.attn_mass_enabled} cfg_gamma={model.amc_mass_gamma}")

    mass_g1, mass_g2 = [], []
    claims, best_iou, obj_counts = [], [], []

    with torch.no_grad():
        seen = 0
        for batch in loader:
            if batch is None:
                continue
            if seen >= n_clips:
                break
            if "batch_padding_mask" in batch:
                batch = model._remove_padding(batch, batch["batch_padding_mask"])
                if batch is None:
                    continue
            batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            out = model.forward(batch, train=False, cycle=False)
            aux = model.aux_forward(batch, out)
            seen += 1

            att = out["processor"]["state_attn_mask"]  # (B, T, S, F) softmax over slots
            for gamma, sink in ((1.0, mass_g1), (2.0, mass_g2)):
                a = att.pow(gamma)
                a = a / a.sum(dim=2, keepdim=True).clamp_min(1e-8)
                sink.append((a.sum(-1) / a.shape[-1]).flatten(0, 1).float().cpu().numpy())

            # per-object fragmentation on the hard decoder masks used for the metrics
            pred = aux.get("decoder_masks_hard")
            gt = batch.get("segmentations")
            if pred is None or gt is None:
                continue
            # pred: (B, T, S, H, W) one-hot over slots; gt: (B, T, H, W) instance ids
            pred_id = pred.argmax(dim=2)  # (B, T, H, W) winning slot per pixel
            n_slots = pred.shape[2]
            ids = [int(i) for i in torch.unique(gt) if int(i) != 0]
            obj_counts.append(len(ids))
            for oid in ids:
                m = gt == oid  # (B, T, H, W)
                area = m.sum().item()
                if area == 0:
                    continue
                inter = torch.stack([(pred_id == s)[m].sum() for s in range(n_slots)]).float()
                share = (inter / area).cpu().numpy()
                claims.append(int((share >= CLAIM_FRAC).sum()))
                # IoU of the best-overlap slot with this object
                s_best = int(inter.argmax())
                union = (m | (pred_id == s_best)).sum().item()
                best_iou.append(inter[s_best].item() / max(union, 1))

    res = {
        "label": label,
        "g1": rank_stats(np.concatenate(mass_g1)),
        "g2": rank_stats(np.concatenate(mass_g2)),
        "clips": seen,
        "frames": sum(len(x) for x in mass_g1),
        "obj_per_clip": float(np.mean(obj_counts)) if obj_counts else float("nan"),
        "claims": float(np.mean(claims)) if claims else float("nan"),
        "claims_ge2": float(np.mean([c >= 2 for c in claims])) if claims else float("nan"),
        "best_iou": float(np.mean(best_iou)) if best_iou else float("nan"),
        "n_obj": len(claims),
    }
    del model
    torch.cuda.empty_cache()
    return res


results = [measure(*r, N_CLIPS) for r in RUNS]

S = len(results[0]["g1"]["ranked"])
print(f"\nclips={results[0]['clips']}  frames={results[0]['frames']}  "
      f"uniform mass={1/S:.4f}  objects/clip={results[0]['obj_per_clip']:.2f}")

for gk, gname in (("g1", "gamma=1 (plain mass)"), ("g2", "gamma=2 (sharpened)")):
    print(f"\n=== rank-ordered per-slot mass, {gname} ===")
    print("%-10s %s %10s %10s %9s" % ("model", "".join("%8s" % f"r{r}" for r in range(S)),
                                      "top1", "top2", "entropy"))
    for r in results:
        st = r[gk]
        print("%-10s %s %10.3f %10.3f %9.3f"
              % (r["label"], "".join("%8.4f" % v for v in st["ranked"]),
                 st["top1"], st["top2"], st["entropy"]))
    print("   (entropy: uniform=%.3f, fully concentrated=0)" % np.log(S))

print("\n=== per-object fragmentation (hard decoder masks vs GT) ===")
print("%-10s %14s %14s %12s" % ("model", "slots/object", "frac >=2 slots", "best-slot IoU"))
for r in results:
    print("%-10s %14.3f %14.3f %12.3f"
          % (r["label"], r["claims"], r["claims_ge2"], r["best_iou"]))
print(f"   (a slot 'claims' an object when it owns >= {CLAIM_FRAC:.0%} of its pixels; "
      f"n_objects={results[0]['n_obj']})")
