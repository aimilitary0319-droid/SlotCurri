"""X^bind = (1-s) X^rel + s X along the schedule, for dense FC, xvid_energy and xframe.

Per s: within-frame / cross-video LN+L2 key cosine, and deviation energy
E||X^bind - mu||^2 / E||X - mu||^2 (mu = batch mean). One frame per video, 32 per batch.

Usage (inside the slotcurri image):
  python event_analysis/featcur_xvid/blend_probe.py movi_e
"""
import sys

sys.path.insert(0, "/workspace/SlotCurri/event_analysis")
sys.path.insert(0, "/workspace/SlotCurri")
import numpy as np
import torch
import torch.nn.functional as F
from featcur_xvid_probe import xframe_rescale
from global_winner_probe import CONFIGS, load_frames
from slotcurri import configuration
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor

ds = sys.argv[1]
dev = torch.device("cuda")
cfg = configuration.load_config(CONFIGS[ds])
bb = cfg.model.encoder.backbone
backbone = TimmExtractor(model=bb.model, pretrained=True, frozen=True, features=bb.features,
                         model_kwargs=dict(bb.get("model_kwargs") or {}) or None).to(dev).eval()
dense_l = NcutRelationalLeveling(chunk_size=4, barrier=False).to(dev)
xvid_l = NcutRelationalLeveling(chunk_size=4, barrier=False, xvid_energy=True).to(dev)
frames = load_frames(cfg, "/workspace/dataset", 128, 6, np.random.default_rng(0))
S = (0.0, 0.25, 0.5, 0.75, 1.0)
MODES = ("dense", "xvid", "xframe")
st = {(m, s): {"S": None, "Q": 0.0, "cin": [], "V": 0, "dev": []} for m in MODES for s in S}
with torch.no_grad():
    for b0 in range(0, len(frames), 32):
        x = torch.stack(frames[b0:b0 + 32]).to(dev)
        tok = torch.cat([(lambda t: (t[bb.features] if isinstance(t, dict) else t).float())(backbone(x[c:c + 8]))
                         for c in range(0, x.shape[0], 8)])
        mu = tok.mean((0, 1), keepdim=True)
        raw_e = (tok - mu).square().sum(-1).mean()
        rel = {"dense": dense_l(tok, 0.0), "xvid": xvid_l(tok, 0.0)}
        rel["xframe"], _ = xframe_rescale(tok, rel["dense"])
        for m in MODES:
            for s in S:
                k = (1.0 - s) * rel[m] + s * tok
                u = F.normalize(F.layer_norm(k, (k.shape[-1],)), dim=-1)
                mm = u.mean(1)
                a = st[(m, s)]
                a["S"] = mm.sum(0) if a["S"] is None else a["S"] + mm.sum(0)
                a["Q"] += float(mm.square().sum())
                a["V"] += mm.shape[0]
                n = tok.shape[1]
                a["cin"] += ((n * mm.square().sum(-1) - 1) / (n - 1)).tolist()
                a["dev"].append(float((k - mu).square().sum(-1).mean() / raw_e))
print(f"{ds}: s | " + " | ".join(f"{m} within / cross / dev-energy" for m in MODES))
for s in S:
    row = [f"s={s:.2f}"]
    for m in MODES:
        a = st[(m, s)]
        V = a["V"]
        cross = (float(a["S"].square().sum()) - a["Q"]) / (V * (V - 1))
        row.append(f"{np.mean(a['cin']):.3f} / {cross:.3f} / {np.mean(a['dev']):.2f}")
    print("  " + "  |  ".join(row))
