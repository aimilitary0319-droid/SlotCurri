"""How much ||m_f|| / ||m_f - mu|| stretches the frame-mean deviation."""
import sys
sys.path.insert(0, "/workspace/SlotCurri/event_analysis")
sys.path.insert(0, "/workspace/SlotCurri")
import numpy as np
import torch
from event_analysis.featcur_xvid.vmean_probe import load_clips, dense_px
from global_winner_probe import CONFIGS
from slotcurri import configuration
from slotcurri.modules.encoders import TimmExtractor

def report(dataset):
    rng = np.random.default_rng(0)
    cfg = configuration.load_config(CONFIGS[dataset])
    bb = cfg.model.encoder.backbone
    backbone = TimmExtractor(
        model=bb.model, pretrained=True, frozen=True, features=bb.features,
        model_kwargs=dict(bb.get("model_kwargs") or {}) or None,
    ).eval()
    clips = load_clips(cfg, "/workspace/dataset", 8, 4, 8, rng)
    imgs = torch.stack(clips, 0).reshape(len(clips) * 4, *clips[0].shape[1:])
    tok = []
    with torch.no_grad():
        for c0 in range(0, imgs.shape[0], 4):
            t = backbone(imgs[c0:c0 + 4])
            tok.append((t[bb.features] if isinstance(t, dict) else t).float())
        tok = torch.cat(tok, 0)
        rel = dense_px(tok)
        mu = tok.mean((0, 1), keepdim=True)
        m = rel.mean(1, keepdim=True)
        full = m.norm(dim=-1).flatten()
        dev = (m - mu).norm(dim=-1).flatten()
        ratio = full / dev.clamp_min(1e-8)
        mu_u = torch.nn.functional.normalize(mu.reshape(-1, mu.shape[-1]), dim=-1)
        mf_u = torch.nn.functional.normalize(m.reshape(-1, m.shape[-1]), dim=-1)
        cos = (mf_u * mu_u).sum(-1)
        print(
            f"{dataset} ||m_f|| {float(full.median()):.2f}  "
            f"||m_f-mu|| {float(dev.median()):.2f}  "
            f"stretch {float(ratio.median()):.1f}x  "
            f"cos(m_f, mu) {float(cos.median()):.3f}",
            flush=True,
        )

for ds in ("movi_c", "ytvis"):
    report(ds)
