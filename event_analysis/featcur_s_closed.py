"""Closed-form start mix from the top axis of X^rel.

X(s) = (1-s) R + s X. Let v be the leading principal axis of centered R.
Energy along v and total energy are both quadratic in s:

    q(s) = ||(1-s) R_c v + s X_c v||^2
    T(s) = ||(1-s) R_c + s X_c||_F^2

s is the root in [0, 1] of q(s) = T(s) / 2, or 0 if q(0) <= T(0) / 2.
Compared with the searched s where the true leading share of X(s) drops to 1/2.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_strength_probe import extract_tokens  # noqa: E402
from featcur_xrel_probe import DATASETS  # noqa: E402
from slotcurri import configuration, data
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor


def closed_form(raw: torch.Tensor, rel: torch.Tensor) -> float:
    raw_c = raw.float() - raw.float().mean(0, keepdim=True)
    rel_c = rel.float() - rel.float().mean(0, keepdim=True)
    # Leading axis of the leveled tokens.
    _, _, vh = torch.linalg.svd(rel_c, full_matrices=False)
    v = vh[0]
    a = rel_c @ v
    b = raw_c @ v
    qa, qb, qc = (a * a).sum(), (b * b).sum(), (a * b).sum()
    ta = (rel_c * rel_c).sum()
    tb = (raw_c * raw_c).sum()
    tc = (rel_c * raw_c).sum()
    # 2 q(s) - T(s) = A s^2 + B s + C
    A = 2 * (qa + qb - 2 * qc) - (ta + tb - 2 * tc)
    B = 2 * (-2 * qa + 2 * qc) - (-2 * ta + 2 * tc)
    C = 2 * qa - ta
    if float(C) <= 0:
        return 0.0
    disc = B * B - 4 * A * C
    roots = []
    if float(A.abs()) < 1e-12:
        if float(B.abs()) > 1e-12:
            roots.append(float((-C / B).item()))
    elif float(disc) >= 0:
        sd = torch.sqrt(disc)
        roots.append(float(((-B + sd) / (2 * A)).item()))
        roots.append(float(((-B - sd) / (2 * A)).item()))
    hits = [r for r in roots if 0.0 <= r <= 1.0]
    return min(hits) if hits else 1.0


def true_alpha(raw: torch.Tensor, rel: torch.Tensor, s: float) -> float:
    mixed = (1.0 - s) * rel + s * raw
    xc = mixed.float() - mixed.float().mean(0, keepdim=True)
    energy = torch.linalg.svdvals(xc).square()
    tot = energy.sum().clamp_min(1e-12)
    return float((energy[0] / tot).item())


def searched(raw: torch.Tensor, rel: torch.Tensor) -> float:
    if true_alpha(raw, rel, 0.0) <= 0.5:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        if true_alpha(raw, rel, mid) > 0.5:
            lo = mid
        else:
            hi = mid
    return hi


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    max_scan = 12
    for name, (cfg_path, _m, _b, label) in DATASETS.items():
        cfg = configuration.load_config(cfg_path)
        cfg.dataset.val_batch_size = 1
        cfg.dataset.num_val_workers = 0
        dm = data.build(cfg.dataset, data_dir="/workspace/dataset")
        dm.setup("validate")
        bb = cfg.model.encoder.backbone
        backbone = TimmExtractor(
            model=bb.model,
            pretrained=True,
            frozen=True,
            features=bb.features,
            model_kwargs=dict(bb.get("model_kwargs") or {}) or None,
        ).to(device).eval()
        level = NcutRelationalLeveling(chunk_size=1, barrier=False).to(device)
        closed, search = [], []
        n = 0
        print(f"\n=== {label} ===", flush=True)
        for batch in dm.val_dataloader():
            if n >= max_scan:
                break
            if "batch_padding_mask" in batch and torch.is_tensor(batch["batch_padding_mask"]) and bool(batch["batch_padding_mask"].any()):
                n += 1
                continue
            frame = batch["video"][0, int(batch["video"].shape[1] // 2)].to(device)
            raw = extract_tokens(backbone, frame.unsqueeze(0), bb.features).float()[0]
            rel = level(raw.unsqueeze(0), 0.0)[0]
            sc, ss = closed_form(raw, rel), searched(raw, rel)
            closed.append(sc)
            search.append(ss)
            n += 1
            print(f"  clip {n-1:3d}  closed={sc:.3f}  search={ss:.3f}", flush=True)
        print(
            f"{label}: median closed={float(np.median(closed)):.3f}  "
            f"median search={float(np.median(search)):.3f}  "
            f"median |gap|={float(np.median(np.abs(np.array(closed)-np.array(search)))):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
