"""Dataset-only start mix for the v39lam1gu feature curriculum.

No reference dataset. On each frame, X^rel = P X is the fully leveled Key and

    X(s) = (1 - s) X^rel + s X.

PR(s) is the participation ratio of the centered tokens,

    PR = (sum_i λ_i)^2 / sum_i λ_i^2.

PR = 1 is a single appearance direction: every slot query ranks patches the
same way. PR = 2 is the smallest spectrum with two effective directions, so a
second slot has a distinct basin. The frame's start mix is the smallest s
with PR(X(s)) >= 2. The dataset value is the median over frames.

Usage:
  python event_analysis/featcur_s_metric.py --data-dir /workspace/dataset
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_strength_probe import extract_tokens  # noqa: E402
from featcur_xrel_probe import DATASETS  # noqa: E402
from slotcurri import configuration, data
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor

S_GRID = [i / 100 for i in range(0, 31, 2)] + [0.40, 0.50, 0.70, 1.0]
PR_TARGET = 2.0


def spectrum(x: torch.Tensor) -> Dict[str, float]:
    xc = x.float() - x.float().mean(0, keepdim=True)
    energy = torch.linalg.svdvals(xc).square()
    tot = energy.sum()
    if float(tot) <= 1e-8:
        return {"pr": 1.0, "alpha": 1.0, "erank": 1.0}
    p = energy / tot
    pr = float((tot.square() / energy.square().sum().clamp_min(1e-12)).item())
    alpha = float(p[0].item()) if p.numel() else 1.0
    # svdvals are descending, so p[0] is the leading share.
    erank = float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item())
    return {"pr": pr, "alpha": alpha, "erank": erank}


def first_s(curve: List[dict], key: str, pred) -> float:
    """Smallest grid s meeting pred, linearly interpolated against the previous point."""
    prev = None
    for row in curve:
        if pred(row[key]):
            if prev is None:
                return float(row["s"])
            v0, v1 = prev[key], row[key]
            if v1 == v0:
                return float(row["s"])
            # Interpolate to the boundary implied by pred. For '>=' the root is
            # where the value crosses the previous failure; caller passes values
            # already compared, so we interpolate using the stored target gap.
            return float(row["s"])
        prev = row
    return 1.0


def crossing(curve: List[dict], key: str, target: float, at_most: bool = False) -> float:
    prev = None
    for row in curve:
        ok = row[key] <= target if at_most else row[key] >= target
        if ok:
            if prev is None:
                return 0.0
            v0, v1 = float(prev[key]), float(row[key])
            if v1 == v0:
                return float(row["s"])
            frac = (target - v0) / (v1 - v0)
            frac = min(max(frac, 0.0), 1.0)
            return float(prev["s"] + frac * (row["s"] - prev["s"]))
        prev = row
    return 1.0


def summarize(values: List[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p25": float(np.quantile(a, 0.25)),
        "p75": float(np.quantile(a, 0.75)),
        "frac_zero": float(np.mean(a <= 1e-8)),
    }


@torch.no_grad()
def collect(name, cfg_path, data_dir, device, max_scan):
    cfg = configuration.load_config(cfg_path)
    cfg.dataset.val_batch_size = 1
    cfg.dataset.num_val_workers = 0
    dm = data.build(cfg.dataset, data_dir=data_dir)
    dm.setup("validate")
    bb = cfg.model.encoder.backbone
    kwargs = dict(bb.get("model_kwargs") or {})
    backbone = (
        TimmExtractor(
            model=bb.model,
            pretrained=True,
            frozen=True,
            features=bb.features,
            model_kwargs=kwargs or None,
        )
        .to(device)
        .eval()
    )
    level = NcutRelationalLeveling(chunk_size=1, n_iter=16, barrier=False).to(device)
    rows = []
    n_clips = 0
    print(f"\n=== {name} ===", flush=True)
    for batch in dm.val_dataloader():
        if n_clips >= max_scan:
            break
        if "batch_padding_mask" in batch and torch.is_tensor(batch["batch_padding_mask"]) and bool(
            batch["batch_padding_mask"].any()
        ):
            n_clips += 1
            continue
        video = batch["video"][0]
        ti = int(video.shape[0] // 2)
        frame = video[ti].to(device)
        raw = extract_tokens(backbone, frame.unsqueeze(0), bb.features).float()[0]
        rel = level(raw.unsqueeze(0), 0.0)[0]
        curve = []
        for s in S_GRID:
            stats = spectrum((1.0 - s) * rel + s * raw)
            stats["s"] = s
            curve.append(stats)
        rec = {
            "clip": n_clips,
            "s_pr2": crossing(curve, "pr", PR_TARGET, at_most=False),
            "s_erank2": crossing(curve, "erank", PR_TARGET, at_most=False),
            "s_alpha50": crossing(curve, "alpha", 0.5, at_most=True),
            "pr0": curve[0]["pr"],
            "alpha0": curve[0]["alpha"],
            "erank0": curve[0]["erank"],
        }
        rows.append(rec)
        n_clips += 1
        print(
            f"  clip {rec['clip']:3d}  pr0={rec['pr0']:.2f} alpha0={rec['alpha0']:.2f} "
            f"erank0={rec['erank0']:.2f}  s(PR>=2)={rec['s_pr2']:.3f} "
            f"s(erank>=2)={rec['s_erank2']:.3f} s(alpha<=0.5)={rec['s_alpha50']:.3f}",
            flush=True,
        )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/workspace/dataset")
    p.add_argument("--out-dir", default="event_analysis/featcur_s0")
    p.add_argument("--max-scan", type=int, default=36)
    p.add_argument("--datasets", nargs="+", default=["ytvis", "movi_c", "movi_e"])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    out = {}
    for name in args.datasets:
        cfg, _min_obj, _bg, label = DATASETS[name]
        rows = collect(name, cfg, args.data_dir, device, args.max_scan)
        out[name] = {
            "label": label,
            "s_pr2": summarize([r["s_pr2"] for r in rows]),
            "s_erank2": summarize([r["s_erank2"] for r in rows]),
            "s_alpha50": summarize([r["s_alpha50"] for r in rows]),
            "pr0": summarize([r["pr0"] for r in rows]),
            "alpha0": summarize([r["alpha0"] for r in rows]),
            "erank0": summarize([r["erank0"] for r in rows]),
            "frames": rows,
        }
        blob = out[name]
        print(
            f"\n{label}: median s | PR>=2 {blob['s_pr2']['median']:.3f} "
            f"(p75 {blob['s_pr2']['p75']:.3f}, zero {blob['s_pr2']['frac_zero']:.2f}) | "
            f"erank>=2 {blob['s_erank2']['median']:.3f} | "
            f"alpha<=0.5 {blob['s_alpha50']['median']:.3f}",
            flush=True,
        )
    path = os.path.join(args.out_dir, "s_metric.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
