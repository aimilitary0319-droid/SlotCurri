"""Find a feature-curriculum start s from foreground/background separability.

X(s) = (1-s) X^rel + s X, with X^rel the v39lam1gu global ReLU-cosine leveling.
s=0 is fully leveled. YouTube-VIS at s=0 is the reference operating point.

Per frame, on the GT foreground (every non-background patch) vs background:

  eta2   correlation ratio: fraction of token variance explained by fg/bg
  fisher between-mean energy over mean within-class energy
  auc    Mann-Whitney AUC of the projection onto mu_fg - mu_bg
  pc1    |cosine| between that axis and the leading principal component
  cos    cosine of the two class means after per-token L2 norm

Usage:
  python event_analysis/featcur_s0_probe.py --data-dir /workspace/dataset
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_strength_probe import extract_tokens, patch_ids_movi, patch_ids_ytvis  # noqa: E402
from featcur_xrel_probe import DATASETS, object_ids  # noqa: E402
from slotcurri import configuration, data
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor

S_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]


def _auc(score: torch.Tensor, positive: torch.Tensor) -> float:
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos < 4 or n_neg < 4:
        return float("nan")
    order = score.argsort()
    ranks = torch.empty(score.shape[0], device=score.device, dtype=torch.float64)
    ranks[order] = torch.arange(1, score.shape[0] + 1, device=score.device, dtype=torch.float64)
    # Average ranks for exact ties so a constant score gives AUC 0.5.
    sorted_score = score[order]
    change = torch.ones(score.shape[0], dtype=torch.bool, device=score.device)
    change[1:] = sorted_score[1:] != sorted_score[:-1]
    group = change.cumsum(0) - 1
    ones = torch.ones_like(ranks)
    sum_r = torch.zeros(int(group[-1]) + 1, device=score.device, dtype=torch.float64)
    cnt = torch.zeros_like(sum_r)
    sum_r.scatter_add_(0, group, ranks)
    cnt.scatter_add_(0, group, ones)
    ranks = (sum_r / cnt.clamp_min(1))[group]
    inv = torch.empty_like(ranks)
    inv[order] = ranks
    sum_pos = inv[positive].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc.item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.float(), dim=0)
    b = F.normalize(b.float(), dim=0)
    return float((a * b).sum().clamp(-1, 1).item())


def fg_bg_metrics(x: torch.Tensor, gt: torch.Tensor, bg_id: int) -> Dict[str, float]:
    fg = gt != bg_id
    n_fg = int(fg.sum())
    n_bg = int((~fg).sum())
    if n_fg < 8 or n_bg < 8:
        return {}
    x = x.float()
    mu = x.mean(0)
    mu_f = x[fg].mean(0)
    mu_b = x[~fg].mean(0)
    total = (x - mu).pow(2).sum().clamp_min(1e-12)
    between = n_fg * (mu_f - mu).pow(2).sum() + n_bg * (mu_b - mu).pow(2).sum()
    within_f = (x[fg] - mu_f).pow(2).sum() / n_fg
    within_b = (x[~fg] - mu_b).pow(2).sum() / n_bg
    sb = (mu_f - mu_b).pow(2).sum()
    direction = mu_f - mu_b
    xc = x - mu
    # Leading PC via the Gram trick when N < D, else the feature Gram.
    if x.shape[0] <= x.shape[1]:
        gram = xc @ xc.T
        evals, evecs = torch.linalg.eigh(gram)
        pc_coeff = evecs[:, -1]
        pc = xc.T @ pc_coeff
    else:
        _, _, vh = torch.linalg.svd(xc, full_matrices=False)
        pc = vh[0]
    xn = F.normalize(x, dim=-1)
    return {
        "eta2": float((between / total).item()),
        "fisher": float((sb / (within_f + within_b).clamp_min(1e-12)).item()),
        "auc": _auc(x @ direction, fg),
        "pc1": abs(_cosine(pc, direction)),
        "cos": _cosine(xn[fg].mean(0), xn[~fg].mean(0)),
        "fg_frac": n_fg / float(n_fg + n_bg),
    }


def nanmean(xs: List[float]) -> float:
    v = [float(x) for x in xs if x == x]
    return float(sum(v) / len(v)) if v else float("nan")


@torch.no_grad()
def collect(name, cfg_path, min_obj, bg_id, data_dir, device, max_scan):
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
    id_fn = patch_ids_ytvis if name == "ytvis" else patch_ids_movi
    buckets = {s: [] for s in S_GRID}
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
        video, seg = batch["video"][0], batch["segmentations"][0]
        ti = int(video.shape[0] // 2)
        frame = video[ti].to(device)
        raw = extract_tokens(backbone, frame.unsqueeze(0), bb.features).float()[0]
        grid = int(round(math.sqrt(raw.shape[0])))
        gt = id_fn(seg[ti].to(device), grid)
        n_obj = len(object_ids(gt, bg_id, min_n=1))
        n_clips += 1
        if n_obj < min_obj:
            continue
        rel = level(raw.unsqueeze(0), 0.0)[0]
        for s in S_GRID:
            mixed = (1.0 - s) * rel + s * raw
            rec = fg_bg_metrics(mixed, gt, bg_id)
            if not rec:
                continue
            rec["s"] = s
            rec["clip"] = n_clips - 1
            buckets[s].append(rec)
        last = buckets[0.0][-1] if buckets[0.0] else None
        if last is not None and last["clip"] == n_clips - 1:
            print(
                f"  clip {n_clips - 1:3d} n_obj={n_obj:2d}  "
                f"s0 eta2={last['eta2']:.3f} auc={last['auc']:.3f} "
                f"pc1={last['pc1']:.3f} cos={last['cos']:.3f}",
                flush=True,
            )
    summary = {}
    for s, rows in buckets.items():
        summary[f"{s:.2f}"] = {
            "n": len(rows),
            **{k: nanmean([r[k] for r in rows]) for k in ("eta2", "fisher", "auc", "pc1", "cos", "fg_frac")},
        }
    return summary


def first_crossing(summary: dict, key: str, threshold: float) -> float:
    for s in S_GRID:
        val = summary[f"{s:.2f}"][key]
        if val == val and val >= threshold:
            return float(s)
    return float("nan")


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
        cfg, min_obj, bg_id, label = DATASETS[name]
        summary = collect(name, cfg, min_obj, bg_id, args.data_dir, device, args.max_scan)
        out[name] = {"label": label, "by_s": summary}
    yt = out["ytvis"]["by_s"]["0.00"]
    print("\n===== reference: YouTube-VIS at s=0 =====", flush=True)
    for k in ("eta2", "fisher", "auc", "pc1", "cos"):
        print(f"  {k:8s} {yt[k]:.4f}", flush=True)
    print("\n===== smallest s reaching the YouTube s=0 value =====", flush=True)
    for name, blob in out.items():
        picks = {}
        for k in ("eta2", "fisher", "auc", "pc1"):
            picks[k] = first_crossing(blob["by_s"], k, yt[k])
        # cos is similarity: separation is 1-cos, so crossing is cos <= YouTube cos
        cos_ref = yt["cos"]
        cos_s = float("nan")
        for s in S_GRID:
            val = blob["by_s"][f"{s:.2f}"]["cos"]
            if val == val and val <= cos_ref:
                cos_s = float(s)
                break
        picks["cos_le_yt"] = cos_s
        blob["s0_match_yt"] = picks
        print(
            f"  {blob['label']:12s}  eta2>={yt['eta2']:.3f} -> {picks['eta2']:.2f}   "
            f"auc>={yt['auc']:.3f} -> {picks['auc']:.2f}   "
            f"pc1>={yt['pc1']:.3f} -> {picks['pc1']:.2f}   "
            f"cos<={cos_ref:.3f} -> {cos_s:.2f}",
            flush=True,
        )
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", os.path.join(args.out_dir, "summary.json"), flush=True)


if __name__ == "__main__":
    main()
