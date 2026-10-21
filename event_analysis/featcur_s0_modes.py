"""Sweep the curriculum mix for metrics that track slot collapse.

X(s) = (1-s) X^rel + s X. Binary fg/bg Fisher is the wrong dial: at s=0 MOVi's
residual variance is still aligned with the fg mean, while the tokens themselves
sit in one cluster. These metrics ask whether more than one slot has a basin:

  pr        participation ratio of centered tokens (effective rank)
  keys      greedy count of L2 keys with mutual cosine < 0.95
  obj_cos   mean pairwise cosine of per-object means (1 = instances merged)
  sil       mean cosine-silhouette of GT instances; 0 if both a and b vanish
  contrast  energy of X(s) in the top-3 principal axes of raw X, over raw energy
  gap2      1 - cosine between the two k-means centers on L2 tokens

Usage:
  python event_analysis/featcur_s0_modes.py --data-dir /workspace/dataset
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
from featcur_xrel_probe import DATASETS, object_ids, unique_keys  # noqa: E402
from slotcurri import configuration, data
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor

S_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


def participation_ratio(x: torch.Tensor) -> float:
    xc = x.float() - x.float().mean(0, keepdim=True)
    sv = torch.linalg.svdvals(xc)
    energy = sv.square()
    tot = energy.sum().clamp_min(1e-12)
    return float((tot.square() / energy.square().sum().clamp_min(1e-12)).item())


def object_mean_cos(x: torch.Tensor, gt: torch.Tensor, bg_id: int) -> float:
    means = []
    xn = F.normalize(x.float(), dim=-1)
    for oid in object_ids(gt, bg_id, min_n=4):
        means.append(F.normalize(xn[gt == oid].mean(0), dim=0))
    if len(means) < 2:
        return float("nan")
    m = torch.stack(means)
    s = m @ m.T
    k = s.shape[0]
    iu = torch.triu_indices(k, k, offset=1, device=s.device)
    return float(s[iu[0], iu[1]].mean().item())


def instance_silhouette(x: torch.Tensor, gt: torch.Tensor, bg_id: int) -> float:
    ids = object_ids(gt, bg_id, min_n=8)
    if len(ids) < 2:
        return float("nan")
    xn = F.normalize(x.float(), dim=-1)
    dist = 1.0 - xn @ xn.T
    vals = []
    index = {oid: (gt == oid).nonzero(as_tuple=False).squeeze(-1) for oid in ids}
    for oid in ids:
        idx = index[oid]
        n = int(idx.numel())
        own = dist[idx][:, idx]
        a = float((own.sum() / max(n - 1, 1)).item())
        b = min(float(dist[idx][:, index[other]].mean().item()) for other in ids if other != oid)
        if a < 1e-3 and b < 1e-3:
            vals.append(0.0)
            continue
        vals.append((b - a) / max(a, b, 1e-8))
    return float(sum(vals) / len(vals))


def kmeans2_gap(x: torch.Tensor, n_iter: int = 15) -> float:
    """1 - cosine of the two centers. 0 means the split is one cluster."""
    xn = F.normalize(x.float(), dim=-1)
    n = xn.shape[0]
    # Farthest-point init so a real second mode is picked when it exists.
    c0 = xn[n // 2]
    d0 = 1.0 - xn @ c0
    c1 = xn[int(d0.argmax())]
    for _ in range(n_iter):
        a0 = xn @ c0
        a1 = xn @ c1
        lab = a1 > a0
        if int(lab.sum()) < 2 or int((~lab).sum()) < 2:
            return 0.0
        c0 = F.normalize(xn[~lab].mean(0), dim=0)
        c1 = F.normalize(xn[lab].mean(0), dim=0)
    return float((1.0 - (c0 * c1).sum().clamp(-1, 1)).item())


def retained_contrast(raw: torch.Tensor, mixed: torch.Tensor, k: int = 3) -> float:
    """Energy of `mixed` lying in the top-k principal axes of `raw`."""
    raw_c = raw.float() - raw.float().mean(0, keepdim=True)
    gram = raw_c @ raw_c.T
    evals, evecs = torch.linalg.eigh(gram)
    coeff = evecs[:, -k:]
    axes = raw_c.T @ coeff
    axes = axes / axes.norm(dim=0, keepdim=True).clamp_min(1e-8)
    raw_e = (raw_c @ axes).square().sum().clamp_min(1e-12)
    mix_c = mixed.float() - mixed.float().mean(0, keepdim=True)
    mix_e = (mix_c @ axes).square().sum()
    return float((mix_e / raw_e).item())


def nanmean(xs: List[float]) -> float:
    v = [float(x) for x in xs if x == x]
    return float(sum(v) / len(v)) if v else float("nan")


def score(raw: torch.Tensor, mixed: torch.Tensor, gt: torch.Tensor, bg_id: int) -> Dict[str, float]:
    return {
        "pr": participation_ratio(mixed),
        "keys": float(unique_keys(mixed, 0.95)),
        "obj_cos": object_mean_cos(mixed, gt, bg_id),
        "sil": instance_silhouette(mixed, gt, bg_id),
        "contrast": retained_contrast(raw, mixed, 3),
        "gap2": kmeans2_gap(mixed),
    }


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
    kept = 0
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
            rec = score(raw, mixed, gt, bg_id)
            rec["clip"] = n_clips - 1
            buckets[s].append(rec)
        z = buckets[0.0][-1]
        kept += 1
        print(
            f"  clip {n_clips - 1:3d}  keys={z['keys']:.0f} pr={z['pr']:.2f} "
            f"obj_cos={z['obj_cos']:.3f} sil={z['sil']:.3f} "
            f"contrast={z['contrast']:.3f} gap2={z['gap2']:.4f}",
            flush=True,
        )
    summary = {}
    for s, rows in buckets.items():
        summary[f"{s:.2f}"] = {
            "n": len(rows),
            **{k: nanmean([r[k] for r in rows]) for k in ("pr", "keys", "obj_cos", "sil", "contrast", "gap2")},
        }
    print(f"  kept {kept}", flush=True)
    return summary


def first_at_least(summary: dict, key: str, threshold: float) -> float:
    for s in S_GRID:
        val = summary[f"{s:.2f}"][key]
        if val == val and val >= threshold:
            return float(s)
    return float("nan")


def first_at_most(summary: dict, key: str, threshold: float) -> float:
    for s in S_GRID:
        val = summary[f"{s:.2f}"][key]
        if val == val and val <= threshold:
            return float(s)
    return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/workspace/dataset")
    p.add_argument("--out-dir", default="event_analysis/featcur_s0")
    p.add_argument("--max-scan", type=int, default=24)
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
    ref = out["ytvis"]["by_s"]["0.00"]
    print("\n===== YouTube-VIS s=0 reference =====", flush=True)
    for k, v in ref.items():
        if k == "n":
            continue
        print(f"  {k:10s} {v:.4f}", flush=True)
    print("\n===== smallest s matching that reference =====", flush=True)
    for name, blob in out.items():
        picks = {
            "pr": first_at_least(blob["by_s"], "pr", ref["pr"]),
            "keys": first_at_least(blob["by_s"], "keys", ref["keys"]),
            "sil": first_at_least(blob["by_s"], "sil", ref["sil"]),
            "contrast": first_at_least(blob["by_s"], "contrast", ref["contrast"]),
            "gap2": first_at_least(blob["by_s"], "gap2", ref["gap2"]),
            "obj_cos": first_at_most(blob["by_s"], "obj_cos", ref["obj_cos"]),
        }
        blob["s0_match_yt"] = picks
        print(
            f"  {blob['label']:12s}  "
            + "  ".join(f"{k}={v:.2f}" for k, v in picks.items()),
            flush=True,
        )
        for s in S_GRID:
            row = blob["by_s"][f"{s:.2f}"]
            print(
                f"    s={s:.2f}  pr={row['pr']:.2f} keys={row['keys']:.1f} "
                f"obj_cos={row['obj_cos']:.3f} sil={row['sil']:.3f} "
                f"contrast={row['contrast']:.3f} gap2={row['gap2']:.4f}",
                flush=True,
            )
    path = os.path.join(args.out_dir, "modes.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
