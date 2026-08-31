"""Measure how hard FeatureSmoothing mixes parts vs objects on real DINO tokens.

The v33 operator is one-step P = softmax(cos/tau) with tau=0.1, window=9.
Claim under test: that strength flattens *within* DINO clusters but does not merge
head/torso-like parts of one object. This script measures that on val frames.

Per object with enough patches:
  - parts = 2-means on RAW (L2-normalized) tokens inside the GT object
    (assignment is frozen; we only ask whether smoothing collapses the two means)
  - spatial halves = split along the object's principal axis in the patch grid
    (a geometric head/torso proxy)
  - first-step affinity mass from part A onto: self / other-part / other-objects / bg
  - within-object and within-part variance ratios (after / before)

Usage (inside the slotcurri container):
  python event_analysis/featcur_strength_probe.py --data-dir /workspace/dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from slotcurri import configuration, data
from slotcurri.modules.encoders import FeatureSmoothing, TimmExtractor


DATASETS = {
    "ytvis": "configs/slotcurri/ytvis2021_attnmass_v33.yaml",
    "movi_c": "configs/slotcurri/movi_c_attnmass_v33.yaml",
}

SETTINGS = [
    {"name": "raw", "tau": None, "steps": 0, "window": None},
    {"name": "v33_t0.1_s1_w9", "tau": 0.1, "steps": 1, "window": 9},
    {"name": "t0.05_s1_w9", "tau": 0.05, "steps": 1, "window": 9},
    {"name": "t0.2_s1_w9", "tau": 0.2, "steps": 1, "window": 9},
    {"name": "t0.3_s1_w9", "tau": 0.3, "steps": 1, "window": 9},
    {"name": "t0.5_s1_w9", "tau": 0.5, "steps": 1, "window": 9},
    {"name": "t0.1_s2_w9", "tau": 0.1, "steps": 2, "window": 9},
    {"name": "t0.1_s3_w9", "tau": 0.1, "steps": 3, "window": 9},
    {"name": "t0.1_s1_glob", "tau": 0.1, "steps": 1, "window": None},
]


def _mean_var(x: torch.Tensor) -> float:
    if x.numel() == 0 or x.shape[0] < 2:
        return float("nan")
    return float(x.var(dim=0, unbiased=False).mean().item())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.float(), dim=0)
    b = F.normalize(b.float(), dim=0)
    return float((a * b).sum().clamp(-1.0, 1.0).item())


def kmeans2(xn: torch.Tensor, n_iter: int = 20) -> Optional[torch.Tensor]:
    """2-means on L2-normalized tokens. Returns None if a cluster is empty."""
    n = xn.shape[0]
    if n < 8:
        return None
    sim = xn @ xn.t()
    j = int(sim[0].argmin().item())
    i = int(sim[j].argmin().item())
    if i == j:
        return None
    c0, c1 = xn[i], xn[j]
    lab = None
    for _ in range(n_iter):
        d0 = (xn * c0).sum(dim=-1)
        d1 = (xn * c1).sum(dim=-1)
        lab = (d1 > d0)
        n1 = int(lab.sum().item())
        n0 = n - n1
        if n0 < 2 or n1 < 2:
            return None
        c0 = F.normalize(xn[~lab].mean(dim=0), dim=0)
        c1 = F.normalize(xn[lab].mean(dim=0), dim=0)
    return lab.long()


def spatial_halves(ys: torch.Tensor, xs: torch.Tensor) -> Optional[torch.Tensor]:
    """Split object patches along the first principal axis of their grid coords."""
    if ys.numel() < 8:
        return None
    coords = torch.stack([ys.float(), xs.float()], dim=1)
    coords = coords - coords.mean(dim=0, keepdim=True)
    cov = coords.t() @ coords / max(coords.shape[0] - 1, 1)
    _, evecs = torch.linalg.eigh(cov)
    axis = evecs[:, -1]
    proj = coords @ axis
    lab = (proj > proj.median()).long()
    if int(lab.sum()) < 2 or int((1 - lab).sum()) < 2:
        return None
    return lab


def patch_ids_ytvis(seg: torch.Tensor, grid: int) -> torch.Tensor:
    """(I, H, W) instance binaries -> (G*G,) ids, -1 background."""
    if seg.ndim != 3:
        raise ValueError(f"expected (I,H,W), got {tuple(seg.shape)}")
    pooled = F.adaptive_avg_pool2d(seg.float().unsqueeze(0), (grid, grid))[0]
    cov, idx = pooled.flatten(1).max(dim=0)
    out = idx.clone()
    out[cov < 0.5] = -1
    return out


def patch_ids_movi(seg: torch.Tensor, grid: int) -> torch.Tensor:
    """(C, H, W) one-hot, channel 0 = background -> (G*G,) ids."""
    pooled = F.adaptive_avg_pool2d(seg.float().unsqueeze(0), (grid, grid))[0]
    return pooled.flatten(1).argmax(dim=0)


@torch.no_grad()
def affinity_p(x: torch.Tensor, tau: float, window: Optional[int]) -> torch.Tensor:
    """x: (F, D) -> P: (F, F)."""
    sm = FeatureSmoothing(tau=tau, window=window, chunk_size=1)
    mask = sm._window_mask(x.shape[0], x.device)
    xn = F.normalize(x.float(), dim=-1)
    logits = xn @ xn.t() / float(tau)
    if mask is not None:
        logits = logits.masked_fill(mask, float("-inf"))
    return logits.softmax(dim=-1)


@torch.no_grad()
def apply_smooth(
    x: torch.Tensor,
    tau: float,
    window: Optional[int],
    steps: int,
    window_ref_grid: Optional[int] = 37,
) -> torch.Tensor:
    if steps <= 0:
        return x
    sm = FeatureSmoothing(
        tau=tau, window=window, chunk_size=1, window_ref_grid=window_ref_grid
    )
    y = x
    for _ in range(steps):
        y = sm._smooth(y.unsqueeze(0))[0]
    return y


def mass_groups(p_row_sum: torch.Tensor, groups: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out = {}
    for name, idx in groups.items():
        if idx.numel() == 0:
            out[name] = float("nan")
        else:
            out[name] = float(p_row_sum.index_select(0, idx).sum().item())
    return out


def analyze_frame(
    raw: torch.Tensor,
    gt: torch.Tensor,
    settings: List[dict],
    bg_id: int,
    min_patches: int,
    contact_radius: int,
) -> List[dict]:
    """raw (F,D), gt (F,) int ids. Returns a list of per-object-per-setting records."""
    f, dim = raw.shape
    grid = int(round(math.sqrt(f)))
    ys = torch.arange(f, device=raw.device) // grid
    xs = torch.arange(f, device=raw.device) % grid
    xn_raw = F.normalize(raw.float(), dim=-1)

    obj_ids = [int(i) for i in gt.unique().tolist() if int(i) != bg_id]
    obj_idx = {oid: (gt == oid).nonzero(as_tuple=False).squeeze(-1) for oid in obj_ids}
    obj_ids = [oid for oid in obj_ids if obj_idx[oid].numel() >= min_patches]
    if not obj_ids:
        return []

    # contact pairs (patch-grid Chebyshev)
    contact = set()
    for a, ia in ((o, obj_idx[o]) for o in obj_ids):
        ya, xa = ys[ia], xs[ia]
        for b, ib in ((o, obj_idx[o]) for o in obj_ids):
            if b <= a:
                continue
            yb, xb = ys[ib], xs[ib]
            cheb = torch.maximum(
                (ya[:, None] - yb[None, :]).abs(), (xa[:, None] - xb[None, :]).abs()
            )
            if int(cheb.min().item()) <= contact_radius:
                contact.add((a, b))

    parts = {}
    halves = {}
    for oid in obj_ids:
        idx = obj_idx[oid]
        lab = kmeans2(xn_raw[idx])
        if lab is not None and int((lab == 0).sum()) >= 4 and int((lab == 1).sum()) >= 4:
            parts[oid] = (idx[lab == 0], idx[lab == 1])
        hlab = spatial_halves(ys[idx], xs[idx])
        if hlab is not None and int((hlab == 0).sum()) >= 4 and int((hlab == 1).sum()) >= 4:
            halves[oid] = (idx[hlab == 0], idx[hlab == 1])

    # First-step P per (tau, window). Multi-step features are chained so
    # steps=1..K for the same graph costs K smooths, not 1+2+...+K.
    p_cache: Dict[Tuple[float, Optional[int]], torch.Tensor] = {}
    feat_cache: Dict[Tuple[float, Optional[int], int], torch.Tensor] = {}
    groups: Dict[Tuple[float, Optional[int]], List[int]] = {}
    for st in settings:
        if st["tau"] is None:
            continue
        key = (float(st["tau"]), st["window"])
        groups.setdefault(key, []).append(int(st["steps"]))
    for key, step_list in groups.items():
        p_cache[key] = affinity_p(raw, key[0], key[1])
        y = raw
        for step in range(1, max(step_list) + 1):
            y = apply_smooth(y, key[0], key[1], 1)
            if step in step_list:
                feat_cache[(key[0], key[1], step)] = y

    records = []
    for st in settings:
        if st["tau"] is None:
            xsmooth = raw
            p = None
        else:
            xsmooth = feat_cache[(float(st["tau"]), st["window"], int(st["steps"]))]
            p = p_cache[(float(st["tau"]), st["window"])]

        means = {oid: raw[obj_idx[oid]].mean(0) for oid in obj_ids}
        means_s = {oid: xsmooth[obj_idx[oid]].mean(0) for oid in obj_ids}

        pair_cos_raw = []
        pair_cos_sm = []
        contact_cos_raw = []
        contact_cos_sm = []
        for i, a in enumerate(obj_ids):
            for b in obj_ids[i + 1 :]:
                cr = _cos(means[a], means[b])
                cs = _cos(means_s[a], means_s[b])
                pair_cos_raw.append(cr)
                pair_cos_sm.append(cs)
                if (min(a, b), max(a, b)) in contact:
                    contact_cos_raw.append(cr)
                    contact_cos_sm.append(cs)

        for oid in obj_ids:
            idx = obj_idx[oid]
            rec = {
                "setting": st["name"],
                "obj_id": oid,
                "n_patches": int(idx.numel()),
                "var_obj_raw": _mean_var(raw[idx]),
                "var_obj_sm": _mean_var(xsmooth[idx]),
                "pair_cos_raw": float(np.mean(pair_cos_raw)) if pair_cos_raw else float("nan"),
                "pair_cos_sm": float(np.mean(pair_cos_sm)) if pair_cos_sm else float("nan"),
                "contact_cos_raw": (
                    float(np.mean(contact_cos_raw)) if contact_cos_raw else float("nan")
                ),
                "contact_cos_sm": (
                    float(np.mean(contact_cos_sm)) if contact_cos_sm else float("nan")
                ),
            }
            rec["var_obj_ratio"] = (
                rec["var_obj_sm"] / rec["var_obj_raw"]
                if rec["var_obj_raw"] and rec["var_obj_raw"] > 1e-8
                else float("nan")
            )

            for tag, part_map in (("kmeans", parts), ("spatial", halves)):
                if oid not in part_map:
                    rec[f"{tag}_cos_raw"] = float("nan")
                    rec[f"{tag}_cos_sm"] = float("nan")
                    rec[f"{tag}_mix"] = float("nan")
                    rec[f"{tag}_var_part_ratio"] = float("nan")
                    rec[f"p_{tag}_self"] = float("nan")
                    rec[f"p_{tag}_other_part"] = float("nan")
                    rec[f"p_{tag}_other_obj"] = float("nan")
                    rec[f"p_{tag}_bg"] = float("nan")
                    continue
                ia, ib = part_map[oid]
                cr = _cos(raw[ia].mean(0), raw[ib].mean(0))
                cs = _cos(xsmooth[ia].mean(0), xsmooth[ib].mean(0))
                rec[f"{tag}_cos_raw"] = cr
                rec[f"{tag}_cos_sm"] = cs
                rec[f"{tag}_mix"] = (cs - cr) / max(1.0 - cr, 1e-6)
                vr_a = _mean_var(xsmooth[ia]) / max(_mean_var(raw[ia]), 1e-8)
                vr_b = _mean_var(xsmooth[ib]) / max(_mean_var(raw[ib]), 1e-8)
                rec[f"{tag}_var_part_ratio"] = 0.5 * (vr_a + vr_b)

                if p is None:
                    rec[f"p_{tag}_self"] = float("nan")
                    rec[f"p_{tag}_other_part"] = float("nan")
                    rec[f"p_{tag}_other_obj"] = float("nan")
                    rec[f"p_{tag}_bg"] = float("nan")
                else:
                    others = torch.cat(
                        [obj_idx[o] for o in obj_ids if o != oid], dim=0
                    ) if len(obj_ids) > 1 else torch.empty(0, dtype=torch.long, device=raw.device)
                    bg = (gt == bg_id).nonzero(as_tuple=False).squeeze(-1)
                    masses = []
                    for src, other_part in ((ia, ib), (ib, ia)):
                        prow = p.index_select(0, src).mean(dim=0)
                        masses.append(
                            mass_groups(
                                prow,
                                {
                                    "self": src,
                                    "other_part": other_part,
                                    "other_obj": others,
                                    "bg": bg,
                                },
                            )
                        )
                    rec[f"p_{tag}_self"] = 0.5 * (masses[0]["self"] + masses[1]["self"])
                    rec[f"p_{tag}_other_part"] = 0.5 * (
                        masses[0]["other_part"] + masses[1]["other_part"]
                    )
                    rec[f"p_{tag}_other_obj"] = 0.5 * (
                        masses[0]["other_obj"] + masses[1]["other_obj"]
                    )
                    rec[f"p_{tag}_bg"] = 0.5 * (masses[0]["bg"] + masses[1]["bg"])

            records.append(rec)
    return records


def extract_tokens(backbone: torch.nn.Module, frames: torch.Tensor, feat_key: str) -> torch.Tensor:
    """frames (N,C,H,W) -> tokens (N,F,D)."""
    out = backbone(frames)
    if isinstance(out, dict):
        return out[feat_key]
    return out


def run_dataset(
    name: str,
    cfg_path: str,
    data_dir: str,
    device: torch.device,
    max_clips: int,
    frames_per_clip: int,
    min_patches: int,
    settings: Optional[List[dict]] = None,
) -> List[dict]:
    settings = settings if settings is not None else SETTINGS
    print(f"\n=== {name} ===", flush=True)
    cfg = configuration.load_config(cfg_path)
    cfg.dataset.val_batch_size = 1
    cfg.dataset.num_val_workers = 0
    dm = data.build(cfg.dataset, data_dir=data_dir)
    dm.setup("validate")

    bb_cfg = cfg.model.encoder.backbone
    kwargs = dict(bb_cfg.get("model_kwargs") or {})
    backbone = TimmExtractor(
        model=bb_cfg.model,
        pretrained=True,
        frozen=True,
        features=bb_cfg.features,
        model_kwargs=kwargs or None,
    ).to(device).eval()
    feat_key = bb_cfg.features
    bg_id = 0 if name == "movi_c" else -1

    rows: List[dict] = []
    n_clips = 0
    n_frames = 0
    for batch in dm.val_dataloader():
        if n_clips >= max_clips:
            break
        if "batch_padding_mask" in batch:
            mask = batch["batch_padding_mask"]
            if torch.is_tensor(mask) and bool(mask.any()):
                continue
        video = batch["video"]
        seg = batch["segmentations"]
        if video.ndim != 5:
            raise RuntimeError(f"expected video (B,T,C,H,W), got {tuple(video.shape)}")
        video = video[0]
        seg = seg[0]
        t = video.shape[0]
        if t <= 0:
            continue
        if t == 1:
            t_idx = [0]
        else:
            picks = torch.linspace(0, t - 1, steps=min(frames_per_clip, t)).round().long().tolist()
            t_idx = sorted(set(int(i) for i in picks))

        frames = video[t_idx].to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            tokens = extract_tokens(backbone, frames, feat_key).float()
        grid = int(round(math.sqrt(tokens.shape[1])))
        if grid * grid != tokens.shape[1]:
            raise RuntimeError(f"{name}: non-square tokens {tokens.shape[1]}")

        for k, ti in enumerate(t_idx):
            raw = tokens[k]
            s = seg[ti]
            if name == "ytvis":
                gt = patch_ids_ytvis(s.to(device), grid)
            else:
                gt = patch_ids_movi(s.to(device), grid)
            recs = analyze_frame(
                raw, gt, settings, bg_id=bg_id, min_patches=min_patches, contact_radius=9
            )
            for r in recs:
                r["dataset"] = name
                r["clip"] = n_clips
                r["frame"] = int(ti)
            rows.extend(recs)
            n_frames += 1
        n_clips += 1
        if n_clips % 10 == 0:
            print(f"  {name}: {n_clips} clips / {n_frames} frames / {len(rows)} obj-rows", flush=True)

    print(f"  done {name}: {n_clips} clips, {n_frames} frames, {len(rows)} obj-rows", flush=True)
    del backbone
    torch.cuda.empty_cache()
    return rows


def _nanmean(xs):
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def aggregate(rows: List[dict]) -> List[dict]:
    keys = [
        "kmeans_cos_raw",
        "kmeans_cos_sm",
        "kmeans_mix",
        "kmeans_var_part_ratio",
        "p_kmeans_self",
        "p_kmeans_other_part",
        "p_kmeans_other_obj",
        "p_kmeans_bg",
        "spatial_cos_raw",
        "spatial_cos_sm",
        "spatial_mix",
        "spatial_var_part_ratio",
        "p_spatial_self",
        "p_spatial_other_part",
        "p_spatial_other_obj",
        "p_spatial_bg",
        "var_obj_ratio",
        "pair_cos_raw",
        "pair_cos_sm",
        "contact_cos_raw",
        "contact_cos_sm",
    ]
    out = []
    datasets = sorted(set(r["dataset"] for r in rows))
    settings = [s["name"] for s in SETTINGS]
    for ds in datasets:
        for st in settings:
            sub = [r for r in rows if r["dataset"] == ds and r["setting"] == st]
            if not sub:
                continue
            rec = {
                "dataset": ds,
                "setting": st,
                "n_objects": len(sub),
                "n_clips": len(set((r["clip"], r["frame"]) for r in sub)),
            }
            for k in keys:
                rec[k] = _nanmean([r[k] for r in sub])
            out.append(rec)
    return out


def plot_summary(agg: List[dict], out_png: str) -> None:
    datasets = sorted(set(r["dataset"] for r in agg))
    settings = [s["name"] for s in SETTINGS if s["name"] != "raw"]
    fig, axes = plt.subplots(len(datasets), 3, figsize=(16, 4.2 * len(datasets)), squeeze=False)
    for row, ds in enumerate(datasets):
        by = {r["setting"]: r for r in agg if r["dataset"] == ds}
        raw = by.get("raw", {})
        xs = np.arange(len(settings))

        ax = axes[row, 0]
        raw_c = [raw.get("kmeans_cos_raw", float("nan"))] * len(settings)
        sm_c = [by.get(s, {}).get("kmeans_cos_sm", float("nan")) for s in settings]
        ax.plot(xs, raw_c, "--", color="0.5", label="raw part-cos")
        ax.plot(xs, sm_c, "o-", label="smoothed part-cos")
        ax.set_xticks(xs)
        ax.set_xticklabels(settings, rotation=40, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("cosine(part A, part B)")
        ax.set_title(f"{ds}: k-means parts (1 = merged)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        width = 0.25
        p_part = [by.get(s, {}).get("p_kmeans_other_part", float("nan")) for s in settings]
        p_obj = [by.get(s, {}).get("p_kmeans_other_obj", float("nan")) for s in settings]
        p_bg = [by.get(s, {}).get("p_kmeans_bg", float("nan")) for s in settings]
        ax.bar(xs - width, p_part, width, label="→ other part")
        ax.bar(xs, p_obj, width, label="→ other object")
        ax.bar(xs + width, p_bg, width, label="→ background")
        ax.set_xticks(xs)
        ax.set_xticklabels(settings, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("first-step affinity mass")
        ax.set_title(f"{ds}: where a part-token attends")
        ax.legend(fontsize=8)
        ax.set_ylim(0.0, 0.6)
        ax.grid(True, axis="y", alpha=0.3)

        ax = axes[row, 2]
        v_obj = [by.get(s, {}).get("var_obj_ratio", float("nan")) for s in settings]
        v_part = [by.get(s, {}).get("kmeans_var_part_ratio", float("nan")) for s in settings]
        ax.plot(xs, v_obj, "o-", label="within-object var ratio")
        ax.plot(xs, v_part, "s-", label="within-part var ratio")
        ax.axhline(1.0, color="0.5", ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(settings, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("var(after) / var(before)")
        ax.set_title(f"{ds}: variance collapse")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--max-clips", type=int, default=50)
    ap.add_argument("--frames-per-clip", type=int, default=2)
    ap.add_argument("--min-patches", type=int, default=16)
    ap.add_argument("--datasets", nargs="+", default=["ytvis", "movi_c"])
    ap.add_argument("--out-dir", default="event_analysis/featcur_strength")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device={device} clips={args.max_clips} frames/clip={args.frames_per_clip}", flush=True)

    all_rows: List[dict] = []
    for ds in args.datasets:
        if ds not in DATASETS:
            raise ValueError(f"unknown dataset {ds}")
        all_rows.extend(
            run_dataset(
                ds,
                DATASETS[ds],
                args.data_dir,
                device,
                args.max_clips,
                args.frames_per_clip,
                args.min_patches,
            )
        )

    raw_path = os.path.join(args.out_dir, "per_object.csv")
    if all_rows:
        with open(raw_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    agg = aggregate(all_rows)
    agg_path = os.path.join(args.out_dir, "summary.csv")
    if agg:
        with open(agg_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
            w.writeheader()
            w.writerows(agg)
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(agg, f, indent=2)

    png = os.path.join(args.out_dir, "featcur_strength.png")
    if agg:
        plot_summary(agg, png)

    print("\n===== SUMMARY =====")
    header = (
        f"{'dataset':8} {'setting':18} {'n':>5} "
        f"{'part_cos0':>9} {'part_cos1':>9} {'part_mix':>8} "
        f"{'P→part':>7} {'P→obj':>7} {'P→bg':>7} "
        f"{'var_obj':>7} {'var_part':>8} {'pair_cos1':>9}"
    )
    print(header)
    for r in agg:
        print(
            f"{r['dataset']:8} {r['setting']:18} {r['n_objects']:5d} "
            f"{r['kmeans_cos_raw']:9.3f} {r['kmeans_cos_sm']:9.3f} {r['kmeans_mix']:8.3f} "
            f"{r['p_kmeans_other_part']:7.3f} {r['p_kmeans_other_obj']:7.3f} {r['p_kmeans_bg']:7.3f} "
            f"{r['var_obj_ratio']:7.3f} {r['kmeans_var_part_ratio']:8.3f} {r['pair_cos_sm']:9.3f}"
        )
    print(f"\nwrote {agg_path}\nwrote {png}")


if __name__ == "__main__":
    main()
