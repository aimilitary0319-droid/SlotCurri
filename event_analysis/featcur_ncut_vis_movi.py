"""Visualize v36 2-way Ncut + region-global leveling on MOVi-C DINO tokens.

Why MOVi-C v36 can collapse: the cut is only 2-way, then P X replaces every
token by a *global* average of its region. Multi-object scenes become ~2 Keys.

Usage (inside the slotcurri image):
  python event_analysis/featcur_ncut_vis_movi.py --data-dir /workspace/dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_dino_vis import denorm_frame, instance_contours, overlay_edges  # noqa: E402
from featcur_dino_vis import pca_basis, pca_map, upsample  # noqa: E402
from featcur_strength_probe import DATASETS, extract_tokens, patch_ids_movi  # noqa: E402
from slotcurri import configuration, data
from slotcurri.modules.encoders import FeatureSmoothing, NcutRelationalLeveling, TimmExtractor


def unique_keys(tokens: torch.Tensor, cos_thr: float = 0.95) -> int:
    """Greedy count of L2-normalized tokens with cosine < thr to all kept means."""
    xn = F.normalize(tokens.float(), dim=-1)
    kept = []
    for i in range(xn.shape[0]):
        if not kept:
            kept.append(xn[i])
            continue
        sim = torch.stack(kept) @ xn[i]
        if float(sim.max()) < cos_thr:
            kept.append(xn[i])
    return len(kept)


def object_region_stats(gt: torch.Tensor, region: torch.Tensor, bg_id: int = 0) -> dict:
    """gt, region: (N,) long/bool. How many GT objects share a 2-way region."""
    obj_ids = [int(i) for i in gt.unique().tolist() if int(i) != bg_id]
    n_obj = len(obj_ids)
    if n_obj == 0:
        return {"n_obj": 0, "n_merged": 0, "n_cut": 0, "pure": 1.0, "maj0": 0, "maj1": 0}
    maj = []
    n_cut = 0
    pures = []
    for oid in obj_ids:
        m = gt == oid
        r = region[m]
        frac = float(r.float().mean())
        pures.append(max(frac, 1.0 - frac))
        maj.append(1 if frac >= 0.5 else 0)
        if 0.15 < frac < 0.85:
            n_cut += 1
    n0 = sum(1 for x in maj if x == 0)
    n1 = sum(1 for x in maj if x == 1)
    n_merged = n_obj - int(n0 > 0) - int(n1 > 0)
    return {
        "n_obj": n_obj,
        "n_merged": max(n_merged, 0),
        "n_cut": n_cut,
        "pure": float(np.mean(pures)) if pures else 1.0,
        "maj0": n0,
        "maj1": n1,
    }


def region_rgb(region_hw: np.ndarray, h: int, w: int) -> np.ndarray:
    cmap = ListedColormap(["#2c7bb6", "#d7191c"])
    img = cmap(region_hw.astype(float))[..., :3]
    return upsample(img, h, w)


@torch.no_grad()
def collect(data_dir, device, max_clips, n_keep):
    cfg = configuration.load_config("configs/slotcurri/movi_c_attnmass_v36.yaml")
    cfg.dataset.val_batch_size = 1
    cfg.dataset.num_val_workers = 0
    dm = data.build(cfg.dataset, data_dir=data_dir)
    dm.setup("validate")
    bb_cfg = cfg.model.encoder.backbone
    kwargs = dict(bb_cfg.get("model_kwargs") or {})
    backbone = (
        TimmExtractor(
            model=bb_cfg.model,
            pretrained=True,
            frozen=True,
            features=bb_cfg.features,
            model_kwargs=kwargs or None,
        )
        .to(device)
        .eval()
    )
    feat_key = bb_cfg.features
    ncut = NcutRelationalLeveling(chunk_size=8, n_iter=16).to(device)
    v33 = FeatureSmoothing(tau=0.1, window=9, window_ref_grid=37)

    rows = []
    picked = []
    n_clips = 0
    print("scanning movi_c val", flush=True)
    for batch in dm.val_dataloader():
        if n_clips >= max_clips:
            break
        if "batch_padding_mask" in batch:
            mask = batch["batch_padding_mask"]
            if torch.is_tensor(mask) and bool(mask.any()):
                n_clips += 1
                continue
        video = batch["video"][0]
        seg = batch["segmentations"][0]
        ti = int(video.shape[0] // 2)
        frame = video[ti].to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            raw = extract_tokens(backbone, frame.unsqueeze(0), feat_key).float()[0]
        grid = int(round(math.sqrt(raw.shape[0])))
        gt = patch_ids_movi(seg[ti].to(device), grid)
        exp = ncut.explain(raw.unsqueeze(0))
        rel = exp["rel"][0]
        region = exp["region"][0]
        sm = v33(raw.unsqueeze(0), mix=0.0)[0]
        st = object_region_stats(gt, region)
        rec = {
            "clip": n_clips,
            "n_obj": st["n_obj"],
            "n_merged": st["n_merged"],
            "n_cut": st["n_cut"],
            "pure": st["pure"],
            "maj0": st["maj0"],
            "maj1": st["maj1"],
            "keys_raw": unique_keys(raw),
            "keys_rel": unique_keys(rel),
            "keys_v33": unique_keys(sm),
        }
        rows.append(rec)
        picked.append(
            {
                **rec,
                "raw": raw.cpu(),
                "rel": rel.cpu(),
                "smooth": sm.cpu(),
                "region": region.cpu(),
                "gt": gt.cpu(),
                "rgb": denorm_frame(frame),
                "edge": instance_contours(seg[ti], bg_id=0),
                "grid": grid,
            }
        )
        n_clips += 1
        if n_clips % 10 == 0:
            print(f"  {n_clips} clips", flush=True)

    del backbone
    torch.cuda.empty_cache()
    picked.sort(key=lambda c: (-c["n_obj"], -c["n_merged"]))
    keep = []
    seen = set()
    for c in picked:
        if c["clip"] in seen:
            continue
        keep.append(c)
        seen.add(c["clip"])
        if len(keep) >= n_keep:
            break
    print("picked", [(c["clip"], c["n_obj"], c["n_merged"], c["keys_rel"]) for c in keep], flush=True)
    return keep, rows


def draw_main(frames, out_path):
    titles = [
        "RGB + GT",
        "2-way Ncut region",
        "PCA raw DINO",
        "PCA v33 w9→6",
        "PCA Ncut + global P",
    ]
    n_row, n_col = len(frames), len(titles)
    fig, axes = plt.subplots(n_row, n_col, figsize=(2.55 * n_col, 2.7 * n_row))
    if n_row == 1:
        axes = np.array([axes])
    for r, fr in enumerate(frames):
        rgb, edge, grid = fr["rgb"], fr["edge"], fr["grid"]
        h, w = rgb.shape[:2]
        basis = pca_basis(fr["raw"])
        proj = (fr["raw"].float() - fr["raw"].float().mean(0)) @ basis
        lo, hi = proj.quantile(0.02, dim=0), proj.quantile(0.98, dim=0)
        pca_raw = overlay_edges(upsample(pca_map(fr["raw"], basis, lo, hi, grid), h, w), edge, (1, 1, 1))
        pca_v33 = overlay_edges(upsample(pca_map(fr["smooth"], basis, lo, hi, grid), h, w), edge, (1, 1, 1))
        pca_rel = overlay_edges(upsample(pca_map(fr["rel"], basis, lo, hi, grid), h, w), edge, (1, 1, 1))
        reg = overlay_edges(
            region_rgb(fr["region"].numpy().reshape(grid, grid), h, w),
            edge,
            (1.0, 1.0, 0.15),
        )
        panels = [
            overlay_edges(rgb, edge),
            reg,
            pca_raw,
            pca_v33,
            pca_rel,
        ]
        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(
                    f"clip {fr['clip']}\n{fr['n_obj']} obj  merge {fr['n_merged']}\n"
                    f"keys {fr['keys_raw']}→{fr['keys_rel']}",
                    fontsize=8,
                )
            if c == 4:
                ax.set_xlabel(f"unique keys={fr['keys_rel']}", fontsize=8)
    fig.suptitle(
        "MOVi-C ViT-S/14 24×24. Same PCA basis (raw). "
        "Ncut is 2-way then global row-normalize P inside each region (mix=0). "
        "Yellow = GT instance edges. v33 column is windowed cosine, not Ncut.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def draw_mix(frames, out_path):
    mixes = [0.0, 0.5, 0.69, 1.0]
    labels = ["s=0  X^rel", "s=0.5", "s=0.69  ~30k", "s=1  raw"]
    n_row, n_col = len(frames), 1 + len(mixes)
    fig, axes = plt.subplots(n_row, n_col, figsize=(2.5 * n_col, 2.6 * n_row))
    if n_row == 1:
        axes = np.array([axes])
    for r, fr in enumerate(frames[: min(4, len(frames))]):
        rgb, edge, grid = fr["rgb"], fr["edge"], fr["grid"]
        h, w = rgb.shape[:2]
        basis = pca_basis(fr["raw"])
        proj = (fr["raw"].float() - fr["raw"].float().mean(0)) @ basis
        lo, hi = proj.quantile(0.02, dim=0), proj.quantile(0.98, dim=0)
        ax = axes[r, 0]
        ax.imshow(overlay_edges(rgb, edge))
        ax.set_ylabel(f"clip {fr['clip']}  {fr['n_obj']} obj", fontsize=8)
        if r == 0:
            ax.set_title("RGB + GT", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for c, (s, lab) in enumerate(zip(mixes, labels), start=1):
            feat = (1.0 - s) * fr["rel"] + s * fr["raw"]
            pca = overlay_edges(upsample(pca_map(feat, basis, lo, hi, grid), h, w), edge, (1, 1, 1))
            ax = axes[r, c]
            ax.imshow(pca)
            nkey = unique_keys(feat)
            if r == 0:
                ax.set_title(lab, fontsize=9)
            ax.set_xlabel(f"keys={nkey}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "Curriculum mix X^bind = (1-s) X^rel + s X. "
        "s=0.69 is MOVi-C v36 around step 31k. Eval uses s=1 (raw Keys).",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--max-clips", type=int, default=40)
    ap.add_argument("--n-keep", type=int, default=6)
    ap.add_argument("--out-dir", default="event_analysis/ncut_vis_movi")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    frames, rows = collect(args.data_dir, device, args.max_clips, args.n_keep)
    draw_main(frames, os.path.join(args.out_dir, "ncut_global_pca.png"))
    draw_mix(frames, os.path.join(args.out_dir, "ncut_mix_pca.png"))
    csv_path = os.path.join(args.out_dir, "ncut_stats.csv")
    keys = [
        "clip", "n_obj", "n_merged", "n_cut", "pure", "maj0", "maj1",
        "keys_raw", "keys_rel", "keys_v33",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    n = len(rows)
    summary = {
        "n_frames": n,
        "mean_n_obj": float(np.mean([r["n_obj"] for r in rows])),
        "mean_n_merged": float(np.mean([r["n_merged"] for r in rows])),
        "mean_keys_raw": float(np.mean([r["keys_raw"] for r in rows])),
        "mean_keys_rel": float(np.mean([r["keys_rel"] for r in rows])),
        "mean_keys_v33": float(np.mean([r["keys_v33"] for r in rows])),
        "frac_keys_rel_le2": float(np.mean([r["keys_rel"] <= 2 for r in rows])),
        "frac_merged_any": float(np.mean([r["n_merged"] > 0 for r in rows])),
    }
    with open(os.path.join(args.out_dir, "ncut_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
