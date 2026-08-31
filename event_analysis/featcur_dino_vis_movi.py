"""MOVi-C DINO vis of the two v33 strength configs, with grid-scaled windows.

YTVIS design grid is 37x37. MOVi-C is 24x24, so
  w=3 -> 2,  w=5 -> 3,  w=9 -> 6
when window_ref_grid=37.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_dino_vis import object_crop, overlay_edges, pca_basis, pca_map, upsample  # noqa: E402
from featcur_dino_vis import denorm_frame, instance_contours  # noqa: E402
from featcur_dino_vis_parts import part_axis, part_cos  # noqa: E402
from featcur_strength_probe import (  # noqa: E402
    DATASETS,
    apply_smooth,
    extract_tokens,
    kmeans2,
    patch_ids_movi,
)
from slotcurri import configuration, data
from slotcurri.modules.encoders import FeatureSmoothing, TimmExtractor

# yaml window, steps, tau, and whether to scale from the 37-grid.
SETTINGS = [
    {"name": "raw", "tau": None, "steps": 0, "window": None, "scale": True, "pick": False},
    {"name": "v33 s1 w9→6", "tau": 0.1, "steps": 1, "window": 9, "scale": True, "pick": False},
    {"name": "s3 w3 abs", "tau": 0.2, "steps": 3, "window": 3, "scale": False, "pick": False},
    {"name": "s3 w3→2 (rec)", "tau": 0.2, "steps": 3, "window": 3, "scale": True, "pick": True},
    {"name": "s5 w5 abs", "tau": 0.1, "steps": 5, "window": 5, "scale": False, "pick": False},
    {"name": "s5 w5→3", "tau": 0.1, "steps": 5, "window": 5, "scale": True, "pick": False},
]


def _ref(scale: bool):
    return 37 if scale else None


@torch.no_grad()
def collect_movi(data_dir, device, max_clips, min_patches, n_keep=4):
    cfg = configuration.load_config(DATASETS["movi_c"])
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
    cands = []
    n_clips = 0
    print("scanning movi_c", flush=True)
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
        obj_ids = [int(i) for i in gt.unique().tolist() if int(i) != 0]
        xn = F.normalize(raw.float(), dim=-1)
        best = None
        for oid in obj_ids:
            idx = (gt == oid).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() < min_patches:
                continue
            lab = kmeans2(xn[idx])
            if lab is None:
                continue
            ia, ib = idx[lab == 0], idx[lab == 1]
            if ia.numel() < 4 or ib.numel() < 4:
                continue
            cand = (int(idx.numel()), oid, ia.cpu(), ib.cpu())
            if best is None or cand[0] > best[0]:
                best = cand
        if best is not None:
            cands.append(
                {
                    "clip": n_clips,
                    "raw": raw.cpu(),
                    "rgb": denorm_frame(frame),
                    "edge": instance_contours(seg[ti], bg_id=0),
                    "gt_grid": gt.cpu().reshape(grid, grid),
                    "oid": best[1],
                    "ia": best[2],
                    "ib": best[3],
                    "n": best[0],
                    "n_obj": len(obj_ids),
                    "part_cos0": part_cos(raw, best[2].to(device), best[3].to(device)),
                }
            )
        n_clips += 1
        if n_clips % 20 == 0:
            print(f"  {n_clips} clips / {len(cands)} usable", flush=True)
    del backbone
    torch.cuda.empty_cache()
    cands.sort(key=lambda c: (-c["n_obj"], c["part_cos0"]))
    picked = []
    seen = set()
    for c in cands:
        if c["clip"] in seen:
            continue
        picked.append(c)
        seen.add(c["clip"])
        if len(picked) >= n_keep:
            break
    print("picked", [(c["clip"], c["n_obj"], c["n"], c["part_cos0"]) for c in picked], flush=True)
    return picked


@torch.no_grad()
def attach(fr, device):
    raw = fr["raw"].to(device)
    ia, ib = fr["ia"].to(device), fr["ib"].to(device)
    grid = int(round(math.sqrt(raw.shape[0])))
    axis = part_axis(raw, ia, ib)
    mid = 0.5 * (raw[ia].float().mean(0) + raw[ib].float().mean(0))
    obj = torch.cat([ia, ib])
    raw_proj = ((raw.float() - mid) @ axis).reshape(grid, grid)
    vmax = max(float(raw_proj[obj // grid, obj % grid].abs().quantile(0.95).item()), 1e-3)
    basis = pca_basis(raw)
    proj = (raw.float() - raw.float().mean(0)) @ basis
    lo, hi = proj.quantile(0.02, dim=0), proj.quantile(0.98, dim=0)

    part_maps, pca_maps, scores, weffs = {}, {}, {}, {}
    for st in SETTINGS:
        if st["tau"] is None:
            feat = raw
            weffs[st["name"]] = None
        else:
            feat = apply_smooth(
                raw,
                float(st["tau"]),
                st["window"],
                int(st["steps"]),
                window_ref_grid=_ref(st["scale"]),
            )
            weffs[st["name"]] = FeatureSmoothing(
                tau=st["tau"],
                window=st["window"],
                window_ref_grid=_ref(st["scale"]),
            ).effective_window(raw.shape[0])
        part_maps[st["name"]] = ((feat.float() - mid) @ axis).reshape(grid, grid).cpu().numpy()
        pca_maps[st["name"]] = pca_map(feat, basis, lo, hi, grid)
        scores[st["name"]] = part_cos(feat, ia, ib)
    fr.update(part_maps=part_maps, pca_maps=pca_maps, scores=scores, weffs=weffs, vmax=vmax)
    return fr


def draw_part(frames, out_path):
    n_row, n_col = len(frames), 1 + len(SETTINGS)
    fig, axes = plt.subplots(n_row, n_col, figsize=(2.55 * n_col, 2.95 * n_row))
    if n_row == 1:
        axes = axes[None, :]
    cmap = plt.get_cmap("coolwarm")
    for r, fr in enumerate(frames):
        rgb = fr["rgb"]
        h, w = rgb.shape[:2]
        dummy = np.zeros((h, w, 3))
        crop = object_crop(rgb / 255.0, dummy, fr["gt_grid"], fr["oid"], pad=4)
        y0, y1, x0, x1 = (0, h, 0, w) if crop is None else crop[2]
        ax = axes[r, 0]
        ax.imshow(rgb[y0:y1, x0:x1])
        ax.set_ylabel(f"clip {fr['clip']}  {fr['n_obj']} obj", fontsize=8)
        ax.set_title("RGB", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        norm = TwoSlopeNorm(vmin=-fr["vmax"], vcenter=0.0, vmax=fr["vmax"])
        for c, st in enumerate(SETTINGS, start=1):
            m = upsample(fr["part_maps"][st["name"]][..., None], h, w)[y0:y1, x0:x1, 0]
            ax = axes[r, c]
            im = ax.imshow(m, cmap=cmap, norm=norm)
            we = fr["weffs"][st["name"]]
            wtag = "" if we is None else f"  weff={we}"
            ax.set_title(
                f"{st['name']}{wtag}\npart-cos {fr['scores'][st['name']]:.2f}",
                fontsize=7,
                color="#c0392b" if st["pick"] else "black",
                fontweight="bold" if st["pick"] else "normal",
            )
            if st["pick"]:
                for spine in ax.spines.values():
                    spine.set_color("#c0392b")
                    spine.set_linewidth(2.2)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == n_col - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "MOVi-C 24×24. yaml window is on the 37-grid; weff = round(w·24/37). "
        "Red/blue = object 2-means part axis. Rec = τ=0.2, n_steps=3, w 3→2.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def draw_pca(frames, out_path):
    n_row, n_col = len(frames), 1 + len(SETTINGS)
    fig, axes = plt.subplots(n_row, n_col, figsize=(2.4 * n_col, 2.4 * n_row))
    if n_row == 1:
        axes = axes[None, :]
    for r, fr in enumerate(frames):
        rgb, edge = fr["rgb"], fr["edge"]
        h, w = rgb.shape[:2]
        ax = axes[r, 0]
        ax.imshow(overlay_edges(rgb, edge))
        ax.set_ylabel(f"clip {fr['clip']}", fontsize=8)
        ax.set_title("RGB + GT", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for c, st in enumerate(SETTINGS, start=1):
            pca = overlay_edges(upsample(fr["pca_maps"][st["name"]], h, w), edge, color=(1, 1, 1))
            ax = axes[r, c]
            ax.imshow(pca)
            if r == 0:
                ax.set_title(st["name"], fontsize=8, color="#c0392b" if st["pick"] else "black",
                             fontweight="bold" if st["pick"] else "normal")
            if st["pick"]:
                for spine in ax.spines.values():
                    spine.set_color("#c0392b")
                    spine.set_linewidth(2.2)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("MOVi-C PCA (fit on raw). Scaled windows keep relative neighbourhood = YTVIS.", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--max-clips", type=int, default=40)
    ap.add_argument("--out-dir", default="event_analysis/featcur_mix_sweep")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    frames = [attach(fr, device) for fr in collect_movi(args.data_dir, device, args.max_clips, 16)]
    for fr in frames:
        print(
            f" clip {fr['clip']}: "
            + ", ".join(f"{st['name']}={fr['scores'][st['name']]:.3f}" for st in SETTINGS),
            flush=True,
        )
    draw_part(frames, os.path.join(args.out_dir, "dino_movi_part_axis.png"))
    draw_pca(frames, os.path.join(args.out_dir, "dino_movi_pca.png"))
    print("wrote dino_movi_part_axis.png\nwrote dino_movi_pca.png", flush=True)


if __name__ == "__main__":
    main()
