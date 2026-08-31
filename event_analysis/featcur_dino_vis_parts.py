"""Show the part-split axis, not object-vs-background PCA.

The earlier grid looked similar across settings because a 3D PCA of the whole
frame is dominated by object/background. This script freezes the raw 2-means
head/torso direction and colors every patch by its projection on that axis.
If parts merge, the map goes from red/blue to flat.

Usage (inside the slotcurri container):
  python event_analysis/featcur_dino_vis_parts.py --data-dir /workspace/dataset
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
from featcur_dino_vis import (  # noqa: E402
    denorm_frame,
    instance_contours,
    object_crop,
    upsample,
)
from featcur_strength_probe import (  # noqa: E402
    DATASETS,
    apply_smooth,
    extract_tokens,
    kmeans2,
    patch_ids_ytvis,
)
from slotcurri import configuration, data
from slotcurri.modules.encoders import TimmExtractor


# Same YTVIS clips the previous figure used for the head/torso crops.
TARGET_CLIPS = {
    "ytvis": [4, 50, 23, 18],
}

SETTINGS = [
    {"name": "raw", "tau": None, "steps": 0, "window": None, "pick": False},
    {"name": "v33 s1w9", "tau": 0.1, "steps": 1, "window": 9, "pick": False},
    {"name": "s5 w5 τ.1", "tau": 0.1, "steps": 5, "window": 5, "pick": False},
    {"name": "s3 w3 τ.2 (rec)", "tau": 0.2, "steps": 3, "window": 3, "pick": True},
    {"name": "s5 w3 τ.2", "tau": 0.2, "steps": 5, "window": 3, "pick": False},
    {"name": "s5 w5 τ.2", "tau": 0.2, "steps": 5, "window": 5, "pick": False},
    {"name": "s5 global", "tau": 0.1, "steps": 5, "window": None, "pick": False},
]


def part_cos(feat: torch.Tensor, ia: torch.Tensor, ib: torch.Tensor) -> float:
    a = F.normalize(feat[ia].float().mean(0), dim=0)
    b = F.normalize(feat[ib].float().mean(0), dim=0)
    return float((a * b).sum().clamp(-1, 1).item())


def part_axis(raw: torch.Tensor, ia: torch.Tensor, ib: torch.Tensor) -> torch.Tensor:
    mu_a = raw[ia].float().mean(0)
    mu_b = raw[ib].float().mean(0)
    axis = mu_a - mu_b
    return F.normalize(axis, dim=0)


@torch.no_grad()
def collect_targets(data_dir, device, min_patches: int):
    cfg = configuration.load_config(DATASETS["ytvis"])
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
    want = set(TARGET_CLIPS["ytvis"])
    found = {}
    n_clips = 0
    print("scanning ytvis for", sorted(want), flush=True)
    for batch in dm.val_dataloader():
        if n_clips in want:
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
            gt = patch_ids_ytvis(seg[ti].to(device), grid)
            obj_ids = [int(i) for i in gt.unique().tolist() if int(i) != -1]
            best = None
            xn = F.normalize(raw.float(), dim=-1)
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
            if best is None:
                print(f"  clip {n_clips}: no 2-means split", flush=True)
            else:
                found[n_clips] = {
                    "clip": n_clips,
                    "raw": raw.cpu(),
                    "rgb": denorm_frame(frame),
                    "gt_grid": gt.cpu().reshape(grid, grid),
                    "oid": best[1],
                    "ia": best[2],
                    "ib": best[3],
                    "n": best[0],
                }
                print(
                    f"  clip {n_clips}: oid={best[1]} n={best[0]} "
                    f"raw_part_cos={part_cos(raw, best[2].to(device), best[3].to(device)):.3f}",
                    flush=True,
                )
        n_clips += 1
        if n_clips > max(want):
            break
    del backbone
    torch.cuda.empty_cache()
    return [found[i] for i in TARGET_CLIPS["ytvis"] if i in found]


@torch.no_grad()
def project_maps(fr, device):
    raw = fr["raw"].to(device)
    ia, ib = fr["ia"].to(device), fr["ib"].to(device)
    axis = part_axis(raw, ia, ib)
    mid = 0.5 * (raw[ia].float().mean(0) + raw[ib].float().mean(0))
    grid = int(round(math.sqrt(raw.shape[0])))
    raw_proj = ((raw.float() - mid) @ axis).reshape(grid, grid)
    # scale from the raw object patches only
    obj = torch.cat([ia, ib])
    ys = obj // grid
    xs = obj % grid
    vals = raw_proj[ys, xs]
    vmax = float(vals.abs().quantile(0.95).item())
    vmax = max(vmax, 1e-3)

    maps = {}
    scores = {}
    for st in SETTINGS:
        if st["tau"] is None:
            feat = raw
        else:
            feat = apply_smooth(raw, float(st["tau"]), st["window"], int(st["steps"]))
        proj = ((feat.float() - mid) @ axis).reshape(grid, grid).cpu().numpy()
        maps[st["name"]] = proj
        scores[st["name"]] = part_cos(feat, ia, ib)
    fr["maps"] = maps
    fr["scores"] = scores
    fr["vmax"] = vmax
    fr["raw_part_cos"] = scores["raw"]
    return fr


def draw(frames, out_path: str) -> None:
    n_row = len(frames)
    n_col = 1 + len(SETTINGS)
    fig, axes = plt.subplots(n_row, n_col, figsize=(2.7 * n_col, 3.15 * n_row))
    if n_row == 1:
        axes = axes[None, :]
    cmap = plt.get_cmap("coolwarm")
    for r, fr in enumerate(frames):
        rgb = fr["rgb"]
        h, w = rgb.shape[:2]
        dummy = np.zeros((h, w, 3))
        crop = object_crop(rgb / 255.0, dummy, fr["gt_grid"], fr["oid"], pad=6)
        if crop is None:
            y0, y1, x0, x1 = 0, h, 0, w
        else:
            _, _, (y0, y1, x0, x1) = crop
        ax = axes[r, 0]
        ax.imshow(rgb[y0:y1, x0:x1])
        ax.set_ylabel(f"clip {fr['clip']}  n={fr['n']}", fontsize=9)
        ax.set_title("RGB crop", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        norm = TwoSlopeNorm(vmin=-fr["vmax"], vcenter=0.0, vmax=fr["vmax"])
        for c, st in enumerate(SETTINGS, start=1):
            m = upsample(fr["maps"][st["name"]][..., None], h, w)[y0:y1, x0:x1, 0]
            ax = axes[r, c]
            im = ax.imshow(m, cmap=cmap, norm=norm)
            title = f"{st['name']}\npart-cos {fr['scores'][st['name']]:.2f}"
            ax.set_title(
                title,
                fontsize=8,
                color="#c0392b" if st["pick"] else "black",
                fontweight="bold" if st["pick"] else "normal",
            )
            if st["pick"]:
                for spine in ax.spines.values():
                    spine.set_color("#c0392b")
                    spine.set_linewidth(2.4)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == n_col - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Projection onto the RAW 2-means part axis (red vs blue = head vs torso). "
        "Same scale per row. Flat / white = parts merged. Number = cosine of the two part means.",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--min-patches", type=int, default=16)
    ap.add_argument("--out-dir", default="event_analysis/featcur_mix_sweep")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    frames = collect_targets(args.data_dir, device, args.min_patches)
    frames = [project_maps(fr, device) for fr in frames]
    print("\npart-cos per setting:", flush=True)
    for fr in frames:
        print(f" clip {fr['clip']}: " + ", ".join(
            f"{k}={v:.3f}" for k, v in fr["scores"].items()
        ), flush=True)
    out = os.path.join(args.out_dir, "dino_part_axis_rec.png")
    draw(frames, out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
