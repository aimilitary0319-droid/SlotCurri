"""PCA-RGB visualization of DINO tokens before / after FeatureSmoothing.

Fits one PCA basis per frame on the raw tokens so every setting shares the same
color space. Parts merging show up as a single object color; instance bleed shows
up as neighboring objects collapsing to the same color.

Usage (inside the slotcurri container):
  python event_analysis/featcur_dino_vis.py --data-dir /workspace/dataset
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_strength_probe import (  # noqa: E402
    DATASETS,
    apply_smooth,
    extract_tokens,
    kmeans2,
    patch_ids_movi,
    patch_ids_ytvis,
)
from slotcurri import configuration, data
from slotcurri.data.transforms import Denormalize
from slotcurri.modules.encoders import TimmExtractor


VIS_SETTINGS = [
    {"name": "raw DINO", "tau": None, "steps": 0, "window": None, "pick": False},
    {"name": "v33  τ.1 s1 w9", "tau": 0.1, "steps": 1, "window": 9, "pick": False},
    {"name": "τ.1 s5 w5", "tau": 0.1, "steps": 5, "window": 5, "pick": False},
    {"name": "τ.2 s3 w3  (rec)", "tau": 0.2, "steps": 3, "window": 3, "pick": True},
    {"name": "τ.2 s5 w3", "tau": 0.2, "steps": 5, "window": 3, "pick": False},
    {"name": "τ.2 s5 w5", "tau": 0.2, "steps": 5, "window": 5, "pick": False},
    {"name": "τ.1 s5 global", "tau": 0.1, "steps": 5, "window": None, "pick": False},
]


def denorm_frame(frame: torch.Tensor) -> np.ndarray:
    """(C,H,W) normalized -> uint8 RGB."""
    x = Denormalize("image")(frame.detach().cpu()).clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def instance_contours(seg: torch.Tensor, bg_id: int) -> np.ndarray:
    """Boolean (H,W) edge map of instance boundaries."""
    if seg.ndim == 3:
        # YTVIS (I,H,W) instance binaries or MOVi (C,H,W) one-hot.
        seg_f = seg.float()
        if bg_id == 0:
            ids = seg_f.argmax(dim=0)
        else:
            cov, idx = seg_f.max(dim=0)
            ids = idx.clone()
            ids[cov < 0.5] = -1
    else:
        ids = seg
    ids = ids.cpu().numpy()
    edge = np.zeros(ids.shape, dtype=bool)
    edge[1:] |= ids[1:] != ids[:-1]
    edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    return edge


def pca_basis(tokens: torch.Tensor) -> torch.Tensor:
    """(F,D) -> (D,3) top principal axes, centered."""
    x = tokens.float()
    x = x - x.mean(dim=0, keepdim=True)
    # thin SVD; Vh[:3] are the top-3 right singular vectors
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    return vh[:3].T.contiguous()


def pca_map(tokens: torch.Tensor, basis: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, grid: int) -> np.ndarray:
    x = tokens.float()
    x = x - x.mean(dim=0, keepdim=True)
    proj = x @ basis
    proj = (proj - lo) / (hi - lo).clamp_min(1e-6)
    proj = proj.clamp(0, 1)
    img = proj.reshape(grid, grid, 3).cpu().numpy()
    return img


def upsample(img: np.ndarray, h: int, w: int) -> np.ndarray:
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=(h, w), mode="nearest")
    return t[0].permute(1, 2, 0).numpy()


def overlay_edges(rgb: np.ndarray, edge: np.ndarray, color=(1.0, 1.0, 0.2)) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    if out.max() > 1.5:
        out = out / 255.0
    out[edge] = np.array(color, dtype=np.float32)
    return np.clip(out, 0, 1)


def object_crop(rgb, pca, gt_ids, oid, pad=8):
    ys, xs = (gt_ids == oid).nonzero(as_tuple=True)
    if ys.numel() == 0:
        return None
    grid = gt_ids.shape[0]
    h, w = rgb.shape[:2]
    y0 = max(int(ys.min()) * h // grid - pad, 0)
    y1 = min(int(ys.max() + 1) * h // grid + pad, h)
    x0 = max(int(xs.min()) * w // grid - pad, 0)
    x1 = min(int(xs.max() + 1) * w // grid + pad, w)
    return rgb[y0:y1, x0:x1], pca[y0:y1, x0:x1], (y0, y1, x0, x1)


@torch.no_grad()
def score_frame(raw: torch.Tensor, gt: torch.Tensor, bg_id: int, min_patches: int) -> dict:
    obj_ids = [int(i) for i in gt.unique().tolist() if int(i) != bg_id]
    xn = F.normalize(raw.float(), dim=-1)
    parts = []
    sizes = []
    for oid in obj_ids:
        idx = (gt == oid).nonzero(as_tuple=False).squeeze(-1)
        n = int(idx.numel())
        sizes.append(n)
        if n < min_patches:
            continue
        lab = kmeans2(xn[idx])
        if lab is None:
            continue
        a = raw[idx[lab == 0]].mean(0)
        b = raw[idx[lab == 1]].mean(0)
        cos = float((F.normalize(a, dim=0) * F.normalize(b, dim=0)).sum().clamp(-1, 1))
        parts.append((n, cos, oid))
    n_obj = len(obj_ids)
    if not parts:
        return {"n_obj": n_obj, "max_n": max(sizes) if sizes else 0, "part_cos": 1.0, "focus": -1}
    parts.sort(reverse=True)
    n, cos, oid = parts[0]
    return {"n_obj": n_obj, "max_n": n, "part_cos": cos, "focus": oid}


def pick_frames(cands, n_part=4, n_multi=2):
    yt = [c for c in cands if c["dataset"] == "ytvis"]
    mv = [c for c in cands if c["dataset"] == "movi_c"]
    part = [c for c in yt if c["max_n"] >= 40 and c["part_cos"] < 0.78]
    part.sort(key=lambda c: (c["part_cos"], -c["max_n"]))
    chosen = []
    seen = set()
    for c in part:
        if c["clip"] in seen:
            continue
        chosen.append(c)
        seen.add(c["clip"])
        if len(chosen) >= n_part:
            break
    multi = [c for c in yt if c["n_obj"] >= 2 and c["max_n"] >= 24 and c["clip"] not in seen]
    multi.sort(key=lambda c: (-c["n_obj"], -c["max_n"]))
    n_added = 0
    for c in multi:
        if c["clip"] in seen:
            continue
        chosen.append(c)
        seen.add(c["clip"])
        n_added += 1
        if n_added >= n_multi:
            break
    mv_ok = [c for c in mv if c["n_obj"] >= 2 and c["max_n"] >= 20]
    mv_ok.sort(key=lambda c: (-c["n_obj"], c["part_cos"]))
    seen_m = set()
    for c in mv_ok:
        if c["clip"] in seen_m:
            continue
        chosen.append(c)
        seen_m.add(c["clip"])
        if sum(1 for x in chosen if x["dataset"] == "movi_c") >= 2:
            break
    return chosen


def draw_grid(frames, out_path: str) -> None:
    n_row = len(frames)
    n_col = 1 + len(VIS_SETTINGS)
    fig, axes = plt.subplots(n_row, n_col, figsize=(2.35 * n_col, 2.45 * n_row))
    if n_row == 1:
        axes = axes[None, :]
    for r, fr in enumerate(frames):
        rgb = fr["rgb"]
        edge = fr["edge"]
        h, w = rgb.shape[:2]
        show_rgb = overlay_edges(rgb, edge)
        ax = axes[r, 0]
        ax.imshow(show_rgb)
        ax.set_ylabel(
            f"{fr['dataset']}\nclip {fr['clip']}  part-cos {fr['part_cos']:.2f}",
            fontsize=8,
        )
        if r == 0:
            ax.set_title("RGB + GT", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for c, st in enumerate(VIS_SETTINGS, start=1):
            pca = upsample(fr["pca"][st["name"]], h, w)
            pca = overlay_edges(pca, edge, color=(1, 1, 1))
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
    fig.suptitle(
        "DINO PCA (same 3 axes + range per row, fit on raw). "
        "Yellow/white = GT boundaries. Rec = τ=0.2, window=3, n_steps=3.",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def draw_zooms(frames, out_path: str) -> None:
    """Largest-object crops so head/torso structure is readable."""
    rows = [fr for fr in frames if fr["focus"] >= 0 and fr["dataset"] == "ytvis"][:4]
    if not rows:
        return
    n_row = len(rows)
    n_col = 1 + len(VIS_SETTINGS)
    fig, axes = plt.subplots(n_row, n_col, figsize=(2.2 * n_col, 2.4 * n_row))
    if n_row == 1:
        axes = axes[None, :]
    for r, fr in enumerate(rows):
        rgb = fr["rgb"]
        h, w = rgb.shape[:2]
        pca_raw = upsample(fr["pca"]["raw DINO"], h, w)
        crop = object_crop(rgb / 255.0, pca_raw, fr["gt_grid"], fr["focus"])
        if crop is None:
            continue
        _, _, (y0, y1, x0, x1) = crop
        ax = axes[r, 0]
        ax.imshow(fr["rgb"][y0:y1, x0:x1])
        ax.set_ylabel(f"clip {fr['clip']}  oid {fr['focus']}", fontsize=8)
        if r == 0:
            ax.set_title("object crop", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for c, st in enumerate(VIS_SETTINGS, start=1):
            pca = upsample(fr["pca"][st["name"]], h, w)[y0:y1, x0:x1]
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
    fig.suptitle("Same PCA, cropped to the largest object (head/torso test).", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def collect_dataset(name, cfg_path, data_dir, device, max_clips, min_patches):
    print(f"\n=== scan {name} ===", flush=True)
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

    cands = []
    n_clips = 0
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
        t = video.shape[0]
        ti = int(t // 2)
        frame = video[ti].to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            raw = extract_tokens(backbone, frame.unsqueeze(0), feat_key).float()[0]
        grid = int(round(math.sqrt(raw.shape[0])))
        if name == "ytvis":
            gt = patch_ids_ytvis(seg[ti].to(device), grid)
        else:
            gt = patch_ids_movi(seg[ti].to(device), grid)
        sc = score_frame(raw, gt, bg_id, min_patches)
        cands.append(
            {
                "dataset": name,
                "clip": n_clips,
                "frame": ti,
                "raw": raw.cpu(),
                "rgb": denorm_frame(frame),
                "edge": instance_contours(seg[ti], bg_id),
                "gt_grid": gt.cpu().reshape(grid, grid),
                **sc,
            }
        )
        n_clips += 1
        if n_clips % 20 == 0:
            print(f"  {name}: {n_clips} clips", flush=True)

    del backbone
    torch.cuda.empty_cache()
    return cands


def apply_settings(fr, device):
    raw = fr["raw"].to(device)
    grid = int(round(math.sqrt(raw.shape[0])))
    basis = pca_basis(raw)
    proj_raw = (raw.float() - raw.float().mean(0)) @ basis
    lo = proj_raw.quantile(0.02, dim=0)
    hi = proj_raw.quantile(0.98, dim=0)
    pca = {}
    for st in VIS_SETTINGS:
        if st["tau"] is None:
            feat = raw
        else:
            feat = apply_smooth(raw, float(st["tau"]), st["window"], int(st["steps"]))
        pca[st["name"]] = pca_map(feat, basis, lo, hi, grid)
    fr["pca"] = pca
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--max-clips", type=int, default=60)
    ap.add_argument("--min-patches", type=int, default=16)
    ap.add_argument("--out-dir", default="event_analysis/featcur_mix_sweep")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device={device}", flush=True)

    cands = []
    for ds in ("ytvis", "movi_c"):
        cands.extend(
            collect_dataset(ds, DATASETS[ds], args.data_dir, device, args.max_clips, args.min_patches)
        )
    chosen = pick_frames(cands)
    print("picked:", [(c["dataset"], c["clip"], c["part_cos"], c["n_obj"], c["max_n"]) for c in chosen], flush=True)

    frames = [apply_settings(c, device) for c in chosen]
    # drop bulky tensors before matplotlib
    for fr in frames:
        fr.pop("raw", None)

    grid_path = os.path.join(args.out_dir, "dino_pca_grid_rec.png")
    zoom_path = os.path.join(args.out_dir, "dino_pca_object_crops_rec.png")
    draw_grid(frames, grid_path)
    draw_zooms(frames, zoom_path)
    print(f"wrote {grid_path}\nwrote {zoom_path}", flush=True)


if __name__ == "__main__":
    main()
