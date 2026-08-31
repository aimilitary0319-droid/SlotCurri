"""Is v38 P X actually mixing objects, or only looking blurry in raw-PCA?

Prints token stats and writes a side-by-side:
  top: PCA fit on raw (same as the schedule figure — washed-out)
  bot: PCA fit on that step's X^bind (collapse shows as flat color)
"""

from __future__ import annotations

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
from featcur_dino_vis import denorm_frame, pca_basis, pca_map, upsample
from featcur_strength_probe import extract_tokens, patch_ids_movi, patch_ids_ytvis
from slotcurri import configuration, data
from slotcurri.models import feature_curriculum_mix
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor

ANNEAL = 50000
STEPS = [0, 10000, 25000, 50000]
DATASETS = {
    "ytvis": ("configs/slotcurri/ytvis2021_attnmass_v38.yaml", 2, -1),
    "movi_c": ("configs/slotcurri/movi_c_attnmass_v38.yaml", 4, 0),
}


def mix_at(step: int) -> float:
    return feature_curriculum_mix(step, ANNEAL, "cosine")


def mean_offdiag_cos(x: torch.Tensor) -> float:
    xn = F.normalize(x.float(), dim=-1)
    n = xn.shape[0]
    s = xn @ xn.T
    return float((s.sum() - n) / (n * (n - 1)))


def object_mean_cos(x: torch.Tensor, gt: torch.Tensor, bg_id: int) -> float:
    ids = [int(i) for i in gt.unique().tolist() if int(i) != bg_id]
    if len(ids) < 2:
        return float("nan")
    means = []
    for oid in ids:
        m = gt == oid
        if int(m.sum()) < 4:
            continue
        means.append(F.normalize(x[m].float().mean(0), dim=0))
    if len(means) < 2:
        return float("nan")
    M = torch.stack(means)
    s = M @ M.T
    k = s.shape[0]
    return float((s.sum() - k) / (k * (k - 1)))


def other_obj_p_mass(x: torch.Tensor, gt: torch.Tensor, bg_id: int) -> float:
    """For each object patch, fraction of P mass that lands on a *different* object."""
    z = F.normalize(x.float(), dim=-1)
    w = (z @ z.T).clamp_min(0.0)
    w.fill_diagonal_(0.0)
    p = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
    ids = [int(i) for i in gt.unique().tolist() if int(i) != bg_id]
    if not ids:
        return float("nan")
    fracs = []
    for oid in ids:
        src = gt == oid
        if int(src.sum()) < 4:
            continue
        other = (gt != oid) & (gt != bg_id)
        if int(other.sum()) == 0:
            continue
        mass = p[src][:, other].sum(-1)
        fracs.append(float(mass.mean()))
    return float(np.mean(fracs)) if fracs else float("nan")


@torch.no_grad()
def collect(name, cfg_path, min_obj, bg_id, data_dir, device, n_keep=2, max_scan=40):
    cfg = configuration.load_config(cfg_path)
    cfg.dataset.val_batch_size = 1
    cfg.dataset.num_val_workers = 0
    dm = data.build(cfg.dataset, data_dir=data_dir)
    dm.setup("validate")
    bb = cfg.model.encoder.backbone
    kwargs = dict(bb.get("model_kwargs") or {})
    backbone = TimmExtractor(
        model=bb.model, pretrained=True, frozen=True, features=bb.features, model_kwargs=kwargs or None
    ).to(device).eval()
    ncut = NcutRelationalLeveling(chunk_size=8, n_iter=16, barrier=False).to(device)
    id_fn = patch_ids_ytvis if name == "ytvis" else patch_ids_movi
    out, n_clips = [], 0
    for batch in dm.val_dataloader():
        if n_clips >= max_scan or len(out) >= n_keep:
            break
        if "batch_padding_mask" in batch and torch.is_tensor(batch["batch_padding_mask"]) and bool(batch["batch_padding_mask"].any()):
            n_clips += 1
            continue
        video, seg = batch["video"][0], batch["segmentations"][0]
        ti = int(video.shape[0] // 2)
        frame = video[ti].to(device)
        raw = extract_tokens(backbone, frame.unsqueeze(0), bb.features).float()[0]
        grid = int(round(math.sqrt(raw.shape[0])))
        gt = id_fn(seg[ti].to(device), grid)
        n_obj = len([i for i in gt.unique().tolist() if int(i) != bg_id])
        n_clips += 1
        if n_obj < min_obj:
            continue
        rel = ncut(raw.unsqueeze(0), 0.0)[0]
        leak = other_obj_p_mass(raw, gt, bg_id)
        rec = {
            "clip": n_clips - 1,
            "n_obj": n_obj,
            "grid": grid,
            "rgb": denorm_frame(frame.cpu()),
            "raw": raw.cpu(),
            "rel": rel.cpu(),
            "gt": gt.cpu(),
            "p_other": leak,
        }
        print(
            f"{name} clip {rec['clip']} n_obj={n_obj}  "
            f"pair-cos raw={mean_offdiag_cos(raw):.3f} rel={mean_offdiag_cos(rel):.3f}  "
            f"obj-mean-cos raw={object_mean_cos(raw, gt, bg_id):.3f} rel={object_mean_cos(rel, gt, bg_id):.3f}  "
            f"std raw={float(raw.std()):.3f} rel={float(rel.std()):.3f}  "
            f"P mass on other objects={leak:.3f}",
            flush=True,
        )
        out.append(rec)
    return out


def bind(raw, rel, step):
    s = mix_at(step)
    if s >= 1.0:
        return raw
    if s <= 0.0:
        return rel
    return (1.0 - s) * rel + s * raw


def cos_to_mean_map(x: torch.Tensor, grid: int) -> np.ndarray:
    xn = F.normalize(x.float(), dim=-1)
    m = F.normalize(xn.mean(0), dim=0)
    return (xn @ m).reshape(grid, grid).cpu().numpy()


def draw(name, frames, out_path, bg_id):
    n_row = len(frames)
    n_col = 1 + len(STEPS)
    fig, axes = plt.subplots(n_row * 2, n_col, figsize=(2.3 * n_col, 2.35 * n_row * 2))
    for r, fr in enumerate(frames):
        rgb, h, w = fr["rgb"], *fr["rgb"].shape[:2]
        grid = fr["grid"]
        raw, rel = fr["raw"], fr["rel"]
        basis_raw = pca_basis(raw)
        proj = (raw.float() - raw.mean(0)) @ basis_raw
        lo, hi = proj.quantile(0.02, dim=0), proj.quantile(0.98, dim=0)
        axes[2 * r, 0].imshow(rgb)
        axes[2 * r, 0].set_ylabel(f"clip {fr['clip']}\nPCA on raw basis", fontsize=8)
        axes[2 * r, 0].set_title("RGB", fontsize=8)
        axes[2 * r, 0].set_xticks([])
        axes[2 * r, 0].set_yticks([])
        axes[2 * r + 1, 0].imshow(rgb)
        axes[2 * r + 1, 0].set_ylabel("cos to frame mean", fontsize=8)
        axes[2 * r + 1, 0].axis("off")
        for c, step in enumerate(STEPS, start=1):
            feat = bind(raw, rel, step)
            axes[2 * r, c].imshow(upsample(pca_map(feat, basis_raw, lo, hi, grid), h, w))
            heat = torch.from_numpy(cos_to_mean_map(feat, grid))[None, None].float()
            heat = F.interpolate(heat, size=(h, w), mode="nearest")[0, 0].numpy()
            im = axes[2 * r + 1, c].imshow(heat, vmin=0.55, vmax=1.0, cmap="magma")
            oc = object_mean_cos(feat, fr["gt"], bg_id)
            cm = float(cos_to_mean_map(feat, grid).mean())
            if r == 0:
                axes[2 * r, c].set_title(
                    f"{step//1000}k s={mix_at(step):.2f}\nobj-cos {oc:.2f}", fontsize=7
                )
            else:
                axes[2 * r, c].set_title(f"obj-cos {oc:.2f}", fontsize=7)
            axes[2 * r + 1, c].set_xlabel(f"mean {cm:.3f}", fontsize=7)
            axes[2 * r, c].set_xticks([])
            axes[2 * r, c].set_yticks([])
            axes[2 * r + 1, c].set_xticks([])
            axes[2 * r + 1, c].set_yticks([])
        fig.colorbar(im, ax=axes[2 * r + 1, 1:], fraction=0.02, pad=0.01)
    fig.suptitle(
        f"{name}: top = same raw-PCA as the schedule figure (looks only washed). "
        "bottom = cosine to the frame-mean token (1.0 = every patch is the same vector).",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("wrote", out_path, flush=True)


def main():
    data_dir = "/workspace/dataset"
    out_dir = "event_analysis/v38_featcur_map"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    for name, (cfg, min_obj, bg_id) in DATASETS.items():
        frames = collect(name, cfg, min_obj, bg_id, data_dir, device)
        if frames:
            print(f"--- {name} bind stats ---", flush=True)
            for fr in frames:
                for step in STEPS:
                    feat = bind(fr["raw"], fr["rel"], step)
                    print(
                        f"  clip {fr['clip']} {step//1000}k  "
                        f"pair={mean_offdiag_cos(feat):.3f}  "
                        f"obj={object_mean_cos(feat, fr['gt'], bg_id):.3f}  "
                        f"std={float(feat.std()):.3f}",
                        flush=True,
                    )
            draw(name, frames, os.path.join(out_dir, f"v38_{name}_mix_check.png"), bg_id)


if __name__ == "__main__":
    main()
