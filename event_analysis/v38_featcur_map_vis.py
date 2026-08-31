"""v38 Key-only Ncut mix: how X^bind dilutes on the patch grid vs training step.

X^rel = P X,  P = row-normalize(ReLU-cosine), barrier=false
X^bind(s) = (1-s) X^rel + s X
s = 1/2 (1 - cos(pi t)), t = clip(step/50k, 0, 1)

Left-to-right is the train clock. PCA uses one basis fit on raw DINO so color
is comparable. The kernel row is the effective mixer K = (1-s) P + s I for
the largest-object centroid: that is the spatial dilution of one query patch.

Usage (slotcurri image):
  python event_analysis/v38_featcur_map_vis.py --data-dir /workspace/dataset
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
from featcur_dino_vis import denorm_frame, pca_basis, pca_map, upsample
from featcur_strength_probe import extract_tokens, patch_ids_movi, patch_ids_ytvis
from slotcurri import configuration, data
from slotcurri.models import feature_curriculum_mix
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor

ANNEAL = 50000
STEPS = [0, 10000, 25000, 40000, 50000]
DATASETS = {
    "ytvis": "configs/slotcurri/ytvis2021_attnmass_v38.yaml",
    "movi_c": "configs/slotcurri/movi_c_attnmass_v38.yaml",
}


def mix_at(step: int) -> float:
    return feature_curriculum_mix(step, ANNEAL, "cosine")


def relation_p(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    z = F.normalize(x.float(), dim=-1)
    w = z @ z.transpose(0, 1)
    w = w.clamp_min(0.0)
    w.fill_diagonal_(0.0)
    return w / w.sum(dim=-1, keepdim=True).clamp_min(eps)


def unique_keys(tokens: torch.Tensor, thr: float = 0.95) -> int:
    xn = F.normalize(tokens.float(), dim=-1)
    kept = []
    for i in range(xn.shape[0]):
        if not kept:
            kept.append(xn[i])
            continue
        if float((torch.stack(kept) @ xn[i]).max()) < thr:
            kept.append(xn[i])
    return len(kept)


def neighbor_cos(tokens: torch.Tensor, grid: int) -> np.ndarray:
    xn = F.normalize(tokens.float(), dim=-1).reshape(grid, grid, -1)
    acc = torch.zeros(grid, grid, device=tokens.device)
    cnt = torch.zeros(grid, grid, device=tokens.device)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        a = xn[max(dy, 0) : grid + min(dy, 0), max(dx, 0) : grid + min(dx, 0)]
        b = xn[max(-dy, 0) : grid + min(-dy, 0), max(-dx, 0) : grid + min(-dx, 0)]
        c = (a * b).sum(-1)
        acc[max(dy, 0) : grid + min(dy, 0), max(dx, 0) : grid + min(dx, 0)] += c
        cnt[max(dy, 0) : grid + min(dy, 0), max(dx, 0) : grid + min(dx, 0)] += 1
    return (acc / cnt.clamp_min(1)).cpu().numpy()


def object_centroid(gt: torch.Tensor, grid: int, bg_id: int) -> int:
    ids = [int(i) for i in gt.unique().tolist() if int(i) != bg_id]
    if not ids:
        return (grid // 2) * grid + (grid // 2)
    best, nbest = ids[0], -1
    for oid in ids:
        n = int((gt == oid).sum())
        if n > nbest:
            best, nbest = oid, n
    ys = torch.arange(grid, device=gt.device).view(grid, 1).expand(grid, grid).reshape(-1)
    xs = torch.arange(grid, device=gt.device).view(1, grid).expand(grid, grid).reshape(-1)
    m = gt == best
    y = int(ys[m].float().mean().round().item())
    x = int(xs[m].float().mean().round().item())
    y = min(max(y, 0), grid - 1)
    x = min(max(x, 0), grid - 1)
    return y * grid + x


@torch.no_grad()
def collect(name, cfg_path, data_dir, device, n_keep, max_scan, min_obj):
    cfg = configuration.load_config(cfg_path)
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
    ncut = NcutRelationalLeveling(chunk_size=8, n_iter=16, barrier=False).to(device)
    feat_key = bb_cfg.features
    id_fn = patch_ids_ytvis if name == "ytvis" else patch_ids_movi
    bg_id = -1 if name == "ytvis" else 0
    picked = []
    n_clips = 0
    print(f"scan {name}", flush=True)
    for batch in dm.val_dataloader():
        if n_clips >= max_scan or len(picked) >= n_keep:
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
        gt = id_fn(seg[ti].to(device), grid)
        n_obj = len([i for i in gt.unique().tolist() if int(i) != bg_id])
        n_clips += 1
        if n_obj < min_obj:
            continue
        rel = ncut(raw.unsqueeze(0), 0.0)[0]
        p = relation_p(raw)
        q = object_centroid(gt, grid, bg_id)
        rgb = denorm_frame(frame.cpu())
        h, w = rgb.shape[:2]
        basis = pca_basis(raw)
        proj = (raw.float() - raw.float().mean(0)) @ basis
        lo, hi = proj.quantile(0.02, dim=0), proj.quantile(0.98, dim=0)
        pca, kern, ncos, nkey = {}, {}, {}, {}
        for step in STEPS:
            s = mix_at(step)
            bind = (1.0 - s) * rel + s * raw if s < 1.0 else raw
            if s <= 0.0:
                bind = rel
            pca[step] = pca_map(bind, basis, lo, hi, grid)
            k = (1.0 - s) * p[q] + (s * F.one_hot(torch.tensor(q, device=p.device), p.shape[0]).float())
            kern[step] = k.reshape(grid, grid).cpu().numpy()
            ncos[step] = neighbor_cos(bind, grid)
            nkey[step] = unique_keys(bind)
        picked.append(
            {
                "clip": n_clips - 1,
                "rgb": rgb,
                "pca": pca,
                "kern": kern,
                "ncos": ncos,
                "nkey": nkey,
                "q": q,
                "grid": grid,
                "n_obj": n_obj,
            }
        )
        print(
            f"  keep clip {n_clips-1} n_obj={n_obj} grid={grid} q={q} "
            f"keys { {st: nkey[st] for st in STEPS} }",
            flush=True,
        )
    return picked


def draw_mix_curve(ax) -> None:
    xs = np.linspace(0, ANNEAL, 400)
    ys = [mix_at(int(s)) for s in xs]
    ax.plot(xs / 1000.0, ys, color="black", lw=1.8)
    for step in STEPS:
        ax.scatter([step / 1000.0], [mix_at(step)], zorder=3, color="#c0392b", s=28)
        ax.annotate(
            f"{mix_at(step):.2f}",
            (step / 1000.0, mix_at(step)),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7,
        )
    ax.set_xlim(-1, 52)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("train step (k)", fontsize=8)
    ax.set_ylabel("s  (0=leveled, 1=raw)", fontsize=8)
    ax.set_title(
        "v38  X^bind = (1−s) P X + s X   ·   P = row-normalize ReLU-cosine, barrier=false",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)


def draw_dataset(name, frames, out_dir: str) -> None:
    n_row = len(frames)
    n_col = 1 + len(STEPS)
    fig = plt.figure(figsize=(2.35 * n_col, 1.35 + 6.4 * n_row))
    gs = fig.add_gridspec(1 + 3 * n_row, n_col, height_ratios=[0.7] + [1, 1, 1] * n_row, hspace=0.28, wspace=0.04)
    axc = fig.add_subplot(gs[0, :])
    draw_mix_curve(axc)

    for r, fr in enumerate(frames):
        rgb = fr["rgb"]
        h, w = rgb.shape[:2]
        grid = fr["grid"]
        qy, qx = divmod(fr["q"], grid)
        base = 1 + 3 * r

        ax = fig.add_subplot(gs[base, 0])
        ax.imshow(rgb)
        ax.set_ylabel(f"clip {fr['clip']}\n{fr['n_obj']} obj", fontsize=8)
        ax.set_title("RGB", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        for c, step in enumerate(STEPS, start=1):
            ax = fig.add_subplot(gs[base, c])
            ax.imshow(upsample(fr["pca"][step], h, w))
            if r == 0:
                ax.set_title(f"{step//1000}k  s={mix_at(step):.2f}\n{fr['nkey'][step]} keys", fontsize=7)
            else:
                ax.set_title(f"{fr['nkey'][step]} keys", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

        ax = fig.add_subplot(gs[base + 1, 0])
        ax.imshow(rgb)
        ax.plot([(qx + 0.5) * w / grid], [(qy + 0.5) * h / grid], "+", ms=10, color="white", mew=1.6)
        ax.set_ylabel("mixer K·e_q", fontsize=8)
        ax.set_title("query", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for c, step in enumerate(STEPS, start=1):
            ax = fig.add_subplot(gs[base + 1, c])
            k = fr["kern"][step]
            vis = np.log10(k + 1e-6)
            ax.imshow(upsample(vis[..., None].repeat(3, axis=2), h, w)[..., 0], cmap="magma", vmin=-5.5, vmax=-0.3)
            ax.plot([(qx + 0.5) * w / grid], [(qy + 0.5) * h / grid], "+", ms=8, color="cyan", mew=1.2)
            ax.set_xticks([])
            ax.set_yticks([])

        ax = fig.add_subplot(gs[base + 2, 0])
        ax.axis("off")
        ax.set_ylabel("4-neigh cos", fontsize=8)
        for c, step in enumerate(STEPS, start=1):
            ax = fig.add_subplot(gs[base + 2, c])
            ax.imshow(upsample(fr["ncos"][step][..., None], h, w)[..., 0], cmap="viridis", vmin=0.55, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"{name}  ·  same PCA (fit on raw)  ·  kernel = (1−s)P + s I at object centroid  ·  "
        "neighbor-cos high = locally diluted",
        fontsize=11,
        y=0.995,
    )
    out = os.path.join(out_dir, f"v38_{name}_featmap_schedule.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--out-dir", default="event_analysis/v38_featcur_map")
    ap.add_argument("--n-keep", type=int, default=2)
    ap.add_argument("--max-scan", type=int, default=40)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, cfg in DATASETS.items():
        min_obj = 2 if name == "ytvis" else 4
        frames = collect(name, cfg, args.data_dir, device, args.n_keep, args.max_scan, min_obj)
        if frames:
            draw_dataset(name, frames, args.out_dir)


if __name__ == "__main__":
    main()
