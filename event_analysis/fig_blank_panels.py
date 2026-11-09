"""Panels for the architecture-diagram blanks.

Frozen features: PCA of raw DINOv2 tokens on a YouTube-VIS frame.
Relational Feature Leveling: the same PCA basis on fully leveled tokens.
Spectral occupancy: the scene-wide 8-neighbor affinity, drawn so edge darkness
tracks similarity with a steep contrast, then one graph per slot.
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
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_dino_vis import denorm_frame, pca_basis, pca_map, score_frame  # noqa: E402
from featcur_strength_probe import extract_tokens, patch_ids_ytvis  # noqa: E402

from slotcurri import configuration, data, models
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor
from slotcurri.modules.video import spectral_graph_n8_eigs

CFG = "configs/slotcurri/ytvis2021_attnmass_v39lam1gu_xframe_lin40.yaml"
CKPT = (
    "logs/_ytvis_attnmass_v39lam1gu_xframe_lin40/checkpoints/"
    "slotcurri_step=step=40000.ckpt"
)
DATA_DIR = "/workspace/dataset"
OUT = "event_analysis/fig_blanks"
N_SCAN = 24
SLOT_COLORS = (
    (46, 155, 79),
    (232, 146, 42),
    (225, 91, 138),
    (61, 126, 219),
)
# Undirected 8-neighborhood, each edge once.
EDGE_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))


def level_tokens(raw: torch.Tensor) -> torch.Tensor:
    """Dense ReLU-cosine leveling, mix 0. raw: (N, D) -> (N, D)."""
    mod = NcutRelationalLeveling(chunk_size=1, barrier=False, xvid_energy=False)
    return mod(raw.float().unsqueeze(0), 0.0)[0]


def neighbor_edges(z: torch.Tensor, grid: int):
    """ReLU-cosine on the 8-neighborhood. z is (N, D), L2-normalized."""
    zh = z.view(grid, grid, -1)
    edges = []
    for dy, dx in EDGE_OFFSETS:
        y0 = max(0, -dy)
        y1 = grid - max(0, dy)
        x0 = max(0, -dx)
        x1 = grid - max(0, dx)
        src = zh[y0:y1, x0:x1]
        dst = zh[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        sim = (src * dst).sum(-1).clamp_min(0)
        ys = torch.arange(y0, y1)
        xs = torch.arange(x0, x1)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        sim_np = sim.detach().cpu().numpy()
        yy_np = yy.cpu().numpy()
        xx_np = xx.cpu().numpy()
        for y, x, s in zip(yy_np.ravel(), xx_np.ravel(), sim_np.ravel()):
            if s <= 0:
                continue
            edges.append((int(y), int(x), int(y + dy), int(x + dx), float(s)))
    return edges


def steep(values: np.ndarray, q_lo: float = 0.45, q_hi: float = 0.90, power: float = 3.2):
    """Map similarities to [0, 1] with a steep contrast. Weak edges fall to 0."""
    pos = values[values > 0]
    if pos.size == 0:
        return np.zeros_like(values)
    lo = np.quantile(pos, q_lo)
    hi = np.quantile(pos, q_hi)
    if hi <= lo + 1e-6:
        hi = lo + 1e-6
    t = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return t ** power


def paint_graph(grid: int, edges, weights, color_of, scale: int = 20, dots=None):
    """White canvas. Edge darkness follows `weights` in [0, 1]; lines stay thin."""
    side = grid * scale
    img = Image.new("RGB", (side, side), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    def center(y, x):
        return ((x + 0.5) * scale, (y + 0.5) * scale)

    order = np.argsort(weights)
    for i in order:
        t = float(weights[i])
        if t < 0.06:
            continue
        y1, x1, y2, x2, _ = edges[i]
        col = color_of(i, t)
        width = 1 if t < 0.72 else 2
        draw.line([center(y1, x1), center(y2, x2)], fill=col, width=width)
    if dots is not None:
        yy, xx, rr, cols = dots
        for y, x, r, col in zip(yy, xx, rr, cols):
            cx, cy = center(int(y), int(x))
            rad = float(r)
            draw.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=col)
    return np.asarray(img)


def save_rgb(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)
    print("wrote", path, flush=True)


def pick_score(sc):
    n_obj = sc["n_obj"]
    if n_obj < 2 or n_obj > 7:
        return -1.0
    if sc["max_n"] < 60:
        return -1.0
    # Low within-object part cosine means leveling has something to merge.
    return (1.0 - sc["part_cos"]) * math.log(sc["max_n"]) * min(n_obj, 4)


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    cfg = configuration.load_config(CFG)
    cfg.dataset.val_batch_size = 1
    cfg.dataset.num_val_workers = 0
    cfg.dataset.val_size = N_SCAN
    dm = data.build(cfg.dataset, data_dir=DATA_DIR)
    dm.setup("validate")

    bb_cfg = cfg.model.encoder.backbone
    backbone = TimmExtractor(
        model=bb_cfg.model,
        pretrained=True,
        frozen=True,
        features=bb_cfg.features,
        model_kwargs=dict(bb_cfg.get("model_kwargs") or {}) or None,
    ).to(device).eval()

    best = None
    thumbs = []
    n_clips = 0
    for batch in dm.val_dataloader():
        if n_clips >= N_SCAN:
            break
        if "batch_padding_mask" in batch:
            mask = batch["batch_padding_mask"]
            if torch.is_tensor(mask) and bool(mask.any()):
                n_clips += 1
                continue
        video = batch["video"][0]
        seg = batch["segmentations"][0]
        t = int(video.shape[0])
        ti = t // 2
        frame = video[ti].to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            raw = extract_tokens(backbone, frame.unsqueeze(0), bb_cfg.features).float()[0]
        grid = int(round(math.sqrt(raw.shape[0])))
        gt = patch_ids_ytvis(seg[ti].to(device), grid)
        sc = score_frame(raw, gt, -1, 16)
        score = pick_score(sc)
        print(
            f"clip {n_clips:02d} T={t} frame={ti} obj={sc['n_obj']} "
            f"max_n={sc['max_n']} part_cos={sc['part_cos']:.3f} score={score:.3f}",
            flush=True,
        )
        rec = {
            "clip": n_clips,
            "ti": ti,
            "raw": raw.detach().cpu(),
            "rgb": denorm_frame(frame.cpu()),
            "video": video,
            "score": score,
            "sc": sc,
            "grid": grid,
        }
        thumbs.append(rec)
        if best is None or score > best["score"]:
            best = rec
        n_clips += 1

    del backbone
    torch.cuda.empty_cache()
    if best is None or best["score"] < 0:
        raise RuntimeError("no suitable YouTube frame")
    print(
        f"picked clip {best['clip']} score {best['score']:.3f} "
        f"part_cos {best['sc']['part_cos']:.3f}",
        flush=True,
    )

    raw = best["raw"]
    grid = best["grid"]
    basis = pca_basis(raw)
    proj = (raw.float() - raw.float().mean(0)) @ basis
    lo = proj.quantile(0.02, dim=0)
    hi = proj.quantile(0.98, dim=0)
    pca_raw = pca_map(raw, basis, lo, hi, grid)
    rel = level_tokens(raw)
    # Same axes as the raw PCA, so a part that leveled away loses its color
    # instead of being stretched back into view.
    pca_rel = pca_map(rel, basis, lo, hi, grid)
    h, w = best["rgb"].shape[:2]
    up_raw = np.array(
        Image.fromarray((np.clip(pca_raw, 0, 1) * 255).astype(np.uint8)).resize(
            (w, h), Image.Resampling.NEAREST
        )
    )
    up_rel = np.array(
        Image.fromarray((np.clip(pca_rel, 0, 1) * 255).astype(np.uint8)).resize(
            (w, h), Image.Resampling.NEAREST
        )
    )
    save_rgb(best["rgb"], os.path.join(OUT, "frame_rgb.png"))
    save_rgb(up_raw, os.path.join(OUT, "frozen_pca.png"))
    save_rgb(up_rel, os.path.join(OUT, "fl_pca.png"))

    # Contact sheet of the scan, so a different frame is easy to swap in.
    show = sorted(thumbs, key=lambda r: -r["score"])[:8]
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.2))
    for c, rec in enumerate(show):
        axes[0, c].imshow(rec["rgb"])
        tok = rec["raw"]
        b = pca_basis(tok)
        p = (tok.float() - tok.float().mean(0)) @ b
        img = pca_map(tok, b, p.quantile(0.02, 0), p.quantile(0.98, 0), rec["grid"])
        axes[1, c].imshow(np.clip(img, 0, 1), interpolation="nearest")
        axes[0, c].set_title(f"c{rec['clip']} {rec['score']:.2f}", fontsize=8)
        for ax in (axes[0, c], axes[1, c]):
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scan_contact.png"), dpi=120)
    plt.close(fig)

    # Scene-wide 8-neighbor affinity of the raw tokens. Rank-mapping makes the
    # darkest edges the most similar ones, and a power drops the weak ones out,
    # so object boundaries read as gaps.
    z_raw = F.normalize(raw.float(), dim=-1)
    edges_raw = neighbor_edges(z_raw, grid)
    sims = np.array([e[4] for e in edges_raw], dtype=np.float32)
    order_e = np.argsort(sims)
    rank = np.empty(len(sims), dtype=np.float32)
    rank[order_e] = np.linspace(0.0, 1.0, len(sims), dtype=np.float32)
    tone = rank ** 2.6

    def gray_of(_i, t):
        g = int(round(255 * (1.0 - t)))
        return (g, g, g)

    scene = paint_graph(grid, edges_raw, tone, gray_of)
    save_rgb(scene, os.path.join(OUT, "scene_S.png"))

    # Slot graphs use the leveled affinity, so each object is one connected mesh.
    z = F.normalize(rel.float(), dim=-1)
    edges = neighbor_edges(z, grid)

    # Slot attention from the trained model, on a short window around this frame.
    print("loading model", flush=True)
    model = models.build(cfg.model, cfg.optimizer)
    ckpt = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    video = best["video"]
    t = int(video.shape[0])
    ti = best["ti"]
    a = max(0, ti - 3)
    b = min(t, a + 6)
    a = max(0, b - 6)
    local_t = ti - a
    clip = video[a:b].unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model({"video": clip}, train=False)
    att = outputs["processor"]["state_attn_mask"]
    print("attn", tuple(att.shape), "local", local_t, flush=True)
    # (B, T, S, N) or (T, S, N)
    if att.ndim == 4:
        att_f = att[0, local_t].float()
    elif att.ndim == 3:
        att_f = att[local_t].float()
    else:
        raise RuntimeError(f"unexpected attention shape {tuple(att.shape)}")
    pi, _, _, q1 = spectral_graph_n8_eigs(
        att_f.unsqueeze(0).cpu(),
        raw.unsqueeze(0),
        use_lambda1=True,
        return_q1=True,
    )
    pi = pi[0]
    q1 = q1[0]
    order = torch.argsort(pi, descending=True)
    print(
        "pi",
        " ".join(f"{int(i)}:{float(pi[i]):.3f}" for i in order.tolist()),
        flush=True,
    )
    chosen = [int(i) for i in order.tolist() if float(pi[i]) > 0.02][:4]
    if len(chosen) < 4:
        chosen = [int(i) for i in order.tolist()[:4]]

    panels = []
    att_np = att_f.cpu().numpy()
    for rank, slot in enumerate(chosen):
        color = SLOT_COLORS[rank]
        a_map = att_np[slot].reshape(grid, grid)
        a_flat = att_np[slot]
        slot_w = []
        for y1, x1, y2, x2, s in edges:
            i1 = y1 * grid + x1
            i2 = y2 * grid + x2
            slot_w.append(s * a_flat[i1] * a_flat[i2])
        slot_w = np.array(slot_w, dtype=np.float32)
        keep = np.array(
            [
                a_flat[y1 * grid + x1] >= 0.22 * a_map.max()
                and a_flat[y2 * grid + x2] >= 0.22 * a_map.max()
                for y1, x1, y2, x2, _s in edges
            ],
            dtype=bool,
        )
        slot_w = np.where(keep, slot_w, 0.0)
        slot_t = steep(slot_w, q_lo=0.05, q_hi=0.70, power=1.6)

        def color_of(_i, t, color=color):
            # Strong edges take the slot color; weak ones fade to white.
            return tuple(int(round(255 * (1 - t) + c * t)) for c in color)

        panel = paint_graph(grid, edges, slot_t, color_of)
        panels.append(panel)
        save_rgb(panel, os.path.join(OUT, f"slot_{rank}_id{slot}.png"))

    row = np.concatenate(panels, axis=1)
    save_rgb(row, os.path.join(OUT, "slot_structures.png"))

    # Overview matching the three blanks, plus the slot row.
    gap = np.full((h, 12, 3), 255, dtype=np.uint8)
    top = np.concatenate(
        [
            up_raw,
            gap,
            up_rel,
            gap,
            np.array(Image.fromarray(scene).resize((w, h), Image.Resampling.NEAREST)),
        ],
        axis=1,
    )
    slots_row = np.array(Image.fromarray(row).resize((top.shape[1], h // 2), Image.Resampling.NEAREST))
    pad = np.full((16, top.shape[1], 3), 255, dtype=np.uint8)
    overview = np.concatenate([top, pad, slots_row], axis=0)
    save_rgb(overview, os.path.join(OUT, "overview.png"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
