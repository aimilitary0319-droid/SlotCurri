"""Aggregation kernels for the box above Frozen features.

Relational Feature Leveling replaces each patch by an affinity-weighted average
of the other patches in the frame. This figure shows, for three query patches,
which locations contribute to that average.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_dino_vis import denorm_frame  # noqa: E402
from featcur_strength_probe import extract_tokens, patch_ids_ytvis  # noqa: E402
from slotcurri import configuration, data
from slotcurri.modules.encoders import TimmExtractor

CFG = "configs/slotcurri/ytvis2021_attnmass_v39lam1gu_xframe_lin40.yaml"
DATA_DIR = "/workspace/dataset"
OUT = "event_analysis/fig_blanks"
CLIP = 15


def relation_p(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    z = F.normalize(x.float(), dim=-1)
    w = (z @ z.transpose(0, 1)).clamp_min(0.0)
    w.fill_diagonal_(0.0)
    return w / w.sum(dim=-1, keepdim=True).clamp_min(eps)


def centroid(mask: torch.Tensor, grid: int) -> int:
    ys = torch.arange(grid, device=mask.device).view(grid, 1).expand(grid, grid).reshape(-1)
    xs = torch.arange(grid, device=mask.device).view(1, grid).expand(grid, grid).reshape(-1)
    y = int(ys[mask].float().mean().round().item())
    x = int(xs[mask].float().mean().round().item())
    y = min(max(y, 0), grid - 1)
    x = min(max(x, 0), grid - 1)
    return y * grid + x


def queries(gt: torch.Tensor, grid: int):
    """Two largest instances, then a background patch that is typical grass."""
    ids = [int(i) for i in gt.unique().tolist() if int(i) >= 0]
    ids.sort(key=lambda i: int((gt == i).sum()), reverse=True)
    chosen = []
    for oid in ids[:2]:
        chosen.append(centroid(gt == oid, grid))
    bg = gt < 0
    if int(bg.sum()) > 10:
        # The background token most similar to other background tokens.
        flat = bg.reshape(-1)
        chosen.append(centroid(bg, grid))
    while len(chosen) < 3:
        chosen.append((grid // 2) * grid + grid // 2)
    return chosen


def render(rgb: np.ndarray, heat: np.ndarray, q: int, grid: int) -> np.ndarray:
    """Dim the frame and paint the aggregation weights in a warm map."""
    h, w = rgb.shape[:2]
    weight = heat.astype(np.float32)
    weight = weight / max(float(weight.max()), 1e-8)
    weight = np.where(weight >= 0.08, weight, 0.0)
    weight = weight ** 0.45
    up = np.array(
        Image.fromarray((weight * 255).astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)
    ).astype(np.float32) / 255.0
    # Black-red-yellow, so strong contributors read as bright yellow.
    r = np.clip(1.6 * up, 0, 1)
    g = np.clip(1.6 * up - 0.45, 0, 1)
    b = np.clip(1.4 * up - 0.95, 0, 1)
    color = np.stack([r, g, b], axis=-1)
    base = rgb.astype(np.float32) / 255.0
    base = base * (0.28 + 0.22 * (1.0 - up[..., None]))
    out = np.clip(base + color * (0.15 + 0.85 * up[..., None]), 0, 1)
    yy, xx = divmod(q, grid)
    cy = int((yy + 0.5) * h / grid)
    cx = int((xx + 0.5) * w / grid)
    img = Image.fromarray((out * 255).round().astype(np.uint8))
    # White ring, black core, so the query stays visible on the yellow mass.
    px = img.load()
    rad, hole = 11, 4
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            d2 = dy * dy + dx * dx
            y, x = cy + dy, cx + dx
            if not (0 <= y < h and 0 <= x < w):
                continue
            if hole * hole <= d2 <= rad * rad and d2 >= (rad - 3) ** 2:
                px[x, y] = (255, 255, 255)
            elif d2 <= hole * hole:
                px[x, y] = (0, 0, 0)
    return np.asarray(img)


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = configuration.load_config(CFG)
    cfg.dataset.val_batch_size = 1
    cfg.dataset.num_val_workers = 0
    cfg.dataset.val_size = CLIP + 1
    dm = data.build(cfg.dataset, data_dir=DATA_DIR)
    dm.setup("validate")
    bb = cfg.model.encoder.backbone
    backbone = TimmExtractor(
        model=bb.model,
        pretrained=True,
        frozen=True,
        features=bb.features,
        model_kwargs=dict(bb.get("model_kwargs") or {}) or None,
    ).to(device).eval()

    target = None
    for n, batch in enumerate(dm.val_dataloader()):
        if n > CLIP:
            break
        if n != CLIP:
            continue
        video = batch["video"][0]
        seg = batch["segmentations"][0]
        ti = int(video.shape[0] // 2)
        frame = video[ti].to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            raw = extract_tokens(backbone, frame.unsqueeze(0), bb.features).float()[0]
        grid = int(round(raw.shape[0] ** 0.5))
        gt = patch_ids_ytvis(seg[ti].to(device), grid)
        target = (denorm_frame(frame.cpu()), raw.detach(), gt.detach(), grid)
        break
    if target is None:
        raise RuntimeError(f"clip {CLIP} was not loaded")
    rgb, raw, gt, grid = target
    p = relation_p(raw)
    qs = queries(gt, grid)
    print("queries", [(q // grid, q % grid) for q in qs], flush=True)
    panels = []
    for i, q in enumerate(qs):
        heat = p[q].reshape(grid, grid).cpu().numpy()
        panel = render(rgb, heat, q, grid)
        path = os.path.join(OUT, f"aggregation_q{i}.png")
        Image.fromarray(panel).save(path)
        print("wrote", path, "mass_top", float(np.sort(heat.ravel())[-10:].sum()), flush=True)
        panels.append(panel)
    gap = np.full((panels[0].shape[0], 14, 3), 255, dtype=np.uint8)
    strip = np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=1)
    Image.fromarray(strip).save(os.path.join(OUT, "feature_aggregation.png"))
    print("wrote", os.path.join(OUT, "feature_aggregation.png"), flush=True)


if __name__ == "__main__":
    main()
