"""Feature-aggregation inset in the style of the diagram sketch.

Frozen-feature PCA is the background. Blue squares are patches whose features
are averaged in. The yellow circle is the patch that receives that average.
Lines connect the two. One object is cropped so the markers stay readable.
"""

from __future__ import annotations

import os
import sys

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_dino_vis import denorm_frame, pca_basis, pca_map  # noqa: E402
from featcur_strength_probe import extract_tokens, patch_ids_ytvis  # noqa: E402
from slotcurri import configuration, data
from slotcurri.modules.encoders import TimmExtractor

CFG = "configs/slotcurri/ytvis2021_attnmass_v39lam1gu_xframe_lin40.yaml"
DATA_DIR = "/workspace/dataset"
OUT = "event_analysis/fig_blanks"
CLIP = 15
BLUE = "#2c3f9e"
GREEN = "#2f9e4f"
GOLD = "#f2c14b"
N_SIMILAR = 6


def load_clip(device):
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
    for n, batch in enumerate(dm.val_dataloader()):
        if n != CLIP:
            if n > CLIP:
                break
            continue
        video = batch["video"][0]
        seg = batch["segmentations"][0]
        ti = int(video.shape[0] // 2)
        frame = video[ti].to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            raw = extract_tokens(backbone, frame.unsqueeze(0), bb.features).float()[0]
        grid = int(round(raw.shape[0] ** 0.5))
        gt = patch_ids_ytvis(seg[ti].to(device), grid)
        return raw.detach(), gt.detach(), grid
    raise RuntimeError(f"clip {CLIP} was not loaded")


def object_mask(gt: torch.Tensor, grid: int):
    """Largest instance, and a query patch near its center that is on the mask."""
    ids = [int(i) for i in gt.unique().tolist() if int(i) >= 0]
    ids.sort(key=lambda i: int((gt == i).sum()), reverse=True)
    mask = (gt == ids[0]).reshape(grid, grid)
    ys, xs = torch.nonzero(mask, as_tuple=True)
    y = int(ys.float().mean().round().item())
    x = int(xs.float().mean().round().item())
    y = min(max(y, 0), grid - 1)
    x = min(max(x, 0), grid - 1)
    if not bool(mask[y, x]):
        # Mean can fall in a hole. Use the mask patch closest to the mean.
        d = (ys.float() - y) ** 2 + (xs.float() - x) ** 2
        k = int(d.argmin())
        y, x = int(ys[k]), int(xs[k])
    return mask, y * grid + x, int(ids[0])


def similar_patches(raw: torch.Tensor, mask: torch.Tensor, q: int, grid: int):
    """Similar patches across the frame, spaced in angle so the lines fan out evenly."""
    z = F.normalize(raw.float(), dim=-1)
    cos = (z[q] * z).sum(-1).clamp_min(0)
    qy, qx = divmod(q, grid)
    order = torch.argsort(cos, descending=True)
    picked = []
    angs = []
    for idx in order.tolist():
        if int(idx) == q or float(cos[idx]) < 0.28:
            if float(cos[idx]) < 0.28 and int(idx) != q:
                break
            continue
        y, x = divmod(int(idx), grid)
        if min(y, x, grid - 1 - y, grid - 1 - x) < 2:
            continue
        if (y - qy) ** 2 + (x - qx) ** 2 < 16:
            continue
        ang = math.atan2(y - qy, x - qx)
        if any((y - py) ** 2 + (x - px) ** 2 < 30 for py, px in picked):
            continue
        if any(abs((ang - a + math.pi) % (2 * math.pi) - math.pi) < 0.42 for a in angs):
            continue
        picked.append((y, x))
        angs.append(ang)
        if len(picked) >= N_SIMILAR:
            break
    on = mask.reshape(-1)
    return picked, float(cos[on].mean())


def mask_edge(mask: np.ndarray) -> np.ndarray:
    """True on pixels that sit on the boundary of a binary mask. Row 0 is the top."""
    edge = np.zeros_like(mask, dtype=bool)
    edge[1:] |= mask[1:] & ~mask[:-1]
    edge[:-1] |= mask[:-1] & ~mask[1:]
    edge[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]
    edge[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]
    thick = edge.copy()
    thick[1:] |= edge[:-1]
    thick[:-1] |= edge[1:]
    thick[:, 1:] |= edge[:, :-1]
    thick[:, :-1] |= edge[:, 1:]
    return thick


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw, gt, grid = load_clip(device)
    bg_path = os.path.join(OUT, "frozen_pca.png")
    image = Image.open(bg_path).convert("RGB")
    width, height = image.size
    if width % grid != 0 or height % grid != 0:
        raise RuntimeError(f"frozen_pca.png {width}x{height} is not a multiple of grid {grid}")
    scale = width // grid
    mask_t, q, oid = object_mask(gt, grid)
    similars, mean_cos = similar_patches(raw, mask_t, q, grid)
    print(
        "object", oid, "query", divmod(q, grid), "similar", similars,
        "mean_cos", round(mean_cos, 3), "scale", scale, flush=True,
    )
    mask = np.repeat(np.repeat(mask_t.cpu().numpy().astype(np.float32), scale, axis=0), scale, axis=1)
    pad = np.pad(mask, 6, mode="edge")
    soft = np.zeros_like(mask)
    for dy in range(13):
        for dx in range(13):
            soft += pad[dy : dy + height, dx : dx + width]
    soft /= 169.0
    edge = (soft > 0.38) & (soft < 0.62)
    arr = np.asarray(image).copy()
    arr[edge] = (255, 255, 255)

    zoom = 3
    canvas = Image.fromarray(arr).resize((width * zoom, height * zoom), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(canvas)

    def center(y, x):
        return np.array([(x + 0.5) * scale * zoom, (y + 0.5) * scale * zoom])

    def shorten(p0, p1, gap0, gap1):
        v = p1 - p0
        length = float(np.hypot(v[0], v[1]))
        if length <= gap0 + gap1 + 1:
            return p0, p1
        u = v / length
        return p0 + u * gap0, p1 - u * gap1

    qy, qx = divmod(q, grid)
    qxy = center(qy, qx)
    box = 16 * zoom
    rad = 18 * zoom
    gap_q, gap_s = float(rad + 3 * zoom), float(box + 3 * zoom)
    for y, x in similars:
        a, b = shorten(center(y, x), qxy, gap_s, gap_q)
        draw.line([tuple(a), tuple(b)], fill="white", width=9 * zoom)
        draw.line([tuple(a), tuple(b)], fill="#f0b429", width=5 * zoom)
    bottom = max(similars, key=lambda yx: yx[0])
    for y, x in similars:
        cx, cy = center(y, x)
        fill = GREEN if (y, x) == bottom else BLUE
        draw.rounded_rectangle(
            (cx - box, cy - box, cx + box, cy + box),
            radius=4 * zoom,
            fill=fill,
            outline="white",
            width=2 * zoom,
        )
    draw.ellipse(
        (qxy[0] - rad, qxy[1] - rad, qxy[0] + rad, qxy[1] + rad),
        fill=GOLD,
        outline="white",
        width=2 * zoom,
    )
    image = canvas.resize((width, height), Image.Resampling.LANCZOS)
    path = os.path.join(OUT, "feature_aggregation_schematic.png")
    image.save(path)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
