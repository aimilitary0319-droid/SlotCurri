"""Draw the s=0 affinity graph on fl_pca.png.

Features are fully leveled (s=0). The 8-neighbor graph is recomputed on those
tokens. Row 0 is the top of fl_pca.png.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_aggregation_schematic import load_clip  # noqa: E402
from fig_blank_panels import level_tokens, neighbor_edges  # noqa: E402

OUT = "event_analysis/fig_blanks"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw, _gt, grid = load_clip(device)
    # s=0: X^bind = P X. Occupancy reads the 8-neighbor graph of these tokens.
    leveled = level_tokens(raw)
    z = F.normalize(leveled.float(), dim=-1)
    edges = neighbor_edges(z, grid)
    sims = np.array([e[4] for e in edges], dtype=np.float32)
    # After leveling, almost every neighbor is similar. The lower tail is where
    # an edge crosses an object boundary, so those links are left undrawn.
    cut = float(np.quantile(sims, 0.55))
    span = max(float(sims.max()) - cut, 1e-6)
    tone = np.clip((sims - cut) / span, 0.0, 1.0) ** 0.55

    bg_path = os.path.join(OUT, "fl_pca.png")
    image = Image.open(bg_path).convert("RGB")
    width, height = image.size
    if width % grid != 0 or height % grid != 0:
        raise RuntimeError(f"{bg_path} {width}x{height} is not a multiple of grid {grid}")
    scale = width // grid

    def center(y, x):
        return ((x + 0.5) * scale, (y + 0.5) * scale)

    draw = ImageDraw.Draw(image)
    for i in np.argsort(tone):
        t = float(tone[i])
        if t <= 0.0:
            continue
        y1, x1, y2, x2, _s = edges[i]
        g = int(round(90 * (1.0 - t)))
        width_px = 2 if t > 0.45 else 1
        draw.line([center(y1, x1), center(y2, x2)], fill=(g, g, g), width=width_px)

    path = os.path.join(OUT, "scene_S.png")
    image.save(path)
    print(
        "wrote", path,
        "edges", int((tone > 0).sum()),
        "cut", round(cut, 4),
        "sim", round(float(sims.min()), 3), round(float(sims.mean()), 3), round(float(sims.max()), 3),
        "scale", scale,
        flush=True,
    )


if __name__ == "__main__":
    main()
