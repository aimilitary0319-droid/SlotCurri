"""Slot graphs on one frozen-PCA frame, plus each slot's attention map.

The four highest-occupancy slots keep their colors. Structures are the s=0
8-neighbor links inside each slot's attention. Attention panels use the same
frame and the same colors.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_aggregation_schematic import CLIP, load_clip, object_mask  # noqa: E402
from fig_blank_panels import (  # noqa: E402
    CFG,
    CKPT,
    OUT,
    SLOT_COLORS,
    level_tokens,
    neighbor_edges,
)
from slotcurri import configuration, models
from slotcurri.modules.video import spectral_graph_n8_eigs


def load_video_window(device):
    cfg = configuration.load_config(CFG)
    cfg.dataset.val_batch_size = 1
    cfg.dataset.num_val_workers = 0
    cfg.dataset.val_size = CLIP + 1
    from slotcurri import data

    dm = data.build(cfg.dataset, data_dir="/workspace/dataset")
    dm.setup("validate")
    for n, batch in enumerate(dm.val_dataloader()):
        if n == CLIP:
            video = batch["video"][0]
            ti = int(video.shape[0] // 2)
            a = max(0, ti - 3)
            b = min(int(video.shape[0]), a + 6)
            a = max(0, b - 6)
            return video[a:b], ti - a, cfg
        if n > CLIP:
            break
    raise RuntimeError(f"clip {CLIP} missing")


def attention_panel(bg: np.ndarray, heat: np.ndarray, color, scale: int) -> np.ndarray:
    weight = np.repeat(np.repeat(heat.astype(np.float32), scale, axis=0), scale, axis=1)
    weight = weight / max(float(weight.max()), 1e-8)
    # Softmax leaves a small value on every patch. Drop that floor so the
    # map shows where the slot actually concentrates.
    shown = np.clip((weight - 0.35) / 0.65, 0.0, 1.0) ** 0.65
    alpha = shown[..., None]
    tint = np.array(color, dtype=np.float32)
    out = bg.astype(np.float32) * (1.0 - 0.88 * alpha) + tint * (0.88 * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw, gt, grid = load_clip(device)
    person, _q, _oid = object_mask(gt, grid)
    person = person.cpu().numpy().reshape(-1)
    clip, local_t, cfg = load_video_window(device)
    leveled = level_tokens(raw)
    edges = neighbor_edges(F.normalize(leveled.float(), dim=-1), grid)

    print("loading model", flush=True)
    model = models.build(cfg.model, cfg.optimizer)
    ckpt = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    with torch.no_grad():
        outputs = model({"video": clip.unsqueeze(0).to(device)}, train=False)
    att = outputs["processor"]["state_attn_mask"]
    att_f = att[0, local_t].float() if att.ndim == 4 else att[local_t].float()
    pi, _, _, _q1 = spectral_graph_n8_eigs(
        att_f.detach().unsqueeze(0).cpu(),
        raw.detach().unsqueeze(0).cpu(),
        use_lambda1=True,
        return_q1=True,
    )
    pi = pi[0]
    order = torch.argsort(pi, descending=True)
    att_np = att_f.cpu().numpy()
    winner = att_np.argmax(axis=0)
    counts = np.bincount(winner, minlength=att_np.shape[0])
    person_votes = np.bincount(winner[person], minlength=att_np.shape[0])
    background = int(np.argmax(counts))
    # The right-hand player is split across slots. Those slots, not the top
    # band that previously took the blue panel, are drawn in blue.
    blue_slots = [int(i) for i, v in enumerate(person_votes) if int(v) >= 15 and int(i) != background]
    rest = [int(i) for i in np.argsort(-counts) if int(i) != background and int(i) not in blue_slots]

    def slot_centroid(slot):
        idx = np.nonzero(winner == slot)[0]
        return float((idx // grid).mean()), float((idx % grid).mean())

    # Purple is the slot that already sits in the bottom-left, not the top band.
    purple = max(rest, key=lambda s: slot_centroid(s)[0] - 0.5 * slot_centroid(s)[1])
    mid = [s for s in rest if s != purple][:2]
    groups = [(background,), (mid[0],), (mid[1],), tuple(blue_slots), (purple,)]
    palette = (
        (46, 155, 79),
        (232, 146, 42),
        (225, 91, 138),
        (61, 126, 219),
        (142, 68, 196),
    )
    print("groups", groups, "person_votes", person_votes.tolist(), flush=True)

    scale = 14
    width = height = grid * scale

    def center(y, x):
        return ((x + 0.5) * scale, (y + 0.5) * scale)

    # Structure is only the 8-neighbor graph inside a slot's own attention.
    # It is not the argmax coloring of every patch.
    attn_panels = []
    gap_d = np.full((height, 8, 3), 12, dtype=np.uint8)
    segments = []
    for group, color in zip(groups, palette):
        heat = att_np[list(group)].sum(axis=0)
        amax = float(heat.max()) + 1e-8
        kept = []
        for y1, x1, y2, x2, _s in edges:
            i1 = y1 * grid + x1
            i2 = y2 * grid + x2
            if heat[i1] < 0.28 * amax or heat[i2] < 0.28 * amax:
                continue
            kept.append((y1, x1, y2, x2))
        segments.append(kept)

        heat_map = heat.reshape(grid, grid) / amax
        rgb = np.array(color, dtype=np.float32)
        panel = np.zeros((height, width, 3), dtype=np.float32)
        for y in range(grid):
            for x in range(grid):
                t = float(heat_map[y, x]) ** 0.6
                panel[y * scale : (y + 1) * scale, x * scale : (x + 1) * scale] = (0.08 + 0.92 * t) * rgb
        attn_panels.append(np.clip(panel, 0, 255).astype(np.uint8))

    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    pen = ImageDraw.Draw(canvas)
    for kept, color in zip(segments, palette):
        for y1, x1, y2, x2 in kept:
            pen.line([center(y1, x1), center(y2, x2)], fill=color + (255,), width=2)
    struct_path = os.path.join(OUT, "slot_structures.png")
    canvas.save(struct_path)
    print("wrote", struct_path, flush=True)

    # One frame, same groups and colors as the structure graph.
    heats = []
    amaxes = []
    for group in groups:
        heat = att_np[list(group)].sum(axis=0)
        heats.append(heat)
        amaxes.append(float(heat.max()) + 1e-8)
    attn = np.zeros((height, width, 4), dtype=np.float32)
    for y in range(grid):
        for x in range(grid):
            i = y * grid + x
            scores = [heats[k][i] / amaxes[k] for k in range(len(groups))]
            k = int(np.argmax(scores))
            t = scores[k]
            if t < 0.28:
                continue
            rgb = np.array(palette[k], dtype=np.float32)
            pix = 255 * (1.0 - t) + rgb * t
            attn[y * scale : (y + 1) * scale, x * scale : (x + 1) * scale, :3] = pix
            attn[y * scale : (y + 1) * scale, x * scale : (x + 1) * scale, 3] = 255
    attn_path = os.path.join(OUT, "slot_attention.png")
    Image.fromarray(np.clip(attn, 0, 255).astype(np.uint8), mode="RGBA").save(attn_path)
    print("wrote", struct_path, flush=True)
    print("wrote", attn_path, flush=True)


if __name__ == "__main__":
    main()
