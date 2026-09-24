"""Slide 3 on a real frame: grid → 8-nbr ReLU cosine → a_s-weighted G_s.

Order matches the talk slide. Needs the slotcurri image + YTVIS shards + a ckpt.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle

from slotcurri import configuration, data, models
from slotcurri.data.transforms import Denormalize

ROOT = Path("/workspace/SlotCurri") if Path("/workspace/SlotCurri/slotcurri").exists() else Path(
    "/mnt/ssd2/hmlee/SlotCurri"
)
OUT = ROOT / "event_analysis/slide_featcur"
CACHE = OUT / "slot_graph_example.npz"
CFG = ROOT / "configs/slotcurri/ytvis2021_attnmass_v39lam1gu.yaml"
CKPT = ROOT / "logs/_ytvis_attnmass_v39lam1gu/checkpoints/slotcurri_step=step=100000-v1.ckpt"
DATA_DIR = os.environ.get("VIDEOSAUR_DATA_DIR", "/workspace/dataset")

NAVY = "#1F3A5F"
RUST = "#C45C26"
INK = "#111111"
GRAY = "#6E6E6E"
N8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def denorm(frame: torch.Tensor) -> np.ndarray:
    x = Denormalize("image")(frame.detach().cpu()).clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def slice_time(batch: dict, idx: list[int]) -> dict:
    t = int(batch["video"].shape[1])
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.ndim >= 2 and v.shape[1] == t:
            out[k] = v[:, idx]
        else:
            out[k] = v
    return out


def save_example(ex: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        rgb=ex["rgb"],
        a=ex["a"],
        z=ex["z"],
        slot=np.array(ex["slot"]),
        clip=np.array(ex["clip"]),
    )


def load_example() -> dict | None:
    if not CACHE.is_file():
        return None
    d = np.load(CACHE)
    return {
        "rgb": d["rgb"],
        "a": d["a"],
        "z": d["z"],
        "slot": int(d["slot"]),
        "clip": int(d["clip"]),
    }


def n8_undirected(grid: int):
    pairs = []
    for y in range(grid):
        for x in range(grid):
            i = y * grid + x
            for dy, dx in N8:
                y2, x2 = y + dy, x + dx
                if not (0 <= y2 < grid and 0 <= x2 < grid):
                    continue
                j = y2 * grid + x2
                if j > i:
                    pairs.append((i, j, x, y, x2, y2))
    return pairs


def peak_component(a: np.ndarray, grid: int, ath: float) -> np.ndarray:
    """Keep the 8-connected blob around argmax(a). Drops stray high-a patches."""
    from collections import deque

    n = grid * grid
    keep = np.zeros(n, dtype=bool)
    vis = np.zeros(n, dtype=bool)
    q0 = int(np.argmax(a))
    dq = deque([q0])
    vis[q0] = True
    while dq:
        i = dq.popleft()
        if a[i] < ath:
            continue
        keep[i] = True
        y, x = i // grid, i % grid
        for dy, dx in N8:
            y2, x2 = y + dy, x + dx
            if not (0 <= y2 < grid and 0 <= x2 < grid):
                continue
            j = y2 * grid + x2
            if not vis[j]:
                vis[j] = True
                dq.append(j)
    return keep


def centers(grid: int, h: int, w: int):
    ys = (np.arange(grid) + 0.5) * (h / grid)
    xs = (np.arange(grid) + 0.5) * (w / grid)
    return xs, ys


def slot_score(a: np.ndarray) -> np.ndarray:
    """Prefer a compact, high-peak slot over a faint smear."""
    mass = a.mean(axis=-1)
    peak = a.max(axis=-1)
    p = a / np.clip(a.sum(axis=-1, keepdims=True), 1e-8, None)
    ent = -(p * np.log(np.clip(p, 1e-8, None))).sum(axis=-1)
    ent = ent / np.log(a.shape[-1])
    return peak * np.sqrt(np.clip(mass, 0, None)) * (1.0 - ent)


@torch.no_grad()
def pick_example(model, loader, device, max_clips: int = 16):
    best = None
    best_sc = -1.0
    n = 0
    for batch in loader:
        if n >= max_clips:
            break
        if "batch_padding_mask" in batch:
            m = batch["batch_padding_mask"]
            if torch.is_tensor(m) and bool(m.any()):
                continue
        video = batch["video"]
        t = int(video.shape[1])
        if t < 1:
            continue
        mid = t // 2
        idx = sorted({max(0, mid - 1), mid, min(t - 1, mid + 1)})
        b = slice_time(batch, idx)
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        cycle = getattr(model, "cyclic_inference", False)
        out = model.forward(b, train=False, cycle=cycle)
        att = out["processor"]["state_attn_mask"]
        bind = out["encoder"].get("backbone_key")
        if bind is None:
            bind = out["encoder"]["backbone_features"]
        # att: (B,T,S,N) or (B,S,N)
        if att.ndim == 3:
            att_t = att[0]
            bind_t = bind[0]
            rgb_t = b["video"][0, 0]
            fi = 0
        else:
            fi = min(1, att.shape[1] - 1)
            att_t = att[0, fi]
            bind_t = bind[0, fi]
            rgb_t = b["video"][0, fi]
        a = att_t.float().cpu().numpy()
        sc = float(slot_score(a).max())
        n += 1
        print(f"  clip {n}  t={t}  slot_score={sc:.3f}", flush=True)
        if sc > best_sc:
            best_sc = sc
            best = {
                "rgb": denorm(rgb_t),
                "a": a,
                "z": F.normalize(bind_t.float(), dim=-1).cpu().numpy(),
                "clip": n,
                "frame": fi,
            }
    if best is None:
        raise RuntimeError("no val clip")
    s = int(slot_score(best["a"]).argmax())
    best["slot"] = s
    print(f"picked clip {best['clip']} slot {s} score {best_sc:.3f}", flush=True)
    return best


def draw_grid(ax, grid, h, w, color="white", alpha=0.35, lw=0.6):
    for k in range(grid + 1):
        y = k * h / grid
        x = k * w / grid
        ax.plot([0, w], [y, y], color=color, lw=lw, alpha=alpha, zorder=3)
        ax.plot([x, x], [0, h], color=color, lw=lw, alpha=alpha, zorder=3)


def caption(ax, text: str) -> None:
    ax.text(
        0.0,
        -0.055,
        text,
        transform=ax.transAxes,
        fontsize=10,
        color=GRAY,
        ha="left",
        va="top",
    )


def panel_grid(ax, rgb, grid, box=None):
    h, w = rgb.shape[:2]
    ax.imshow(rgb, origin="upper")
    draw_grid(ax, grid, h, w)
    if box is not None:
        x0, y0, x1, y1 = box
        pw, ph = w / grid, h / grid
        ax.add_patch(
            Rectangle(
                (x0 * pw, y0 * ph),
                (x1 - x0 + 1) * pw,
                (y1 - y0 + 1) * ph,
                fill=False,
                edgecolor=RUST,
                lw=1.8,
                zorder=5,
            )
        )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()
    ax.set_title("(a)  Patches sit on a square grid", fontsize=13, color=INK, loc="left", pad=8)
    caption(ax, "DINOv2 tokens on the frame  ·  14 px cells")


def crop_around(q: int, grid: int, rad: int = 4):
    qx, qy = q % grid, q // grid
    x0 = max(0, qx - rad)
    y0 = max(0, qy - rad)
    x1 = min(grid - 1, qx + rad)
    y1 = min(grid - 1, qy + rad)
    return qx, qy, x0, y0, x1, y1


def panel_relu(ax, rgb, z, grid, q):
    h, w = rgb.shape[:2]
    qx, qy, x0, y0, x1, y1 = crop_around(q, grid, rad=4)
    pw, ph = w / grid, h / grid
    ypix0, ypix1 = int(round(y0 * ph)), int(round((y1 + 1) * ph))
    xpix0, xpix1 = int(round(x0 * pw)), int(round((x1 + 1) * pw))
    crop = rgb[ypix0:ypix1, xpix0:xpix1]
    ch, cw = crop.shape[:2]
    ax.imshow(np.clip(crop * 0.78, 0, 1), origin="upper")
    gy, gx = y1 - y0 + 1, x1 - x0 + 1
    for k in range(gx + 1):
        x = k * cw / gx
        ax.plot([x, x], [0, ch], color="white", lw=0.7, alpha=0.45, zorder=3)
    for k in range(gy + 1):
        y = k * ch / gy
        ax.plot([0, cw], [y, y], color="white", lw=0.7, alpha=0.45, zorder=3)
    # local patch centers in crop pixels
    def loc(x, y):
        return ((x - x0) + 0.5) * cw / gx, ((y - y0) + 0.5) * ch / gy

    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            i = y * grid + x
            for dy, dx in N8:
                y2, x2 = y + dy, x + dx
                if not (x0 <= x2 <= x1 and y0 <= y2 <= y1):
                    continue
                if (y2, x2) <= (y, x):
                    continue
                r = max(0.0, float(np.dot(z[i], z[y2 * grid + x2])))
                t = float(np.clip((r - 0.35) / 0.60, 0.0, 1.0)) ** 1.4
                if t < 0.04:
                    continue
                x_a, y_a = loc(x, y)
                x_b, y_b = loc(x2, y2)
                ax.plot(
                    [x_a, x_b],
                    [y_a, y_b],
                    color=(1.0, 1.0, 1.0, 0.08 + 0.42 * t),
                    lw=0.35 + 1.7 * t,
                    zorder=4,
                    solid_capstyle="round",
                )
    # one patch: 8-neighbor stencil, width ∝ ReLU cosine
    cx, cy = loc(qx, qy)
    ax.add_patch(Circle((cx, cy), 0.38 * cw / gx, facecolor=RUST, edgecolor="white", lw=1.2, zorder=6))
    for dy, dx in N8:
        y2, x2 = qy + dy, qx + dx
        if not (x0 <= x2 <= x1 and y0 <= y2 <= y1):
            continue
        r = max(0.0, float(np.dot(z[q], z[y2 * grid + x2])))
        nx, ny = loc(x2, y2)
        ax.plot([cx, nx], [cy, ny], color=RUST, lw=0.7 + 4.2 * r, zorder=7, solid_capstyle="round")
        ax.add_patch(
            Circle((nx, ny), 0.26 * cw / gx, facecolor="white", edgecolor=RUST, lw=1.1, zorder=8)
        )
    ax.set_xlim(0, cw)
    ax.set_ylim(ch, 0)
    ax.set_axis_off()
    ax.set_title("(b)  Connect 8 neighbors with ReLU cosine", fontsize=13, color=INK, loc="left", pad=8)
    caption(ax, r"$R_{ij}=\mathrm{ReLU}(z_i^{T}\,z_j)$  if $j\in N_8(i)$, else 0")


def panel_gate(ax, rgb, z, a, grid):
    h, w = rgb.shape[:2]
    xs, ys = centers(grid, h, w)
    pw, ph = w / grid, h / grid
    ax.imshow(rgb * 0.50, origin="upper")
    a_hw = a.reshape(grid, grid)
    ax.imshow(
        a_hw,
        cmap="Blues",
        origin="upper",
        extent=[0, w, h, 0],
        alpha=0.28,
        interpolation="nearest",
        vmin=0.0,
        vmax=max(float(a.max()), 1e-6),
        zorder=2,
    )
    pairs = n8_undirected(grid)
    segs, widths, cols = [], [], []
    ath = 0.40 * float(a.max())
    keep = peak_component(a, grid, ath)
    for i, j, x, y, x2, y2 in pairs:
        if not (keep[i] and keep[j]):
            continue
        r = max(0.0, float(np.dot(z[i], z[j])))
        g = float(a[i] * a[j] * r)
        if g < 0.08 * ath * ath:
            continue
        segs.append([(xs[x], ys[y]), (xs[x2], ys[y2])])
        t = float(np.clip(g, 0.0, 1.0))
        widths.append(0.35 + 1.1 * t)
        cols.append((1.0, 0.92, 0.72, 0.25 + 0.45 * t))
    if segs:
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=widths, zorder=4))
    r0 = 0.28 * min(pw, ph)
    for i in range(grid * grid):
        if not keep[i]:
            continue
        x, y = i % grid, i // grid
        ax.add_patch(
            Circle(
                (xs[x], ys[y]),
                r0 * (0.40 + 0.70 * float(a[i])),
                facecolor=RUST,
                edgecolor="white",
                lw=0.35,
                alpha=0.90,
                zorder=6,
            )
        )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()
    ax.set_title(r"(c)  Weight by this slot's $a_s$", fontsize=13, color=INK, loc="left", pad=8)
    caption(ax, r"$S=D^{-1/2}RD^{-1/2}$" + "    " + r"$G_s=\mathrm{diag}(a_s)\,S\,\mathrm{diag}(a_s)$")


def main():
    style()
    ex = load_example()
    if ex is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device", device, flush=True)
        cfg = configuration.load_config(str(CFG))
        cfg.model.visualize = False
        cfg.dataset.val_batch_size = 1
        cfg.dataset.num_val_workers = 0
        cfg.dataset.val_size = 80
        model = models.build(cfg.model, cfg.optimizer)
        model.load_weights_from_checkpoint(str(CKPT))
        model.to(device).eval()
        dm = data.build(cfg.dataset, data_dir=DATA_DIR)
        dm.setup("validate")
        ex = pick_example(model, dm.val_dataloader(), device)
        save_example(ex)
    else:
        print(f"cached clip {ex['clip']} slot {ex['slot']}", flush=True)
    a = ex["a"]
    z = ex["z"]
    rgb = ex["rgb"]
    grid = int(round(z.shape[0] ** 0.5))
    assert grid * grid == z.shape[0], z.shape
    s = ex["slot"]
    a_s = a[s]
    q = int(a_s.argmax())
    _, _, x0, y0, x1, y1 = crop_around(q, grid, rad=4)

    fig = plt.figure(figsize=(16.0, 9.0), dpi=180)
    fig.suptitle(
        "Each slot induces an 8-neighbor patch graph",
        fontsize=18,
        color=INK,
        x=0.035,
        ha="left",
        y=0.975,
    )
    gs = fig.add_gridspec(1, 3, wspace=0.10, left=0.03, right=0.99, top=0.86, bottom=0.22)
    panel_grid(fig.add_subplot(gs[0, 0]), rgb, grid, box=(x0, y0, x1, y1))
    panel_relu(fig.add_subplot(gs[0, 1]), rgb, z, grid, q)
    panel_gate(fig.add_subplot(gs[0, 2]), rgb, z, a_s, grid)
    fig.text(
        0.035,
        0.045,
        r"$a_s$: last Slot-Attention iteration, softmax over slots.   "
        "A slot that owns little mass induces a small graph.  That scale is the occupancy signal.",
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.035,
        0.018,
        f"YTVIS val  ·  v39lam1gu  ·  slot {s}  ·  {grid}×{grid} patches",
        fontsize=9,
        color=GRAY,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "07_slot_graph_n8.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
