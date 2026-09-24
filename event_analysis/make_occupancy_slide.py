"""Slide 2: three pictures for spectral occupancy. π is how much, q is where.

Uses the cached frame from make_slot_graph_real_slide.py. Numpy only.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path("/workspace/SlotCurri") if Path("/workspace/SlotCurri/slotcurri").exists() else Path(
    "/mnt/ssd2/hmlee/SlotCurri"
)
OUT = ROOT / "event_analysis/slide_featcur"
CACHE = OUT / "slot_graph_example.npz"

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


def load_example() -> dict:
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
                    pairs.append((i, j))
    return pairs


def build_S(z: np.ndarray, grid: int):
    n = grid * grid
    pairs = n8_undirected(grid)
    deg = np.zeros(n, dtype=np.float64)
    raw = []
    for i, j in pairs:
        r = max(0.0, float(np.dot(z[i], z[j])))
        deg[i] += r
        deg[j] += r
        raw.append((i, j, r))
    dinv = 1.0 / np.sqrt(np.clip(deg, 1e-8, None))
    S = np.zeros((n, n), dtype=np.float64)
    for i, j, r in raw:
        s = r * dinv[i] * dinv[j]
        S[i, j] = s
        S[j, i] = s
    return S


def apply_g(S: np.ndarray, a: np.ndarray, v: np.ndarray) -> np.ndarray:
    return a * (S @ (a * v))


def perron(S: np.ndarray, a: np.ndarray, n_iter: int = 40, eps: float = 1e-8):
    n = a.shape[0]
    q = np.ones(n, dtype=np.float64)
    q /= np.linalg.norm(q)
    for _ in range(n_iter):
        gq = apply_g(S, a, q)
        nrm = float(np.linalg.norm(gq))
        if nrm < eps:
            break
        q = gq / nrm
    if float(q.sum()) < 0:
        q = -q
    q = np.clip(q, 0.0, None)
    nrm = float(np.linalg.norm(q))
    if nrm > eps:
        q = q / nrm
    lam = float(q @ apply_g(S, a, q))
    return q, max(lam, 0.0)


def hide(ax) -> None:
    ax.set_axis_off()


def caption(ax, text: str, y: float = -0.08) -> None:
    ax.text(0.0, y, text, transform=ax.transAxes, fontsize=10, color=GRAY, ha="left", va="top")


def overlay_field(ax, rgb, field, grid, cmap="Oranges", alpha=0.62):
    h, w = rgb.shape[:2]
    ax.imshow(np.clip(rgb * 0.55, 0, 1), origin="upper")
    vis = field.reshape(grid, grid)
    ax.imshow(
        vis,
        cmap=cmap,
        origin="upper",
        extent=[0, w, h, 0],
        alpha=alpha,
        interpolation="bilinear",
        vmin=0.0,
        vmax=max(float(vis.max()), 1e-8),
        zorder=2,
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    hide(ax)


def pi_badge(ax, rgb, pi, color=RUST):
    h, w = rgb.shape[:2]
    ax.add_patch(
        Rectangle(
            (0.58 * w, 0.05 * h),
            0.38 * w,
            0.15 * h,
            facecolor="white",
            edgecolor=color,
            lw=1.6,
            zorder=8,
        )
    )
    ax.text(
        0.77 * w,
        0.125 * h,
        r"$\pi=%.2f$" % pi,
        fontsize=14,
        color=color,
        ha="center",
        va="center",
        zorder=9,
        fontweight="bold",
    )


def square_crop(rgb, field, grid, frac=0.58):
    h, w = rgb.shape[:2]
    f = field.reshape(grid, grid)
    ys, xs = np.where(f >= 0.15 * max(float(f.max()), 1e-8))
    if xs.size == 0:
        i = int(np.argmax(f))
        xs, ys = np.array([i % grid]), np.array([i // grid])
    cx = (float(xs.mean()) + 0.5) * (w / grid)
    cy = (float(ys.mean()) + 0.5) * (h / grid)
    side = int(round(frac * min(h, w)))
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    return rgb[y0 : y0 + side, x0 : x0 + side], (x0, y0, x0 + side, y0 + side)


def field_in_pixels(field, grid, rgb_hw, box):
    """Nearest-neighbor upsample of a patch field onto a pixel crop."""
    h, w = rgb_hw
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[y0:y1, x0:x1]
    px = np.clip((xx * grid) // w, 0, grid - 1)
    py = np.clip((yy * grid) // h, 0, grid - 1)
    return field.reshape(grid, grid)[py, px]


def draw_crop_field(ax, crop, fld, cmap):
    ch, cw = crop.shape[:2]
    ax.imshow(np.clip(crop * 0.55, 0, 1), origin="upper")
    ax.imshow(
        fld,
        cmap=cmap,
        origin="upper",
        extent=[0, cw, ch, 0],
        alpha=0.65,
        interpolation="bilinear",
        vmin=0.0,
        vmax=max(float(fld.max()), 1e-8),
        zorder=2,
    )
    ax.set_xlim(0, cw)
    ax.set_ylim(ch, 0)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    hide(ax)


def main():
    style()
    if not CACHE.is_file():
        raise SystemExit(f"missing cache {CACHE} — run make_slot_graph_real_slide.py first")
    ex = load_example()
    rgb = ex["rgb"]
    a = ex["a"].astype(np.float64)
    z = ex["z"].astype(np.float64)
    slot = ex["slot"]
    grid = int(round(z.shape[0] ** 0.5))
    assert grid * grid == z.shape[0]
    print("building S", grid, flush=True)
    S = build_S(z, grid)
    qs, pis = [], []
    for s in range(a.shape[0]):
        q_s, pi_s = perron(S, a[s])
        qs.append(q_s)
        pis.append(pi_s)
        print(f"  slot {s}  pi={pi_s:.4f}  q_peak={q_s.max():.4f}", flush=True)
    pis = np.array(pis)
    q = qs[slot]
    pi = float(pis[slot])
    other = int(np.argmin(pis))
    if other == slot:
        other = int(np.argsort(pis)[0 if slot != 0 else 1])
    print(f"use slot {slot} pi={pi:.4f}  contrast slot {other} pi={pis[other]:.4f}", flush=True)

    fig = plt.figure(figsize=(16.0, 9.0), dpi=180)
    fig.text(
        0.05,
        0.955,
        r"$\pi$ is How Much.   $q$ is Where.",
        fontsize=18,
        color=INK,
        ha="left",
        va="top",
    )

    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.05, 0.95, 1.25],
        wspace=0.10,
        left=0.04,
        right=0.99,
        top=0.84,
        bottom=0.18,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    overlay_field(ax_a, rgb, q, grid)
    ax_a.set_title(r"(a)  $q$ is Where", fontsize=14, color=INK, loc="left", pad=8)
    caption(ax_a, r"$G_s q=\lambda_1 q$,   $||q||_2=1$,   $q\geq 0$")

    inner_b = gs[0, 1].subgridspec(2, 1, hspace=0.16)
    ax_b1 = fig.add_subplot(inner_b[0, 0])
    ax_b2 = fig.add_subplot(inner_b[1, 0])
    overlay_field(ax_b1, rgb, qs[slot], grid)
    pi_badge(ax_b1, rgb, pi, color=RUST)
    ax_b1.set_title(r"(b)  $\pi$ is How Much", fontsize=14, color=INK, loc="left", pad=6)
    overlay_field(ax_b2, rgb, qs[other], grid)
    pi_badge(ax_b2, rgb, float(pis[other]), color=NAVY)
    caption(ax_b2, r"$\pi=\max(\lambda_1,0)=\max(q^{T} G_s q,\,0)$", y=-0.16)

    inner_c = gs[0, 2].subgridspec(1, 3, wspace=0.14)
    ax_c1 = fig.add_subplot(inner_c[0, 0])
    ax_c2 = fig.add_subplot(inner_c[0, 1])
    ax_c3 = fig.add_subplot(inner_c[0, 2])
    wmode = a[slot] * q
    crop, box = square_crop(rgb, wmode, grid, frac=0.62)
    h, w = rgb.shape[:2]
    draw_crop_field(ax_c1, crop, field_in_pixels(a[slot], grid, (h, w), box), "Oranges")
    draw_crop_field(ax_c2, crop, field_in_pixels(q, grid, (h, w), box), "Oranges")
    draw_crop_field(ax_c3, crop, field_in_pixels(wmode, grid, (h, w), box), "Blues")
    ax_c1.set_title(r"(c)  $a_s$", fontsize=14, color=INK, loc="left", pad=8)
    ax_c2.set_title(r"$q$", fontsize=14, color=INK, loc="center", pad=8)
    ax_c3.set_title(r"$a_s\odot q$", fontsize=14, color=INK, loc="center", pad=8)
    fig.canvas.draw()
    for left, right, mark in ((ax_c1, ax_c2, r"$\odot$"), (ax_c2, ax_c3, r"$=$")):
        b1 = left.get_position()
        b2 = right.get_position()
        fig.text(
            0.5 * (b1.x1 + b2.x0),
            0.5 * (b1.y0 + b1.y1),
            mark,
            fontsize=16,
            color=INK,
            ha="center",
            va="center",
        )
    fig.text(
        0.79,
        0.145,
        r"Affinity of the weighted mode:  $q^{T} G_s q=(a_s\odot q)^{T} S(a_s\odot q)$",
        fontsize=10,
        color=GRAY,
        ha="center",
    )

    fig.text(
        0.035,
        0.045,
        "The leading mode is the slot's place.  Its Perron root is the slot's mass.",
        fontsize=12,
        color=INK,
    )
    fig.text(
        0.035,
        0.018,
        f"YTVIS val  ·  v39lam1gu  ·  slot {slot} vs slot {other}  ·  {grid}x{grid} patches",
        fontsize=9,
        color=GRAY,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "08_spectral_occupancy.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
