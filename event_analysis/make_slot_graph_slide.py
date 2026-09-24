"""Slide figure: each slot induces an 8-neighbor patch graph G_s.

Visual, not a wall of text. 16:9 PNG for the occupancy / n8 talk slide.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "event_analysis" / "slide_featcur"

NAVY = "#1F3A5F"
TEAL = "#3C6E71"
INK = "#111111"
GRAY = "#6E6E6E"
RULE = "#C8CCD0"
FILL = "#F4F6F8"
SOFT = "#E8EEF4"
N8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


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


def plateau(i, j, cx, cy, rx, ry, edge=0.7):
    d = np.sqrt(((i - cx) / rx) ** 2 + ((j - cy) / ry) ** 2)
    t = np.clip((1.0 - d) / (edge / max(rx, ry)), 0.0, 1.0)
    inside = d <= 1.0 - edge / max(rx, ry)
    return np.where(inside, 1.0, np.where(d >= 1.0, 0.0, t * t * (3.0 - 2.0 * t)))


def make_fields(n=16):
    yy, xx = np.mgrid[0:n, 0:n]
    obj = plateau(xx, yy, 8.2, 8.0, 4.6, 5.2, edge=1.1)
    small = plateau(xx, yy, 6.5, 6.8, 1.7, 1.55, edge=0.5)
    large_a = 0.04 + 0.92 * obj
    small_a = 0.03 + 0.93 * small
    rng = np.random.default_rng(2)
    ghost_a = 0.07 + 0.05 * rng.random((n, n))
    # Features: object vs background, so ReLU-cosine is 1 inside / 0 across.
    z = np.zeros((n, n, 2), dtype=np.float64)
    z[..., 0] = 1.0 - obj
    z[..., 1] = obj
    z = z / np.clip(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8, None)
    z_small = np.zeros((n, n, 2), dtype=np.float64)
    z_small[..., 0] = 1.0 - small
    z_small[..., 1] = small
    z_small = z_small / np.clip(np.linalg.norm(z_small, axis=-1, keepdims=True), 1e-8, None)
    # Ghost has no exclusive region: mild spatial texture, mostly similar neighbors.
    z_g = np.zeros((n, n, 2), dtype=np.float64)
    z_g[..., 0] = 1.0
    z_g[..., 1] = 0.15 * np.sin(xx / 2.2) * np.sin(yy / 2.4)
    z_g = z_g / np.clip(np.linalg.norm(z_g, axis=-1, keepdims=True), 1e-8, None)
    return large_a, small_a, ghost_a, z, z_small, z_g


def n8_edges(a, z, thr=None):
    n = a.shape[0]
    if thr is None:
        thr = 0.035 * float(np.clip(a.max(), 0.05, 1.0) ** 2)
    segs, widths, alphas = [], [], []
    for y in range(n):
        for x in range(n):
            for dy, dx in N8:
                y2, x2 = y + dy, x + dx
                if not (0 <= y2 < n and 0 <= x2 < n):
                    continue
                if (y2, x2) <= (y, x):
                    continue
                r = max(0.0, float(np.dot(z[y, x], z[y2, x2])))
                w = float(a[y, x] * a[y2, x2] * r)
                if w < thr:
                    continue
                segs.append([(x, y), (x2, y2)])
                widths.append(0.4 + 2.6 * w)
                alphas.append(0.18 + 0.82 * min(1.0, w / 0.85))
    return segs, widths, alphas


def draw_grid(ax, n, color=RULE, lw=0.45):
    for k in range(n + 1):
        ax.plot([-0.5, n - 0.5], [k - 0.5, k - 0.5], color=color, lw=lw, zorder=0)
        ax.plot([k - 0.5, k - 0.5], [-0.5, n - 0.5], color=color, lw=lw, zorder=0)


def draw_graph(ax, a, z, title, note, cmap_vmax=1.0, node_scale=42.0):
    n = a.shape[0]
    ax.set_aspect("equal")
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(n - 0.3, -0.7)
    ax.axis("off")
    draw_grid(ax, n)
    ax.imshow(
        a,
        cmap="Blues",
        vmin=0.0,
        vmax=cmap_vmax,
        origin="upper",
        extent=[-0.5, n - 0.5, n - 0.5, -0.5],
        interpolation="nearest",
        zorder=1,
        alpha=0.92,
    )
    segs, widths, alphas = n8_edges(a, z)
    if segs:
        lc = LineCollection(
            segs,
            colors=[(0.12, 0.22, 0.37, al) for al in alphas],
            linewidths=widths,
            zorder=3,
        )
        ax.add_collection(lc)
    ys, xs = np.mgrid[0:n, 0:n]
    sizes = 6.0 + node_scale * np.clip(a, 0, 1) ** 1.2
    ax.scatter(
        xs.ravel(),
        ys.ravel(),
        s=sizes.ravel(),
        c="#1F3A5F",
        alpha=0.35 + 0.55 * np.clip(a, 0, 1).ravel(),
        linewidths=0,
        zorder=4,
    )
    ax.set_title(title, fontsize=13, color=INK, loc="left", pad=8)
    ax.text(0.0, n + 0.15, note, fontsize=10, color=GRAY, transform=ax.transData, va="bottom")


def panel_n8(ax):
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"(a)  8 neighbors of a patch", fontsize=13, color=INK, loc="left", pad=8)
    # faint 3x3 cells
    for y in range(-1, 2):
        for x in range(-1, 2):
            ax.add_patch(
                Rectangle(
                    (x - 0.46, y - 0.46),
                    0.92,
                    0.92,
                    facecolor=SOFT if (x, y) != (0, 0) else "#D5DEEA",
                    edgecolor=RULE,
                    lw=0.8,
                    zorder=1,
                )
            )
    for dy, dx in N8:
        ax.plot([0, dx], [0, dy], color=NAVY, lw=1.7, zorder=2, solid_capstyle="round")
    for dy, dx in N8:
        ax.add_patch(Circle((dx, dy), 0.18, facecolor="white", edgecolor=NAVY, lw=1.3, zorder=3))
    ax.add_patch(Circle((0, 0), 0.26, facecolor=NAVY, edgecolor=NAVY, lw=1.2, zorder=4))
    ax.text(0, 0, r"$i$", color="white", ha="center", va="center", fontsize=11, zorder=5)
    ax.text(0, -1.55, r"$N_8(i)$", fontsize=11, color=GRAY, ha="center")


def panel_relu(ax):
    """Half-plane crop: 8-nbr edges exist only between similar patches."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"(b)  Keep similar neighbors", fontsize=13, color=INK, loc="left", pad=8)
    s = 7
    obj = np.zeros((s, s), dtype=np.float64)
    obj[:, 3:] = 1.0
    z = np.zeros((s, s, 2), dtype=np.float64)
    z[..., 0] = 1.0 - obj
    z[..., 1] = obj
    ax.set_xlim(-0.7, s - 0.3)
    ax.set_ylim(s - 0.3, -0.7)
    draw_grid(ax, s)
    ax.imshow(
        obj,
        cmap="Blues",
        vmin=0,
        vmax=1,
        origin="upper",
        extent=[-0.5, s - 0.5, s - 0.5, -0.5],
        interpolation="nearest",
        alpha=0.50,
        zorder=1,
    )
    segs, _, _ = n8_edges(np.ones((s, s)), z, thr=0.15)
    if segs:
        ax.add_collection(
            LineCollection(
                segs,
                colors=[(0.12, 0.22, 0.37, 0.92)] * len(segs),
                linewidths=[2.15] * len(segs),
                zorder=3,
            )
        )
    ys, xs = np.mgrid[0:s, 0:s]
    ax.scatter(xs, ys, s=26, c=NAVY, zorder=4, linewidths=0)
    ax.plot([2.5, 2.5], [-0.5, s - 0.5], color="#A33B20", lw=1.4, ls=(0, (3.2, 2.0)), zorder=5)
    ax.text(
        s / 2 - 0.5,
        s + 0.12,
        r"ReLU$(z_i^{\top} z_j)$ on $N_8$  ·  edges stop at appearance",
        fontsize=10,
        color=GRAY,
        ha="center",
        va="bottom",
    )


def panel_times_a(ax, a, z):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"(c)  Weight by this slot", fontsize=13, color=INK, loc="left", pad=8)
    y0, x0, s = 3, 3, 10
    ca, cz = a[y0 : y0 + s, x0 : x0 + s], z[y0 : y0 + s, x0 : x0 + s]
    ax.set_xlim(-0.7, s - 0.3)
    ax.set_ylim(s - 0.3, -0.7)
    draw_grid(ax, s)
    ax.imshow(
        ca,
        cmap="Blues",
        vmin=0,
        vmax=1,
        origin="upper",
        extent=[-0.5, s - 0.5, s - 0.5, -0.5],
        interpolation="nearest",
        alpha=0.9,
        zorder=1,
    )
    segs, widths, alphas = n8_edges(ca, cz, thr=0.02)
    if segs:
        lc = LineCollection(
            segs,
            colors=[(0.12, 0.22, 0.37, al) for al in alphas],
            linewidths=widths,
            zorder=3,
        )
        ax.add_collection(lc)
    ys, xs = np.mgrid[0:s, 0:s]
    sizes = 8 + 48 * np.clip(ca, 0, 1)
    ax.scatter(
        xs.ravel(),
        ys.ravel(),
        s=sizes.ravel(),
        c="#1F3A5F",
        alpha=0.3 + 0.6 * np.clip(ca, 0, 1).ravel(),
        linewidths=0,
        zorder=4,
    )
    ax.text(
        s / 2 - 0.5,
        s + 0.05,
        r"$G_s=\mathrm{diag}(a_s)\,S\,\mathrm{diag}(a_s)$",
        fontsize=10,
        color=GRAY,
        ha="center",
        va="bottom",
    )


def main():
    style()
    large_a, small_a, ghost_a, z, z_small, z_g = make_fields(16)

    fig = plt.figure(figsize=(16.0, 9.0), dpi=180)
    fig.suptitle(
        "Each slot induces an 8-neighbor patch graph",
        fontsize=18,
        color=INK,
        x=0.035,
        ha="left",
        y=0.975,
    )
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.02, 1.18],
        hspace=0.34,
        wspace=0.18,
        left=0.04,
        right=0.98,
        top=0.88,
        bottom=0.10,
    )

    panel_n8(fig.add_subplot(gs[0, 0]))
    panel_relu(fig.add_subplot(gs[0, 1]))
    panel_times_a(fig.add_subplot(gs[0, 2]), large_a, z)

    draw_graph(
        fig.add_subplot(gs[1, 0]),
        large_a,
        z,
        "(d)  Large object",
        "big owned support  ·  large graph",
    )
    draw_graph(
        fig.add_subplot(gs[1, 1]),
        small_a,
        z_small,
        "(e)  Small object",
        "tiny owned support  ·  small graph",
    )
    draw_graph(
        fig.add_subplot(gs[1, 2]),
        ghost_a,
        z_g,
        "(f)  Ghost",
        "no ownership  ·  faint graph, no scale",
        cmap_vmax=1.0,
        node_scale=18.0,
    )

    fig.text(
        0.035,
        0.035,
        r"A slot that owns little mass induces a small graph.  That scale is the occupancy signal.",
        fontsize=13,
        color=INK,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "07_slot_graph_n8.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
