"""Slide 3: one picture above and one below each block.

Left column  — reconstruction: gated mix vs softmax mix.
Right column — predictor SA: live key vs silenced key.

Uses the cached YTVIS frame. Numpy only.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path("/workspace/SlotCurri") if Path("/workspace/SlotCurri/slotcurri").exists() else Path(
    "/mnt/ssd2/hmlee/SlotCurri"
)
OUT = ROOT / "event_analysis/slide_featcur"
CACHE = OUT / "slot_graph_example.npz"

NAVY = "#1F3A5F"
RUST = "#C45C26"
INK = "#111111"
GRAY = "#6E6E6E"
MUTE = "#B33A3A"
N8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

# Distinct slot paints. Slot 0 (child) is rust.
PALETTE = np.array(
    [
        [0.77, 0.36, 0.15],
        [0.18, 0.47, 0.72],
        [0.18, 0.62, 0.48],
        [0.80, 0.58, 0.16],
        [0.52, 0.33, 0.68],
        [0.72, 0.32, 0.48],
        [0.45, 0.45, 0.48],
    ]
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


def load_example() -> dict:
    d = np.load(CACHE)
    return {"rgb": d["rgb"], "a": d["a"], "z": d["z"], "slot": int(d["slot"])}


def n8_undirected(grid: int):
    pairs = []
    for y in range(grid):
        for x in range(grid):
            i = y * grid + x
            for dy, dx in N8:
                y2, x2 = y + dy, x + dx
                if 0 <= y2 < grid and 0 <= x2 < grid:
                    j = y2 * grid + x2
                    if j > i:
                        pairs.append((i, j))
    return pairs


def build_S(z: np.ndarray, grid: int):
    n = grid * grid
    deg = np.zeros(n, dtype=np.float64)
    raw = []
    for i, j in n8_undirected(grid):
        r = max(0.0, float(np.dot(z[i], z[j])))
        deg[i] += r
        deg[j] += r
        raw.append((i, j, r))
    dinv = 1.0 / np.sqrt(np.clip(deg, 1e-8, None))
    S = np.zeros((n, n), dtype=np.float64)
    for i, j, r in raw:
        s = r * dinv[i] * dinv[j]
        S[i, j] = S[j, i] = s
    return S


def perron(S: np.ndarray, a: np.ndarray, n_iter: int = 40, eps: float = 1e-8):
    q = np.ones(a.shape[0], dtype=np.float64)
    q /= np.linalg.norm(q)
    for _ in range(n_iter):
        gq = a * (S @ (a * q))
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
    lam = float(q @ (a * (S @ (a * q))))
    return q, max(lam, 0.0)


def upsample(field_n: np.ndarray, grid: int, h: int, w: int) -> np.ndarray:
    cell_h, cell_w = h // grid, w // grid
    img = np.repeat(np.repeat(field_n.reshape(grid, grid), cell_h, 0), cell_w, 1)
    return img[:h, :w]


def paint_slots(rgb, a, grid, ids, peak_norm=False):
    h, w, _ = rgb.shape
    ids = list(ids)
    mix = np.zeros_like(rgb)
    acc = np.zeros((h, w), dtype=np.float64)
    for s in ids:
        m = a[s].copy()
        if peak_norm:
            m = m / max(float(m.max()), 1e-8)
        img = upsample(m, grid, h, w)
        mix += img[..., None] * PALETTE[s]
        acc += img
    acc = np.clip(acc, 1e-8, None)
    mix = mix / acc[..., None]
    vis = np.clip(acc / np.percentile(acc, 99), 0.0, 1.0)
    out = rgb * 0.42 * (1.0 - 0.75 * vis[..., None]) + mix * (0.35 + 0.65 * vis[..., None])
    return np.clip(out, 0.0, 1.0)


def add_ghost_specks(rgb, a, grid, ghosts, thresh=0.10):
    h, w, _ = rgb.shape
    out = rgb.copy()
    for s in ghosts:
        m = upsample(a[s] / max(float(a[s].max()), 1e-8), grid, h, w)
        spec = m > thresh
        if not spec.any():
            continue
        col = PALETTE[s]
        out[spec] = 0.15 * out[spec] + 0.85 * col
    return np.clip(out, 0.0, 1.0)


def peak_xy(a_s: np.ndarray, grid: int, h: int, w: int):
    i = int(np.argmax(a_s))
    x, y = i % grid, i // grid
    return (x + 0.5) * w / grid, (y + 0.5) * h / grid


def hide(ax) -> None:
    ax.set_axis_off()


def caption(ax, text: str) -> None:
    ax.text(0.0, -0.06, text, transform=ax.transAxes, fontsize=10, color=GRAY, ha="left", va="top")


def panel_mix(ax, img, title, cap, ghost_marks=None, grid=None, a=None):
    ax.imshow(img, origin="upper")
    h, w = img.shape[:2]
    if ghost_marks and a is not None and grid is not None:
        xs, ys = [], []
        for s in ghost_marks:
            x, y = peak_xy(a[s], grid, h, w)
            xs.append(x)
            ys.append(y)
            ax.add_patch(
                Circle((x, y), 0.032 * min(h, w), fill=False, edgecolor=MUTE, lw=2.0, zorder=8)
            )
        ax.text(
            min(xs) - 0.02 * w,
            float(np.mean(ys)),
            "ghosts",
            color=MUTE,
            fontsize=10,
            ha="right",
            va="center",
            zorder=8,
            fontweight="bold",
        )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    hide(ax)
    ax.set_title(title, fontsize=13, color=INK, loc="left", pad=8)
    caption(ax, cap)


def overlay_glow(ax, rgb, field, grid, cmap, alpha=0.55):
    h, w = rgb.shape[:2]
    ax.imshow(np.clip(rgb * 0.50, 0, 1), origin="upper")
    vis = field.reshape(grid, grid)
    vis = vis / max(float(vis.max()), 1e-8)
    ax.imshow(
        vis,
        cmap=cmap,
        origin="upper",
        extent=[0, w, h, 0],
        alpha=alpha,
        interpolation="bilinear",
        vmin=0.0,
        vmax=1.0,
        zorder=2,
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)


def fat_arrow(ax, p0, p1, color, lw=2.6, rad=0.08):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=18,
            lw=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=5,
            shrinkA=16,
            shrinkB=18,
        )
    )


def token(ax, xy, radius, face, edge, label="", lw=2.0):
    ax.add_patch(
        Circle(xy, radius, facecolor=face, edgecolor=edge, lw=lw, zorder=6, alpha=0.96)
    )
    if label:
        ax.text(
            xy[0],
            xy[1],
            label,
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            zorder=7,
            fontweight="bold",
        )


def badge(ax, xy, text, edge, w_frac, h_frac, h, w):
    ax.add_patch(
        FancyBboxPatch(
            (xy[0] - 0.5 * w_frac * w, xy[1] - 0.55 * h_frac * h),
            w_frac * w,
            h_frac * h,
            boxstyle="round,pad=0.02",
            facecolor="white",
            edgecolor=edge,
            lw=1.5,
            zorder=8,
        )
    )
    ax.text(xy[0], xy[1], text, color=edge, fontsize=10, ha="center", va="center", zorder=9, fontweight="bold")


def panel_live(ax, rgb, a, grid, live, queries):
    h, w = rgb.shape[:2]
    overlay_glow(ax, rgb, a[live], grid, "Oranges", alpha=0.60)
    r_key = 0.055 * min(h, w)
    r_q = 0.038 * min(h, w)
    kxy = peak_xy(a[live], grid, h, w)
    qxy = [peak_xy(a[s], grid, h, w) for s in queries]
    # broadcast rings
    for k in (1, 2, 3):
        ax.add_patch(
            Circle(
                kxy,
                r_key * (1.35 + 0.55 * k),
                fill=False,
                edgecolor=RUST,
                lw=1.3,
                alpha=0.35,
                linestyle="--",
                zorder=3,
            )
        )
    token(ax, kxy, r_key, PALETTE[live], "white", "", lw=2.4)
    for i, (xy, s) in enumerate(zip(qxy, queries)):
        token(ax, xy, r_q, PALETTE[s], "white", "")
        fat_arrow(ax, kxy, xy, RUST, rad=0.12 if xy[0] > kxy[0] else -0.12)
    ax.text(
        qxy[0][0],
        min(h - 12, qxy[0][1] + 0.08 * h),
        "other slots",
        color="white",
        fontsize=9,
        ha="center",
        zorder=8,
        bbox=dict(facecolor="black", alpha=0.35, boxstyle="round,pad=0.25", edgecolor="none"),
    )
    badge(ax, (kxy[0], min(h - 24, kxy[1] + 0.14 * h)), r"ON   $\pi \approx 1$", RUST, 0.28, 0.09, h, w)
    hide(ax)
    ax.set_title(r"(c)  Live key   $\pi_s \approx 1$", fontsize=13, color=INK, loc="left", pad=8)
    caption(ax, "This slot writes into the others.  They may attend to it.")


def panel_silent(ax, rgb, a, grid, silent, queries):
    h, w = rgb.shape[:2]
    overlay_glow(ax, rgb, a[silent], grid, "Greys", alpha=0.28)
    r_key = 0.055 * min(h, w)
    r_q = 0.038 * min(h, w)
    kxy = peak_xy(a[silent], grid, h, w)
    qxy = [peak_xy(a[s], grid, h, w) for s in queries]
    token(ax, kxy, r_key, np.array([0.55, 0.55, 0.57]), MUTE, lw=2.2)
    mark = 0.030 * min(h, w)
    ax.plot([kxy[0] - mark, kxy[0] + mark], [kxy[1] - mark, kxy[1] + mark], color=MUTE, lw=3.0, zorder=8)
    ax.plot([kxy[0] - mark, kxy[0] + mark], [kxy[1] + mark, kxy[1] - mark], color=MUTE, lw=3.0, zorder=8)
    for xy, s in zip(qxy, queries):
        token(ax, xy, r_q, PALETTE[s], "white")
    ax.text(
        qxy[0][0],
        min(h - 12, qxy[0][1] + 0.08 * h),
        "other slots",
        color="white",
        fontsize=9,
        ha="center",
        zorder=8,
        bbox=dict(facecolor="black", alpha=0.35, boxstyle="round,pad=0.25", edgecolor="none"),
    )
    bx = float(np.clip(kxy[0] + 0.20 * w, 0.22 * w, 0.72 * w))
    by = float(np.clip(kxy[1] + 0.12 * h, 0.12 * h, 0.85 * h))
    badge(ax, (bx, by), r"OFF   $\pi \approx 0$", MUTE, 0.30, 0.09, h, w)
    hide(ax)
    ax.set_title(r"(d)  Silenced key   $\pi_s \approx 0$", fontsize=13, color=INK, loc="left", pad=8)
    caption(ax, "Muted as a key.  It does not write into other slots.")


def main():
    style()
    if not CACHE.is_file():
        raise SystemExit(f"missing cache {CACHE}")
    ex = load_example()
    rgb = ex["rgb"]
    a = ex["a"].astype(np.float64)
    z = ex["z"].astype(np.float64)
    grid = int(round(z.shape[0] ** 0.5))
    print("building S", grid, flush=True)
    S = build_S(z, grid)
    pis = []
    for s in range(a.shape[0]):
        _, pi = perron(S, a[s])
        pis.append(pi)
        print(f"  slot {s}  pi={pi:.3f}  mass={a[s].mean():.4f}", flush=True)
    pis = np.array(pis)
    live = int(ex["slot"])
    silent = int(np.argmin(pis))
    if silent == live:
        silent = int(np.argsort(pis)[0 if live != 0 else 1])
    live_ids = [s for s in range(a.shape[0]) if pis[s] >= 0.80]
    ghost_ids = [s for s in range(a.shape[0]) if s not in live_ids]
    print("live key", live, "silent key", silent, "live mix", live_ids, "ghosts", ghost_ids, flush=True)

    gated = paint_slots(rgb, a, grid, live_ids, peak_norm=False)
    softmax = add_ghost_specks(paint_slots(rgb, a, grid, range(a.shape[0]), peak_norm=False), a, grid, ghost_ids)

    fig = plt.figure(figsize=(16.0, 9.2), dpi=180)
    fig.text(0.04, 0.96, "Training objective  &  gate-aware predictor", fontsize=18, color=INK, ha="left", va="top")

    gs = fig.add_gridspec(
        2,
        2,
        wspace=0.10,
        hspace=0.30,
        left=0.04,
        right=0.99,
        top=0.90,
        bottom=0.10,
    )
    panel_mix(
        fig.add_subplot(gs[0, 0]),
        gated,
        r"(a)  Main   $\hat{X}_{\pi}$  occupancy-gated",
        r"Low-$\pi$ slots drop out of the mix.   $||X-\hat{X}_{\pi}||^{2}$",
    )
    panel_mix(
        fig.add_subplot(gs[1, 0]),
        softmax,
        r"(b)  Auxiliary   $\hat{X}_{\mathrm{sm}}$  plain softmax",
        r"Candidate / ghost slots still paint.   $||X-\hat{X}_{\mathrm{sm}}||^{2}$",
        ghost_marks=ghost_ids,
        grid=grid,
        a=a,
    )
    h, w = rgb.shape[:2]
    kxy = np.array(peak_xy(a[live], grid, h, w))
    cands = [s for s in range(a.shape[0]) if s != live and s != silent]
    far = sorted(cands, key=lambda s: -np.hypot(*(np.array(peak_xy(a[s], grid, h, w)) - kxy)))
    queries = far[:2]
    print("queries", queries, flush=True)
    panel_live(fig.add_subplot(gs[0, 1]), rgb, a, grid, live, queries)
    panel_silent(fig.add_subplot(gs[1, 1]), rgb, a, grid, silent, queries)

    fig.text(
        0.04,
        0.035,
        r"$\pi$ mixes the decoder.  As a key, $\pi$ also decides who the predictor may read.",
        fontsize=12,
        color=INK,
    )
    path = OUT / "09_objective_gate.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", path, flush=True)

    fig2 = plt.figure(figsize=(16.0, 7.2), dpi=180)
    fig2.text(
        0.035,
        0.955,
        "Source-gated slot self-attention",
        fontsize=18,
        color=INK,
        ha="left",
        va="top",
    )
    gs2 = fig2.add_gridspec(1, 2, wspace=0.08, left=0.03, right=0.99, top=0.86, bottom=0.14)
    panel_live(fig2.add_subplot(gs2[0, 0]), rgb, a, grid, live, queries)
    panel_silent(fig2.add_subplot(gs2[0, 1]), rgb, a, grid, silent, queries)
    fig2.text(
        0.035,
        0.04,
        r"A live key writes into other slots.  A silenced key does not.",
        fontsize=12,
        color=INK,
    )
    path2 = OUT / "09_src_gate_keys.png"
    fig2.savefig(path2, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig2)
    print("wrote", path2, flush=True)


if __name__ == "__main__":
    main()
