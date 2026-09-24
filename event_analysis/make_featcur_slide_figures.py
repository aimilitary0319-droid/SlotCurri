"""Presentation figures for the feature curriculum (task-level coarse-to-fine).

Existing slide_*.png files cover the *gate* (p, beta, purity), not this module.
DINO PCA dumps in featcur_mix_sweep / ncut_vis_movi are analysis grids, not slides.

Writes 16:9 PNGs to event_analysis/slide_featcur/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, RegularPolygon, Wedge
from matplotlib.ticker import FuncFormatter, MultipleLocator
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "event_analysis" / "slide_featcur"

NAVY = "#1F3A5F"
TEAL = "#3C6E71"
RUST = "#A33B20"
GRAY = "#6E6E6E"
INK = "#111111"
RULE = "#D0D4D8"
FILL = "#F4F6F8"
A_COL = np.array([0.18, 0.38, 0.72])
B_COL = np.array([0.78, 0.42, 0.16])
BG_COL = np.array([0.88, 0.89, 0.90])


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.linewidth": 0.8,
            "lines.solid_capstyle": "round",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def cosine_s(step, anneal: float):
    t = np.clip(np.asarray(step, dtype=float) / float(anneal), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * t))


def kfmt(x, _):
    if abs(x) < 1e-9:
        return "0"
    return f"{int(round(x / 1000))}"


def apply_axes(ax, tmax=100_000):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ax.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.8)
    ax.tick_params(direction="in", length=3.2, width=0.8, colors=INK, labelsize=9, pad=3)
    ax.set_xlim(0, tmax)
    ax.set_xticks([0, 25_000, 50_000, 75_000, 100_000])
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.xaxis.set_minor_locator(MultipleLocator(12_500))
    ax.tick_params(which="minor", direction="in", length=1.6, width=0.6)


def normalize(x, eps=1e-8):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, eps, None)


def fit_pca(x):
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:3].T
    rgb = xc @ basis
    return mean, basis, rgb.min(axis=0), rgb.max(axis=0)


def project_pca(x, mean, basis, lo, hi):
    rgb = (x - mean) @ basis
    return np.clip((rgb - lo) / np.clip(hi - lo, 1e-6, None), 0.0, 1.0)


def unique_keys(x, thr=0.95):
    xn = normalize(x)
    kept = []
    for row in xn:
        if not kept:
            kept.append(row)
            continue
        if float(np.max(np.stack(kept) @ row)) < thr:
            kept.append(row)
    return len(kept)


def ncut_level(x, eps=1e-6):
    z = normalize(x)
    w = np.maximum(z @ z.T, 0.0)
    np.fill_diagonal(w, 0.0)
    row = w.sum(axis=1, keepdims=True)
    rel = np.where(row < eps, x, (w @ x) / np.clip(row, eps, None))
    return rel, w / np.clip(row, eps, None)


def affinity_smooth(x, tau=0.1, window=None, side=None):
    z = normalize(x)
    logits = (z @ z.T) / float(tau)
    n = x.shape[0]
    if window is not None and side is not None:
        idx = np.arange(n)
        ys, xs = np.divmod(idx, side)
        cheb = np.maximum(np.abs(ys[:, None] - ys[None, :]), np.abs(xs[:, None] - xs[None, :]))
        logits = np.where(cheb > int(window), -1e9, logits)
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-8, None)
    return p @ x


def modes_cover(x, h, n_iter=6, delta=0.03, h_min=0.02):
    if h is None or h <= h_min:
        return x
    z = normalize(x)
    m = z.copy()
    inv_h = 1.0 / max(float(h), 1e-6)
    for _ in range(n_iter):
        w = np.exp((m @ m.T - 1.0) * inv_h)
        m = normalize(w @ m)
    n = m.shape[0]
    assigned = np.full(n, -1, dtype=int)
    centers = []
    for i in range(n):
        hit = None
        for k, c in enumerate(centers):
            if float(m[i] @ c) > 1.0 - delta:
                hit = k
                break
        if hit is None:
            hit = len(centers)
            centers.append(m[i])
        assigned[i] = hit
    means = np.zeros_like(x)
    for k in range(len(centers)):
        sel = assigned == k
        means[sel] = x[sel].mean(axis=0)
    return means


def make_scene(g=24):
    yy, xx = np.mgrid[0:g, 0:g]
    labels = np.zeros((g, g), dtype=int)
    body_a = ((xx - 7.2) / 3.1) ** 2 + ((yy - 14.5) / 6.6) ** 2 < 1.0
    head_a = ((xx - 7.2) / 2.3) ** 2 + ((yy - 6.4) / 2.5) ** 2 < 1.0
    body_b = ((xx - 17.2) / 3.6) ** 2 + ((yy - 13.8) / 5.4) ** 2 < 1.0
    head_b = ((xx - 17.2) / 2.15) ** 2 + ((yy - 7.0) / 2.3) ** 2 < 1.0
    labels[body_a] = 1
    labels[head_a] = 2
    labels[body_b] = 3
    labels[head_b] = 4
    # Within-object cosine is high so one step of P X collapses parts;
    # objects are nearly orthogonal so ReLU-cosine does not mix them.
    proto = np.array(
        [
            [1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 1.00, 0.08, 0.00, 0.00, 0.00],
            [0.00, 1.00, 0.28, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 1.00, 0.08, 0.00],
            [0.00, 0.00, 0.00, 1.00, 0.28, 0.00],
        ],
        dtype=np.float64,
    )
    x = proto[labels.reshape(-1)]
    rng = np.random.default_rng(1)
    x = x + 0.015 * rng.normal(size=x.shape)
    rgb_map = np.array(
        [
            BG_COL,
            A_COL * 0.92 + 0.08,
            A_COL * 0.55 + np.array([0.05, 0.12, 0.35]),
            B_COL * 0.92 + 0.08,
            B_COL * 0.55 + np.array([0.35, 0.18, 0.05]),
        ]
    )
    return labels, x, rgb_map[labels]


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def fig_idea_and_clock():
    """Slide 1: why coarsen tokens, and the shared cosine clock."""
    fig = plt.figure(figsize=(16.0, 9.0), dpi=180)
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.15, 1.0],
        hspace=0.38,
        wspace=0.28,
        left=0.06,
        right=0.97,
        top=0.90,
        bottom=0.10,
    )
    fig.suptitle(
        "Feature curriculum  —  coarsen the task, not the gate",
        fontsize=16,
        color=INK,
        x=0.06,
        ha="left",
        y=0.97,
    )

    labels, x, rgb = make_scene(24)
    rel, _ = ncut_level(x)
    mean, basis, lo, hi = fit_pca(x)
    rgb_raw = project_pca(x, mean, basis, lo, hi)
    rgb_rel = project_pca(rel, mean, basis, lo, hi)
    g = 24
    panels = [
        (rgb.reshape(g, g, 3), r"(a)  Scene", "two objects, each with a part"),
        (rgb_rel.reshape(g, g, 3), r"(b)  Early  $s=0$", r"$X^{\mathrm{rel}}=PX$  ·  parts collapse"),
        (rgb_raw.reshape(g, g, 3), r"(c)  Late / eval  $s=1$", "raw DINO  ·  parts return"),
    ]
    for i, (img, title, note) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(np.clip(img, 0, 1), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(0.9)
        ax.set_title(title, fontsize=12, pad=8, loc="left", color=INK)
        ax.set_xlabel(note, fontsize=10, color=GRAY, labelpad=6)

    ax = fig.add_subplot(gs[1, :2])
    t = np.linspace(0, 100_000, 800)
    s30 = np.where(t <= 30_000, cosine_s(t, 30_000), 1.0)
    s50 = np.where(t <= 50_000, cosine_s(t, 50_000), 1.0)
    ax.plot(t, s50, color=NAVY, lw=2.0, label=r"$s(t)$  ncut  (v36+, 50k)")
    ax.plot(t, s30, color=TEAL, lw=1.8, ls=(0, (3.4, 2.0)), label=r"$s(t)$  mix / window / modes  (v33–v35, 30k)")
    ax.axvline(50_000, color=RULE, lw=0.8)
    ax.axvline(30_000, color=RULE, lw=0.8)
    ax.set_ylim(-0.04, 1.08)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel(r"Training step ($\times 10^3$)", labelpad=4)
    ax.set_ylabel(r"Raw-token weight $s$", labelpad=4)
    apply_axes(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=10, handlelength=2.2)
    ax.set_title(
        r"(d)  Shared cosine clock   $s=\frac{1}{2}(1-\cos\pi t),\ t=\mathrm{clip}(\mathrm{step}/T,0,1)$",
        fontsize=12,
        loc="left",
        pad=6,
    )

    ax = fig.add_subplot(gs[1, 2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(e)  What $s$ mixes", fontsize=12, loc="left", pad=6)
    box = dict(boxstyle="round,pad=0.45", facecolor=FILL, edgecolor=RULE, linewidth=0.9)
    ax.text(
        0.0,
        0.92,
        r"$X^{\mathrm{bind}}=(1-s)\,X^{\mathrm{coarse}}+s\,X$",
        fontsize=12,
        color=INK,
        va="top",
        bbox=box,
    )
    ax.text(0.0, 0.58, r"$s=0$  object-level Keys" "\n" r"part-split earns no recon" "\n" r"and cannot hold exclusive $q{\cdot}k$", fontsize=10.5, color=INK, va="top")
    ax.text(0.0, 0.22, r"$s=1$  and eval: identity" "\n" r"encoder is byte-identical" "\n" r"to the uncurriculumed model", fontsize=10.5, color=INK, va="top")
    return save(fig, "01_idea_and_clock.png")


def _card(ax, title, clock, formula, body, current=False):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    edge = RUST if current else RULE
    lw = 1.6 if current else 0.9
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.04),
            0.96,
            0.92,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor="white",
            edgecolor=edge,
            linewidth=lw,
        )
    )
    ax.text(0.08, 0.88, title, fontsize=13, color=INK, va="center", fontweight="bold")
    if current:
        ax.text(0.92, 0.88, "current", fontsize=9, color=RUST, ha="right", va="center")
    ax.text(0.08, 0.68, formula, fontsize=12, color=INK, va="center")
    ax.text(0.08, 0.48, clock, fontsize=10, color=TEAL, va="center")
    ax.text(0.08, 0.22, body, fontsize=9.5, color=GRAY, va="center", linespacing=1.45)


def fig_four_anneals():
    """Slide 2: the four anneal knobs, formulas, and a computed toy row."""
    fig = plt.figure(figsize=(16.0, 9.0), dpi=180)
    fig.suptitle(
        "Four feature-curriculum clocks  —  same $s(t)$, different coarsener",
        fontsize=16,
        color=INK,
        x=0.04,
        ha="left",
        y=0.97,
    )
    gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[1.15, 1.35, 0.08],
        hspace=0.22,
        wspace=0.16,
        left=0.04,
        right=0.98,
        top=0.90,
        bottom=0.06,
    )

    cards = [
        (
            "v33  mix",
            r"clock: blend $s$  ·  $T=30$k  ·  apply=tokens",
            r"$X_{\mathrm{used}}=(1-s)\tilde{X}+s X$",
            r"$\tilde{X}=$ softmax affinity of $X$  ($\tau{=}0.1$, $w{=}9$)"
            "\nfixed neighbourhood; only the mix with raw changes",
            False,
        ),
        (
            "v34  window",
            r"clock: radius $w(t)=\mathrm{round}(w_0(1-s))$",
            r"$X\leftarrow \mathrm{smooth}(X;\,w)$   (no blend)",
            r"$w_0{=}13$, 3 hops.  $w{=}0$ is raw."
            "\nwide early neighbourhood → shrink to local parts",
            False,
        ),
        (
            "v35  modes",
            r"clock: bandwidth $h(t)=h_0(1-s)$",
            r"$X\leftarrow \mathrm{mode}$-$\mathrm{cover}(X;\,h)$",
            r"$h_0{=}0.5$.  $h \leq h_{\min}$ is raw."
            "\n"
            "n_keys = C(scene, h), not leftover seats",
            False,
        ),
        (
            "v36+  ncut",
            r"clock: blend $s$  ·  $T=50$k  ·  apply=key",
            r"$X^{\mathrm{bind}}=(1-s)PX+s X$",
            r"$P=$ row-normalize ReLU-cosine of $X$"
            "\nKeys only. Value / recon stay on raw $X$.  v37+: no Fiedler cut",
            True,
        ),
    ]
    for i, args in enumerate(cards):
        _card(fig.add_subplot(gs[0, i]), *args)

    labels, x, _ = make_scene(24)
    g = 24
    rel, p = ncut_level(x)
    mean, basis, lo, hi = fit_pca(x)
    ss = [0.0, 0.5, 1.0]
    row_fns = [
        (
            "mix",
            lambda s: (1 - s) * affinity_smooth(x, tau=0.1, window=9, side=g) + s * x,
        ),
        (
            "window",
            lambda s: affinity_smooth(
                x,
                tau=0.1,
                window=max(0, int(np.floor(8 * (1 - s) + 0.5))),
                side=g,
            )
            if s < 1.0 and int(np.floor(8 * (1 - s) + 0.5)) > 0
            else x,
        ),
        (
            "modes",
            lambda s: modes_cover(x, 0.18 * (1 - s)) if s < 1.0 else x,
        ),
        (
            "ncut",
            lambda s: (1 - s) * rel + s * x,
        ),
    ]
    for c, (name, fn) in enumerate(row_fns):
        ax = fig.add_subplot(gs[1, c])
        imgs = []
        keys = []
        for s in ss:
            xb = fn(s)
            rgb = project_pca(xb, mean, basis, lo, hi)
            imgs.append(rgb.reshape(g, g, 3))
            keys.append(unique_keys(xb))
        strip = np.concatenate(imgs, axis=1)
        ax.imshow(np.clip(strip, 0, 1), interpolation="nearest")
        ax.set_xticks([g * 0.5, g * 1.5, g * 2.5])
        ax.set_xticklabels(
            [rf"$s={s:g}$" + f"\n{k} keys" for s, k in zip(ss, keys)],
            fontsize=9,
        )
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(0.8)
        ax.set_title(name, fontsize=11, loc="left", color=INK if name != "ncut" else RUST)

    fig.text(
        0.04,
        0.045,
        "Toy 24×24 tokens, PCA fit on raw.  Same scene, four coarseners.  "
        "“keys” = # distinct directions at cos < 0.95.  "
        "v39lam1gu uses ncut, apply=key, barrier=false, T=50k.",
        fontsize=9.5,
        color=GRAY,
    )
    return save(fig, "02_four_anneals.png")


def fig_ncut_pipeline():
    """Slide 3: current (v37/v39) ncut + apply=key pipeline."""
    fig, ax = plt.figure(figsize=(16.0, 9.0), dpi=180), None
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.86])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.suptitle(
        r"Current recipe  (v37–v39)   Key-only Ncut mix,  $\mathrm{barrier{=}false}$",
        fontsize=16,
        color=INK,
        x=0.04,
        ha="left",
        y=0.97,
    )

    def box(x, y, w, h, text, fill="white", edge=NAVY, fs=11):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=fill,
                edgecolor=edge,
                linewidth=1.1,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                lw=1.1,
                color=INK,
            )
        )

    box(0.4, 6.7, 3.2, 1.5, "frozen DINO\n" r"$X \in R^{N \times d}$", FILL, NAVY)
    box(4.3, 6.7, 3.6, 1.5, r"$Z=X/||X||$" "\n" r"$W=\mathrm{ReLU}(ZZ^\top),\ W_{ii}{=}0$", "white", NAVY)
    box(8.6, 6.7, 3.4, 1.5, r"$P=\mathrm{rownorm}(W)$" "\n" r"$X^{\mathrm{rel}}=PX$", "white", NAVY)
    box(12.6, 6.7, 3.1, 1.5, r"$X^{\mathrm{bind}}$" "\n" r"$=(1-s)X^{\mathrm{rel}}+sX$", FILL, RUST)

    arrow(3.6, 7.45, 4.3, 7.45)
    arrow(7.9, 7.45, 8.6, 7.45)
    arrow(12.0, 7.45, 12.6, 7.45)

    box(0.4, 3.4, 4.4, 1.7, "Value  /  recon target\n" r"raw $X$  (not mixed)", "white", TEAL, 12)
    box(5.8, 3.4, 4.6, 1.7, "Keys\n" r"$W_K\,\mathrm{LN}(\varphi(X^{\mathrm{bind}}))$", FILL, RUST, 12)
    box(11.1, 3.4, 4.6, 1.7, "Gate graph $G_s$\n" r"$Z$ from $X^{\mathrm{bind}}$", "white", NAVY, 12)

    arrow(2.0, 6.7, 2.6, 5.1)
    arrow(14.15, 6.7, 8.1, 5.1)
    arrow(14.15, 6.7, 13.4, 5.1)

    ax.text(2.6, 5.55, "raw $X$", fontsize=9, color=TEAL, ha="center")
    ax.text(10.4, 5.55, r"$X^{\mathrm{bind}}$", fontsize=9, color=RUST, ha="center")

    ax.text(8.0, 2.55, r"apply = key", fontsize=13, color=RUST, ha="center")
    ax.text(
        8.0,
        1.95,
        r"v33–v35 used apply=tokens: Keys, Values, and $L_{\mathrm{featrec}}$ all saw the coarsened tensor.",
        fontsize=10.5,
        color=GRAY,
        ha="center",
    )
    ax.text(
        8.0,
        1.15,
        r"Train only.  Validation / eval skip the mix ($s{=}1$).  "
        r"After 50k the encoder is byte-identical to no-curriculum Keys.",
        fontsize=10.5,
        color=GRAY,
        ha="center",
    )
    ax.text(
        8.0,
        0.35,
        r"v36: 2-way Fiedler/median barrier on $W$  ·  v37+: barrier off, global $P$  ·  "
        r"v39 occupancy $\pi$ is a separate module on 8-nbr $G_s$.",
        fontsize=10.5,
        color=GRAY,
        ha="center",
    )
    return save(fig, "03_ncut_pipeline.png")


def fig_toy_leveling():
    """Slide 4: computed ncut mix + mixer kernel K=(1-s)P + s I."""
    fig = plt.figure(figsize=(16.0, 9.0), dpi=180)
    fig.suptitle(
        r"Ncut mix on a toy scene    $X^{\mathrm{bind}}=(1-s)PX+sX$    $K=(1-s)P+sI$",
        fontsize=15,
        color=INK,
        x=0.04,
        ha="left",
        y=0.97,
    )
    gs = fig.add_gridspec(
        3,
        6,
        height_ratios=[1.05, 1.05, 0.95],
        hspace=0.32,
        wspace=0.12,
        left=0.05,
        right=0.98,
        top=0.88,
        bottom=0.08,
    )
    labels, x, rgb = make_scene(24)
    g = 24
    rel, p = ncut_level(x)
    mean, basis, lo, hi = fit_pca(x)
    steps = [0.0, 0.25, 0.5, 0.75, 1.0]
    q = int(np.argmax((labels.reshape(-1) == 2)))  # a head-A patch

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
    qy, qx = divmod(q, g)
    ax.plot(qx, qy, "+", color="white", ms=10, mew=1.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("scene  (+ query)", fontsize=10, loc="left")
    for sp in ax.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.8)

    for i, s in enumerate(steps):
        xb = (1 - s) * rel + s * x
        rgb_s = project_pca(xb, mean, basis, lo, hi)
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(np.clip(rgb_s.reshape(g, g, 3), 0, 1), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$s={s:g}$  ·  {unique_keys(xb)} keys", fontsize=10)
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(0.8)

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
    ax.plot(qx, qy, "+", color="white", ms=10, mew=1.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(r"mixer $K_{q,:}$", fontsize=11)
    ax.set_title("query patch", fontsize=10, loc="left")
    for sp in ax.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.8)

    i_mat = np.eye(p.shape[0])
    vmax = None
    kmaps = []
    for s in steps:
        k = (1 - s) * p + s * i_mat
        km = k[q].reshape(g, g)
        kmaps.append(km)
        vmax = km.max() if vmax is None else max(vmax, km.max())
    for i, (s, km) in enumerate(zip(steps, kmaps)):
        ax = fig.add_subplot(gs[1, i + 1])
        vis = np.log10(km + 1e-6)
        ax.imshow(vis, cmap="magma", vmin=-4.5, vmax=np.log10(max(vmax, 1e-3)))
        ax.plot(qx, qy, "+", color="cyan", ms=8, mew=1.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$s={s:g}$", fontsize=10)
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(0.8)

    ax = fig.add_subplot(gs[2, :])
    t = np.linspace(0, 100_000, 600)
    s = np.where(t <= 50_000, cosine_s(t, 50_000), 1.0)
    ax.plot(t, s, color=NAVY, lw=1.8, label=r"$s$  (raw)")
    ax.plot(t, 1 - s, color=TEAL, lw=1.8, ls=(0, (3.2, 2.0)), label=r"$1-s$  (leveled $PX$)")
    for mark, lab in [(0, "0"), (25_000, "25k"), (50_000, "50k")]:
        ax.scatter([mark], [float(np.interp(mark, t, s))], color=RUST, s=22, zorder=3)
    apply_axes(ax)
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(r"Training step ($\times 10^3$)", labelpad=3)
    ax.set_ylabel("mix weight", labelpad=3)
    ax.legend(frameon=False, loc="center right", fontsize=10)
    ax.set_title(r"$T=50$k cosine, then $s=1$ through 100k   ·   eval always $s=1$", fontsize=11, loc="left")
    return save(fig, "04_toy_ncut_mix.png")


def fig_apply_key():
    """Slide 5: apply=tokens vs apply=key."""
    fig, ax = plt.subplots(figsize=(16.0, 9.0), dpi=180)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.suptitle(
        r"Where $X^{\mathrm{bind}}$ is consumed    apply=tokens  vs  apply=key",
        fontsize=16,
        color=INK,
        x=0.04,
        ha="left",
        y=0.97,
    )

    def col(x0, title, items, highlight):
        ax.add_patch(
            FancyBboxPatch(
                (x0, 0.7),
                7.2,
                7.4,
                boxstyle="round,pad=0.04,rounding_size=0.1",
                facecolor="white",
                edgecolor=RUST if highlight else RULE,
                linewidth=1.5 if highlight else 0.9,
            )
        )
        ax.text(x0 + 3.6, 7.6, title, ha="center", fontsize=14, color=RUST if highlight else INK)
        y = 6.6
        for name, uses_bind, note in items:
            face = FILL if uses_bind else "white"
            edge = NAVY if uses_bind else TEAL
            ax.add_patch(
                FancyBboxPatch(
                    (x0 + 0.35, y - 0.7),
                    6.5,
                    1.15,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.0,
                )
            )
            tag = r"$X^{\mathrm{bind}}$" if uses_bind else r"raw $X$"
            ax.text(x0 + 0.55, y - 0.12, name, fontsize=12, color=INK, va="center")
            ax.text(x0 + 6.55, y - 0.12, tag, fontsize=11, color=edge, ha="right", va="center")
            ax.text(x0 + 0.55, y - 0.48, note, fontsize=9.5, color=GRAY, va="center")
            y -= 1.45

    col(
        0.5,
        "v33–v35    apply = tokens",
        [
            ("SlotAttention Keys", True, "part-splits see identical keys on the blob"),
            ("SlotAttention Values", True, "updates read the coarsened content"),
            (r"$L_{\mathrm{featrec}}$ target", True, "reconstructing parts is not rewarded"),
            (r"Gate statistic on $Z$", True, "same tensor as Keys"),
        ],
        False,
    )
    col(
        8.3,
        "v36–v39    apply = key   (current)",
        [
            ("SlotAttention Keys", True, "binding still sees the coarsened graph"),
            ("SlotAttention Values", False, "slot content stays high-frequency DINO"),
            (r"$L_{\mathrm{featrec}}$ target", False, "decoder still fits raw features"),
            (r"Gate $Z$ / $G_s$", True, "occupancy measured on the bind tokens"),
        ],
        True,
    )
    ax.text(
        8.0,
        0.28,
        r"Why split: coarsening the reconstruction target made $L_{\mathrm{featrec}}$ too easy and "
        r"hid part structure from the decoder.  Keys-only keeps the anti-split pressure on $q{\cdot}k$ only.",
        fontsize=10.5,
        color=GRAY,
        ha="center",
    )
    return save(fig, "05_apply_tokens_vs_key.png")


def fig_real_crop():
    """Slide 6: reuse the existing real DINO ncut mix grid, cropped and captioned."""
    src = ROOT / "event_analysis" / "ncut_vis_movi" / "ncut_mix_pca.png"
    if not src.is_file():
        print("skip real crop: missing", src)
        return None
    im = Image.open(src).convert("RGB")
    w, h = im.size
    # title + first two clips; drop the empty bottom axes
    crop = im.crop((0, 0, w, int(h * 0.48)))
    fig = plt.figure(figsize=(16.0, 9.0), dpi=180)
    ax = fig.add_axes([0.03, 0.12, 0.94, 0.78])
    ax.imshow(crop)
    ax.axis("off")
    fig.suptitle(
        r"Same mix on real frozen DINO  (MOVi-C)    $s=0$ object-level  $\rightarrow$  $s=1$ raw",
        fontsize=15,
        color=INK,
        x=0.04,
        ha="left",
        y=0.97,
    )
    fig.text(
        0.04,
        0.045,
        "Source: event_analysis/ncut_vis_movi/ncut_mix_pca.png  ·  "
        "PCA of X^bind at s ∈ {0, 0.5, ~0.69, 1}.  "
        "keys = # distinct token directions.  "
        "This is the ncut coarsener used by v39lam1gu.",
        fontsize=9.5,
        color=GRAY,
    )
    return save(fig, "06_real_ncut_dino.png")


def main():
    style()
    fig_idea_and_clock()
    fig_four_anneals()
    fig_ncut_pipeline()
    fig_toy_leveling()
    fig_apply_key()
    fig_real_crop()


if __name__ == "__main__":
    main()
