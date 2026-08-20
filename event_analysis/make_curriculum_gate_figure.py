"""Two-panel figure in a paper style."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

OUT = Path("/mnt/ssd2/hmlee/SlotCurri/event_analysis/slide_curriculum_gate.png")

S = 7
TAU_G = 0.5
EPS = 1e-6
T_ANNEAL, T_MAX = 75_000, 100_000

NAVY = "#1F3A5F"
TEAL = "#3C6E71"
GRAY = "#6E6E6E"
INK = "#111111"


def schedule(step):
    q = np.clip(step / T_ANNEAL, 0.0, 1.0)
    lam = 0.5 * (1.0 - np.cos(np.pi * q))
    p = (1.5 + (0.1 - 1.5) * lam) / S
    beta = 1.0 - lam * 0.3
    return p, beta


def gate(m, c, p, beta):
    r = np.exp(beta * np.log(m + EPS) + (1.0 - beta) * np.log(c + EPS))
    return 1.0 / (1.0 + np.exp(-(np.log(r + EPS) - np.log(p + EPS)) / TAU_G))


def kfmt(x, _):
    return "0" if x == 0 else f"{int(x / 1000)}"


def apply_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ax.spines.values():
        s.set_color(INK)
        s.set_linewidth(0.8)
    ax.tick_params(
        direction="in",
        length=3.2,
        width=0.8,
        colors=INK,
        labelsize=8.5,
        pad=3,
    )
    ax.set_xlim(0, T_MAX)
    ax.set_xticks([0, 25000, 50000, 75000, 100000])
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.xaxis.set_minor_locator(MultipleLocator(12500))
    ax.tick_params(which="minor", direction="in", length=1.6, width=0.6)


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "lines.solid_capstyle": "round",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    t = np.linspace(0, T_MAX, 800)
    p, beta = schedule(t)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), dpi=220)
    fig.subplots_adjust(left=0.08, right=0.93, top=0.90, bottom=0.22, wspace=0.42)

    # (a)
    ax = axes[0]
    ax.plot(t, p, color=NAVY, lw=1.6, label=r"$p_t$")
    ax.set_ylim(0.0, 0.25)
    ax.set_yticks([0.00, 0.10, 0.20])
    ax.set_xlabel(r"Training step ($\times 10^3$)", labelpad=4)
    ax.set_ylabel(r"Threshold $p_t$", labelpad=4)
    apply_axes(ax)

    axr = ax.twinx()
    axr.plot(t, beta, color=TEAL, lw=1.6, ls=(0, (3.2, 2.0)), label=r"$\beta_t$")
    axr.set_ylim(0.64, 1.06)
    axr.set_yticks([0.7, 1.0])
    axr.set_ylabel(r"Weight $\beta_t$", labelpad=4)
    axr.spines["top"].set_visible(False)
    axr.spines["left"].set_visible(False)
    axr.spines["right"].set_linewidth(0.8)
    axr.spines["right"].set_color(INK)
    axr.tick_params(direction="in", length=3.2, width=0.8, colors=INK, labelsize=8.5, pad=3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        frameon=False,
        loc="upper right",
        fontsize=8,
        handlelength=1.8,
        borderaxespad=0.2,
    )
    ax.text(-0.18, 1.06, r"(a)", transform=ax.transAxes, fontsize=10, va="bottom")

    # (b)
    ax = axes[1]
    ax.axhline(0.5, color="#B5B5B5", lw=0.7, ls=":", zorder=0)
    ax.plot(t, gate(0.30, 0.94, p, beta), color=NAVY, lw=1.6, label="Large")
    ax.plot(t, gate(0.024, 0.90, p, beta), color=TEAL, lw=1.6, label="Small")
    ax.plot(t, gate(0.005, 0.10, p, beta), color=GRAY, lw=1.6, ls=(0, (3.2, 2.0)), label="Ghost")
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel(r"Training step ($\times 10^3$)", labelpad=4)
    ax.set_ylabel(r"Gate $g_s$", labelpad=4)
    apply_axes(ax)
    ax.legend(
        frameon=False,
        loc="center left",
        fontsize=8,
        handlelength=1.8,
        borderaxespad=0.4,
    )
    ax.text(-0.18, 1.06, r"(b)", transform=ax.transAxes, fontsize=10, va="bottom")

    fig.savefig(OUT, dpi=220, bbox_inches="tight", pad_inches=0.04)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
