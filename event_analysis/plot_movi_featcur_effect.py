"""Plot MOVi-C feature-curriculum occupancy vs the mix clock."""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(name):
    path = os.path.join("logs", f"_{name}", "metrics", "slotcurri", "metrics.csv")
    train, val = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                step = float(row.get("step") or "")
            except ValueError:
                continue

            def g(k):
                v = row.get(k, "")
                if not v:
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None

            if g("val/ari") is not None:
                val.append((step, g("val/ari"), g("val/mbo")))
            elif g("train/active_slots") is not None:
                train.append((step, g("train/active_slots"), g("train/featcur_mix")))
    return train, val


def main():
    v33t, v33v = load("movi_c_attnmass_v33")
    t2t, t2v = load("movi_c_attnmass_v33t2w3s3")
    y32t, _ = load("ytvis_attnmass_v32")
    y33t, _ = load("ytvis_attnmass_v33")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))

    ax = axes[0, 0]
    ax.plot([s / 1000 for s, _, _ in v33t], [a for _, a, _ in v33t], color="#2c7fb8", lw=1.2, label="MOVi v33")
    ax.plot([s / 1000 for s, _, _ in t2t], [a for _, a, _ in t2t], color="#e6550d", lw=1.2, label="MOVi t2w3s3")
    ax.axvline(30, color="0.4", ls="--", lw=1)
    ax.axhline(11, color="0.7", ls=":", lw=1)
    ax.set_ylim(0, 11.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("step (k)")
    ax.set_ylabel("train Σc  (of 11)")
    ax.set_title("MOVi-C occupancy  (dashed = mix hits 1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(
        [s / 1000 for s, _, m in v33t if m is not None],
        [m for *_, m in v33t if m is not None],
        color="0.2",
        lw=1.6,
    )
    ax.axvline(30, color="0.4", ls="--", lw=1)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(0, 100)
    ax.set_xlabel("step (k)")
    ax.set_ylabel("featcur mix s")
    ax.set_title("same mix clock on both MOVi runs")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot([s / 1000 for s, a, _ in v33v], [a for _, a, _ in v33v], color="#2c7fb8", lw=1.4, label="v33 ARI")
    ax.plot([s / 1000 for s, a, _ in t2v], [a for _, a, _ in t2v], color="#e6550d", lw=1.4, label="t2 ARI")
    ax.axvline(30, color="0.4", ls="--", lw=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0.3, 0.75)
    ax.set_xlabel("step (k)")
    ax.set_ylabel("val video ARI")
    ax.set_title("MOVi-C val ARI")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot([s / 1000 for s, a, _ in y32t], [a for _, a, _ in y32t], color="0.35", lw=1.3, label="YTVIS v32 (no featcur)")
    ax.plot([s / 1000 for s, a, _ in y33t], [a for _, a, _ in y33t], color="#2c7fb8", lw=1.3, label="YTVIS v33")
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 7.2)
    ax.axhline(7, color="0.7", ls=":", lw=1)
    ax.set_xlabel("step (k)")
    ax.set_ylabel("train Σc  (of 7)")
    ax.set_title("YTVIS control: featcur does not delay fill")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Feature curriculum on MOVi-C: occupancy is not paced by mix s(t)", fontsize=12)
    fig.tight_layout()
    out = "event_analysis/featcur_window_vis/movi_featcur_effect.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
