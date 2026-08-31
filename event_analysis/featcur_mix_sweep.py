"""Sweep FeatureSmoothing (tau, steps, window) and rank mix settings.

Reuses the v33 probe (same clips / part definitions). The ranking target is the
YTVIS failure mode: DINO head/torso parts should become more similar than two
different objects, without collapsing instances.

    margin   = part_cos - object_cos     (MOVi-raw ≈ +0.16)
    part_mix = (part_cos_1 - part_cos_0) / (1 - part_cos_0)
    inst_mix = (obj_cos_1  - obj_cos_0)  / (1 - obj_cos_0)
    score    = margin - 0.5 * inst_mix

Usage (inside the slotcurri container):
  python event_analysis/featcur_mix_sweep.py --data-dir /workspace/dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_strength_probe import DATASETS, _nanmean, run_dataset  # noqa: E402


def _wtag(window: Optional[int]) -> str:
    return "g" if window is None else str(int(window))


def setting(tau: Optional[float], steps: int, window: Optional[int]) -> dict:
    if tau is None:
        return {"name": "raw", "tau": None, "steps": 0, "window": None}
    return {
        "name": f"t{tau:g}_s{steps}_w{_wtag(window)}",
        "tau": float(tau),
        "steps": int(steps),
        "window": window,
    }


def build_settings() -> List[dict]:
    out = [setting(None, 0, None)]
    seen = {"raw"}
    # Main grid: keep the affinity graph (tau=0.1) and vary spatial reach × diffusion.
    for window in (3, 5, 7, 9, 13, 17, None):
        for steps in (1, 2, 3, 4, 5):
            st = setting(0.1, steps, window)
            if st["name"] not in seen:
                out.append(st)
                seen.add(st["name"])
    # Softer / sharper tau at the windows that can actually see a part pair.
    for tau in (0.08, 0.12, 0.15, 0.20):
        for window in (5, 7, 9):
            for steps in (1, 2, 3):
                st = setting(tau, steps, window)
                if st["name"] not in seen:
                    out.append(st)
                    seen.add(st["name"])
    return out


SETTINGS = build_settings()


def with_scores(agg: List[dict]) -> List[dict]:
    out = []
    for r in agg:
        rec = dict(r)
        part0 = r.get("kmeans_cos_raw", float("nan"))
        part1 = r.get("kmeans_cos_sm", float("nan"))
        obj0 = r.get("pair_cos_raw", float("nan"))
        obj1 = r.get("pair_cos_sm", float("nan"))
        spat0 = r.get("spatial_cos_raw", float("nan"))
        spat1 = r.get("spatial_cos_sm", float("nan"))
        c0 = r.get("contact_cos_raw", float("nan"))
        c1 = r.get("contact_cos_sm", float("nan"))
        var_p = r.get("kmeans_var_part_ratio", float("nan"))
        rec["margin"] = part1 - obj1
        rec["spatial_margin"] = spat1 - obj1
        rec["inst_mix"] = (obj1 - obj0) / max(1.0 - obj0, 1e-6)
        rec["contact_mix"] = (c1 - c0) / max(1.0 - c0, 1e-6)
        rec["part_gap"] = 1.0 - part1
        rec["obj_gap"] = 1.0 - obj1
        rec["gap_ratio"] = rec["part_gap"] / max(rec["obj_gap"], 1e-6)
        rec["rel_sep"] = ((1.0 - part1) / max(1.0 - part0, 1e-6)) / max(var_p, 1e-6)
        rec["score"] = rec["margin"] - 0.5 * rec["inst_mix"]
        out.append(rec)
    return out


def parse_name(name: str) -> dict:
    if name == "raw":
        return {"tau": None, "steps": 0, "window": None}
    # t0.1_s3_w9 / t0.1_s3_wg
    parts = name.split("_")
    tau = float(parts[0][1:])
    steps = int(parts[1][1:])
    w = parts[2][1:]
    window = None if w == "g" else int(w)
    return {"tau": tau, "steps": steps, "window": window}


def rank_table(scored: List[dict], dataset: str) -> List[dict]:
    rows = [r for r in scored if r["dataset"] == dataset and r["setting"] != "raw"]
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def pick_constrained(rows: List[dict], min_part_mix: float, max_inst_mix: float) -> Optional[dict]:
    cand = [
        r
        for r in rows
        if r.get("kmeans_mix", 0) >= min_part_mix and r.get("inst_mix", 1) <= max_inst_mix
    ]
    if not cand:
        return None
    cand.sort(key=lambda r: (r["margin"], -r["inst_mix"]), reverse=True)
    return cand[0]


def _window_order(windows):
    nums = sorted(w for w in windows if w is not None)
    if None in windows:
        nums.append(None)
    return nums


def plot_heatmaps(scored: List[dict], out_png: str) -> None:
    """tau=0.1 grid: window × steps for the four decision metrics."""
    metrics = [
        ("kmeans_mix", "part mix (2-means)", 0.0, 1.0, "viridis"),
        ("margin", "part-cos − object-cos", -0.05, 0.20, "RdYlGn"),
        ("inst_mix", "object–object mix", 0.0, 0.70, "YlOrRd"),
        ("score", "score = margin − 0.5·inst_mix", -0.15, 0.15, "RdYlGn"),
    ]
    datasets = ["ytvis", "movi_c"]
    fig, axes = plt.subplots(len(datasets), 4, figsize=(18, 7.2))
    for row, ds in enumerate(datasets):
        grid_rows = [
            r
            for r in scored
            if r["dataset"] == ds and r["setting"] != "raw" and parse_name(r["setting"])["tau"] == 0.1
        ]
        windows = _window_order({parse_name(r["setting"])["window"] for r in grid_rows})
        steps = sorted({parse_name(r["setting"])["steps"] for r in grid_rows})
        by = {}
        for r in grid_rows:
            p = parse_name(r["setting"])
            by[(p["window"], p["steps"])] = r
        ylabels = ["glob" if w is None else str(w) for w in windows]
        for col, (key, title, vmin, vmax, cmap) in enumerate(metrics):
            ax = axes[row, col]
            mat = np.full((len(windows), len(steps)), np.nan)
            for i, w in enumerate(windows):
                for j, s in enumerate(steps):
                    rec = by.get((w, s))
                    if rec is not None:
                        mat[i, j] = rec.get(key, np.nan)
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(steps)))
            ax.set_xticklabels(steps)
            ax.set_yticks(range(len(windows)))
            ax.set_yticklabels(ylabels)
            if row == len(datasets) - 1:
                ax.set_xlabel("n_steps")
            if col == 0:
                ax.set_ylabel(f"{ds}\nwindow")
            ax.set_title(title if row == 0 else "")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if np.isfinite(mat[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{mat[i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="black",
                        )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_tau_slice(scored: List[dict], out_png: str) -> None:
    """At window 7/9, compare tau across steps on YTVIS."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ytvis = [r for r in scored if r["dataset"] == "ytvis" and r["setting"] != "raw"]
    for ax, window in zip(axes, (7, 9)):
        rows = [r for r in ytvis if parse_name(r["setting"])["window"] == window]
        taus = sorted({parse_name(r["setting"])["tau"] for r in rows})
        for tau in taus:
            sub = [r for r in rows if parse_name(r["setting"])["tau"] == tau]
            sub.sort(key=lambda r: parse_name(r["setting"])["steps"])
            xs = [parse_name(r["setting"])["steps"] for r in sub]
            ys = [r["score"] for r in sub]
            ax.plot(xs, ys, "o-", label=f"tau={tau:g}")
        ax.set_xlabel("n_steps")
        ax.set_ylabel("score")
        ax.set_title(f"YTVIS window={window}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--max-clips", type=int, default=50)
    ap.add_argument("--frames-per-clip", type=int, default=2)
    ap.add_argument("--min-patches", type=int, default=16)
    ap.add_argument("--datasets", nargs="+", default=["ytvis", "movi_c"])
    ap.add_argument("--out-dir", default="event_analysis/featcur_mix_sweep")
    args = ap.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(
        f"device={device} clips={args.max_clips} settings={len(SETTINGS)}",
        flush=True,
    )
    for st in SETTINGS:
        print(f"  {st['name']}", flush=True)

    all_rows: List[dict] = []
    for ds in args.datasets:
        all_rows.extend(
            run_dataset(
                ds,
                DATASETS[ds],
                args.data_dir,
                device,
                args.max_clips,
                args.frames_per_clip,
                args.min_patches,
                settings=SETTINGS,
            )
        )

    raw_path = os.path.join(args.out_dir, "per_object.csv")
    if all_rows:
        with open(raw_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    # probe aggregate uses its own SETTINGS order; rebuild with ours
    keys = [
        "kmeans_cos_raw",
        "kmeans_cos_sm",
        "kmeans_mix",
        "kmeans_var_part_ratio",
        "p_kmeans_self",
        "p_kmeans_other_part",
        "p_kmeans_other_obj",
        "p_kmeans_bg",
        "spatial_cos_raw",
        "spatial_cos_sm",
        "spatial_mix",
        "spatial_var_part_ratio",
        "p_spatial_self",
        "p_spatial_other_part",
        "p_spatial_other_obj",
        "p_spatial_bg",
        "var_obj_ratio",
        "pair_cos_raw",
        "pair_cos_sm",
        "contact_cos_raw",
        "contact_cos_sm",
    ]
    agg = []
    for ds in args.datasets:
        for st in SETTINGS:
            sub = [r for r in all_rows if r["dataset"] == ds and r["setting"] == st["name"]]
            if not sub:
                continue
            rec = {
                "dataset": ds,
                "setting": st["name"],
                "n_objects": len(sub),
                "n_clips": len(set((r["clip"], r["frame"]) for r in sub)),
            }
            for k in keys:
                rec[k] = _nanmean([r[k] for r in sub])
            agg.append(rec)

    scored = with_scores(agg)
    agg_path = os.path.join(args.out_dir, "summary.csv")
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        w.writeheader()
        w.writerows(scored)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(scored, f, indent=2)

    heat = os.path.join(args.out_dir, "heatmaps_tau01.png")
    plot_heatmaps(scored, heat)
    tau_png = os.path.join(args.out_dir, "tau_slice.png")
    plot_tau_slice(scored, tau_png)

    print("\n===== YTVIS RANKING (score = margin - 0.5*inst_mix) =====")
    header = (
        f"{'setting':16} {'part_mix':>8} {'spat_mix':>8} {'margin':>7} "
        f"{'inst_mix':>8} {'ctc_mix':>7} {'score':>7} {'P→obj':>6} {'P→bg':>6}"
    )
    print(header)
    yt = rank_table(scored, "ytvis")
    for r in yt:
        print(
            f"{r['setting']:16} {r['kmeans_mix']:8.3f} {r['spatial_mix']:8.3f} "
            f"{r['margin']:7.3f} {r['inst_mix']:8.3f} {r['contact_mix']:7.3f} "
            f"{r['score']:7.3f} {r['p_kmeans_other_obj']:6.3f} {r['p_kmeans_bg']:6.3f}"
        )

    constrained = pick_constrained(yt, min_part_mix=0.55, max_inst_mix=0.35)
    print("\n===== CONSTRAINED PICK (part_mix>=0.55, inst_mix<=0.35, max margin) =====")
    if constrained:
        print(json.dumps({k: constrained[k] for k in (
            "setting", "kmeans_mix", "spatial_mix", "margin", "inst_mix",
            "contact_mix", "score", "kmeans_cos_sm", "pair_cos_sm",
        )}, indent=2))
    else:
        print("none")

    print(f"\nbest unconstrained YTVIS: {yt[0]['setting']} score={yt[0]['score']:.3f}")
    print(f"wrote {agg_path}\nwrote {heat}\nwrote {tau_png}")


if __name__ == "__main__":
    main()
