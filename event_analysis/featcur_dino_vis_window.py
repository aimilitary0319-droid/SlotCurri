"""Window-curriculum (v34) DINO vis, same layout as featcur_dino_vis_schedule.py.

v34 operator: tau=0.1, n_steps=3, window w(t) = round(13 * (1-s(t))), s cosine 30k.
No raw/smooth blend: fully smoothed at the current w, raw when w=0.

Also draws a hop comparison (repeat P x at fixed w=13) so "hop" is visible, and
prints collapse stats (token variance ratio, cosine-to-centroid, part-cos).

Usage (inside the slotcurri container):
  python event_analysis/featcur_dino_vis_window.py --data-dir /workspace/dataset
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_dino_vis import object_crop, overlay_edges, pca_basis, pca_map, upsample  # noqa: E402
from featcur_dino_vis_parts import part_axis, part_cos  # noqa: E402
from featcur_strength_probe import apply_smooth  # noqa: E402
from slotcurri.models import feature_curriculum_mix, feature_curriculum_window
from slotcurri.modules.encoders import FeatureSmoothing

TAU = 0.1
N_STEPS = 3
W0 = 13
ANNEAL = 30000
# Steps where w actually changes (cosine lingers at 13 until ~3.8k).
SCHEDULE_STEPS = [0, 4000, 10000, 15000, 20000, 24000, 30000]
HOP_SETTINGS = [
    {"name": "raw", "window": None, "steps": 0, "pick": False},
    {"name": "1 hop  w13", "window": 13, "steps": 1, "pick": False},
    {"name": "v34  3 hops  w13", "window": 13, "steps": 3, "pick": True},
    {"name": "5 hops  w13", "window": 13, "steps": 5, "pick": False},
    {"name": "1 hop  global", "window": None, "steps": 1, "pick": False},
]


def w_at(step: int) -> int:
    return feature_curriculum_window(step, ANNEAL, W0, "cosine")


def s_at(step: int) -> float:
    return feature_curriculum_mix(step, ANNEAL, "cosine")


def weff_at(window: int, n_tokens: int) -> int:
    if window <= 0:
        return 0
    return int(
        FeatureSmoothing(tau=TAU, window=window, window_ref_grid=37).effective_window(n_tokens)
        or 0
    )


def smooth_w(raw: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 0:
        return raw
    return apply_smooth(raw, TAU, window, N_STEPS, window_ref_grid=37)


def collapse_stats(feat: torch.Tensor, raw: torch.Tensor) -> dict:
    """How close the token cloud is to a single vector."""
    raw_f = raw.float()
    feat_f = feat.float()
    var_raw = float(raw_f.var(dim=0, unbiased=False).mean().item())
    var_feat = float(feat_f.var(dim=0, unbiased=False).mean().item())
    xn = F.normalize(feat_f, dim=-1)
    mu = F.normalize(xn.mean(0), dim=0)
    mean_cos = float((xn * mu).sum(-1).mean().item())
    return {
        "var_ratio": var_feat / max(var_raw, 1e-12),
        "mean_cos_to_centroid": mean_cos,
    }


def instance_cos(feat: torch.Tensor, gt_flat: torch.Tensor, bg_id: int = -1) -> float:
    """Mean pairwise cosine of GT-object means. nan if <2 objects."""
    ids = [int(i) for i in gt_flat.unique().tolist() if int(i) != bg_id]
    means = []
    for oid in ids:
        idx = (gt_flat == oid).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() < 4:
            continue
        means.append(F.normalize(feat[idx].float().mean(0), dim=0))
    if len(means) < 2:
        return float("nan")
    m = torch.stack(means, 0)
    sim = (m @ m.T).fill_diagonal_(float("nan"))
    return float(sim.nanmean().item())


@torch.no_grad()
def attach_schedule(fr, device):
    raw = fr["raw"].to(device)
    ia, ib = fr["ia"].to(device), fr["ib"].to(device)
    axis = part_axis(raw, ia, ib)
    mid = 0.5 * (raw[ia].float().mean(0) + raw[ib].float().mean(0))
    grid = int(round(math.sqrt(raw.shape[0])))
    obj = torch.cat([ia, ib])
    raw_proj = ((raw.float() - mid) @ axis).reshape(grid, grid)
    ys, xs = obj // grid, obj % grid
    vmax = max(float(raw_proj[ys, xs].abs().quantile(0.95).item()), 1e-3)
    basis = pca_basis(raw)
    proj_raw = (raw.float() - raw.float().mean(0)) @ basis
    lo = proj_raw.quantile(0.02, dim=0)
    hi = proj_raw.quantile(0.98, dim=0)
    gt_flat = fr["gt_grid"].to(device).reshape(-1)
    bg_id = int(fr.get("bg_id", -1))
    n_tokens = int(raw.shape[0])

    part_maps, pca_maps, scores, stats = {}, {}, {}, {}
    weffs = {}
    for step in SCHEDULE_STEPS:
        feat = smooth_w(raw, w_at(step))
        part_maps[step] = ((feat.float() - mid) @ axis).reshape(grid, grid).cpu().numpy()
        pca_maps[step] = pca_map(feat, basis, lo, hi, grid)
        scores[step] = part_cos(feat, ia, ib)
        st = collapse_stats(feat, raw)
        st["part_cos"] = scores[step]
        st["inst_cos"] = instance_cos(feat, gt_flat, bg_id=bg_id)
        st["window"] = w_at(step)
        st["weff"] = weff_at(w_at(step), n_tokens)
        stats[step] = st
        weffs[step] = st["weff"]

    hop_pca, hop_part, hop_scores, hop_stats = {}, {}, {}, {}
    for hs in HOP_SETTINGS:
        if hs["steps"] <= 0:
            feat = raw
        else:
            feat = apply_smooth(raw, TAU, hs["window"], hs["steps"], window_ref_grid=37)
        hop_pca[hs["name"]] = pca_map(feat, basis, lo, hi, grid)
        hop_part[hs["name"]] = ((feat.float() - mid) @ axis).reshape(grid, grid).cpu().numpy()
        hop_scores[hs["name"]] = part_cos(feat, ia, ib)
        st = collapse_stats(feat, raw)
        st["part_cos"] = hop_scores[hs["name"]]
        st["inst_cos"] = instance_cos(feat, gt_flat, bg_id=bg_id)
        hop_stats[hs["name"]] = st

    fr["part_maps"] = part_maps
    fr["pca_maps"] = pca_maps
    fr["scores"] = scores
    fr["stats"] = stats
    fr["vmax"] = vmax
    fr["windows"] = {step: w_at(step) for step in SCHEDULE_STEPS}
    fr["weffs"] = weffs
    fr["hop_pca"] = hop_pca
    fr["hop_part"] = hop_part
    fr["hop_scores"] = hop_scores
    fr["hop_stats"] = hop_stats
    return fr


def _crop_box(fr):
    rgb = fr["rgb"]
    h, w = rgb.shape[:2]
    dummy = np.zeros((h, w, 3))
    crop = object_crop(rgb / 255.0, dummy, fr["gt_grid"], fr["oid"], pad=6)
    if crop is None:
        return 0, h, 0, w
    _, _, box = crop
    return box


def draw_window_curve(ax) -> None:
    xs = np.linspace(0, ANNEAL, 400)
    ys = [w_at(int(s)) for s in xs]
    ax.plot(xs / 1000.0, ys, color="black", lw=1.8)
    for step in SCHEDULE_STEPS:
        ax.scatter([step / 1000.0], [w_at(step)], zorder=3, color="#c0392b", s=28)
        ax.annotate(
            f"w={w_at(step)}",
            (step / 1000.0, w_at(step)),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7,
        )
    ax.set_xlim(-0.5, 31)
    ax.set_ylim(-0.6, W0 + 1.4)
    ax.set_xlabel("train step (k)", fontsize=8)
    ax.set_ylabel("Chebyshev window w", fontsize=8)
    ax.set_title(
        f"v34 window clock   ·   x = smooth(τ={TAU}, w(t), hops={N_STEPS})   "
        f"or raw if w=0    ·    w0={W0}  n_steps={N_STEPS}",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)


def draw_part_axis(frames, out_path: str, note: str = "") -> None:
    n_row = len(frames)
    n_col = 1 + len(SCHEDULE_STEPS)
    fig = plt.figure(figsize=(2.55 * n_col, 3.05 * n_row + 1.6))
    gs = fig.add_gridspec(n_row + 1, n_col, height_ratios=[0.55] + [1] * n_row)
    axc = fig.add_subplot(gs[0, :])
    draw_window_curve(axc)

    cmap = plt.get_cmap("coolwarm")
    for r, fr in enumerate(frames):
        y0, y1, x0, x1 = _crop_box(fr)
        rgb = fr["rgb"]
        ax = fig.add_subplot(gs[r + 1, 0])
        ax.imshow(rgb[y0:y1, x0:x1])
        ax.set_ylabel(f"clip {fr['clip']}", fontsize=9)
        ax.set_title("RGB", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        norm = TwoSlopeNorm(vmin=-fr["vmax"], vcenter=0.0, vmax=fr["vmax"])
        h, w = rgb.shape[:2]
        for c, step in enumerate(SCHEDULE_STEPS, start=1):
            m = upsample(fr["part_maps"][step][..., None], h, w)[y0:y1, x0:x1, 0]
            ax = fig.add_subplot(gs[r + 1, c])
            im = ax.imshow(m, cmap=cmap, norm=norm)
            ww = fr["windows"][step]
            we = fr.get("weffs", {}).get(step)
            wtag = f"w={ww}" if we in (None, ww) else f"w={ww}→{we}"
            ax.set_title(
                f"{step // 1000}k  {wtag}  {N_STEPS}hop\n"
                f"part-cos {fr['scores'][step]:.2f}",
                fontsize=7,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if c == n_col - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Part axis along v34 (τ=0.1, 3 hops, w: 13→0). Red/blue = raw head vs torso. "
        "Left = coarse scene. Right = 30k raw DINO."
        + note,
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def draw_pca(frames, out_path: str, note: str = "") -> None:
    n_row = len(frames)
    n_col = 1 + len(SCHEDULE_STEPS)
    fig, axes = plt.subplots(n_row, n_col, figsize=(2.4 * n_col, 2.45 * n_row))
    if n_row == 1:
        axes = axes[None, :]
    for r, fr in enumerate(frames):
        rgb = fr["rgb"]
        h, w = rgb.shape[:2]
        edge = fr.get("edge")
        show = overlay_edges(rgb, edge) if edge is not None else rgb / 255.0
        ax = axes[r, 0]
        ax.imshow(show)
        ax.set_ylabel(f"clip {fr['clip']}", fontsize=8)
        ax.set_title("RGB + GT", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for c, step in enumerate(SCHEDULE_STEPS, start=1):
            pca = upsample(fr["pca_maps"][step], h, w)
            if edge is not None:
                pca = overlay_edges(pca, edge, color=(1, 1, 1))
            ax = axes[r, c]
            ax.imshow(pca)
            st = fr["stats"][step]
            inst = st["inst_cos"]
            inst_s = f"  inst {inst:.2f}" if inst == inst else ""
            if r == 0:
                we = fr.get("weffs", {}).get(step)
                ww = fr["windows"][step]
                wtag = f"w={ww}" if we in (None, ww) else f"w={ww}→{we}"
                ax.set_title(
                    f"{step // 1000}k  {wtag}\n{N_STEPS} hops",
                    fontsize=8,
                )
            ax.set_xlabel(f"var {st['var_ratio']:.2f}{inst_s}", fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "Same PCA (fit on raw) along v34: 3 hops at the current w. "
        "Left should be coarse (few blobs). Right is raw."
        + note,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def draw_hop_schematic(ax) -> None:
    """1-D cartoon: one hop stays inside the window; more hops walk the graph."""
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.35, 1.55)
    ax.axis("off")
    ax.set_title(
        "A hop is one multiply x ← P x.  P averages each patch with similar neighbours "
        "inside the Chebyshev window.  Repeating P lets information walk: radius ≈ hops × w.",
        fontsize=9,
        loc="left",
    )
    xs = np.arange(11)
    ax.scatter(xs, np.zeros_like(xs), s=36, c="#444444", zorder=3)
    ax.annotate("patch i", (5, 0.0), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=8)
    # 1 hop window w=2
    ax.add_patch(FancyBboxPatch((2.55, -0.12), 4.9, 0.24, boxstyle="round,pad=0.02",
                                facecolor="#4e79a7", alpha=0.18, edgecolor="#4e79a7"))
    ax.annotate("1 hop, w=2: i only sees i±2", (5, 0.38), ha="center", fontsize=8,
                color="#4e79a7")
    # 2 hops
    ax.annotate("", xy=(1, 0.85), xytext=(5, 0.55),
                arrowprops=dict(arrowstyle="->", color="#e15759", lw=1.2))
    ax.annotate("", xy=(9, 0.85), xytext=(5, 0.55),
                arrowprops=dict(arrowstyle="->", color="#e15759", lw=1.2))
    ax.annotate("2 hops: i±2 already mixed with i±4, so i feels radius 4",
                (5, 1.15), ha="center", fontsize=8, color="#e15759")
    ax.add_patch(Rectangle((0.55, 0.72), 8.9, 0.22, fill=False, ls="--",
                           edgecolor="#e15759", lw=1.0))


def draw_hops_pca(frames, out_path: str) -> None:
    n_row = len(frames)
    n_col = 1 + len(HOP_SETTINGS)
    fig = plt.figure(figsize=(2.45 * n_col, 2.55 * n_row + 1.7))
    gs = fig.add_gridspec(n_row + 1, n_col, height_ratios=[0.62] + [1] * n_row)
    axh = fig.add_subplot(gs[0, :])
    draw_hop_schematic(axh)
    for r, fr in enumerate(frames):
        rgb = fr["rgb"]
        h, w = rgb.shape[:2]
        edge = fr.get("edge")
        show = overlay_edges(rgb, edge) if edge is not None else rgb / 255.0
        ax = fig.add_subplot(gs[r + 1, 0])
        ax.imshow(show)
        ax.set_ylabel(f"clip {fr['clip']}", fontsize=8)
        ax.set_title("RGB + GT", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for c, hs in enumerate(HOP_SETTINGS, start=1):
            pca = upsample(fr["hop_pca"][hs["name"]], h, w)
            if edge is not None:
                pca = overlay_edges(pca, edge, color=(1, 1, 1))
            ax = fig.add_subplot(gs[r + 1, c])
            ax.imshow(pca)
            vr = fr["hop_stats"][hs["name"]]["var_ratio"]
            if r == 0:
                ax.set_title(hs["name"], fontsize=8,
                             color="#c0392b" if hs["pick"] else "black",
                             fontweight="bold" if hs["pick"] else "normal")
            ax.set_xlabel(f"var {vr:.2f}", fontsize=7)
            if hs["pick"]:
                for spine in ax.spines.values():
                    spine.set_color("#c0392b")
                    spine.set_linewidth(2.2)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "Same raw-fit PCA. v34 is the red column (1 hop, w=13). "
        "More hops / global = closer to one color (feature collapse).",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def draw_hops_part(frames, out_path: str) -> None:
    n_row = len(frames)
    n_col = 1 + len(HOP_SETTINGS)
    fig, axes = plt.subplots(n_row, n_col, figsize=(2.55 * n_col, 3.05 * n_row))
    if n_row == 1:
        axes = axes[None, :]
    cmap = plt.get_cmap("coolwarm")
    for r, fr in enumerate(frames):
        y0, y1, x0, x1 = _crop_box(fr)
        rgb = fr["rgb"]
        h, w = rgb.shape[:2]
        ax = axes[r, 0]
        ax.imshow(rgb[y0:y1, x0:x1])
        ax.set_ylabel(f"clip {fr['clip']}", fontsize=9)
        ax.set_title("RGB", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        norm = TwoSlopeNorm(vmin=-fr["vmax"], vcenter=0.0, vmax=fr["vmax"])
        for c, hs in enumerate(HOP_SETTINGS, start=1):
            m = upsample(fr["hop_part"][hs["name"]][..., None], h, w)[y0:y1, x0:x1, 0]
            ax = axes[r, c]
            im = ax.imshow(m, cmap=cmap, norm=norm)
            ax.set_title(
                f"{hs['name']}\npart-cos {fr['hop_scores'][hs['name']]:.2f}",
                fontsize=7,
                color="#c0392b" if hs["pick"] else "black",
                fontweight="bold" if hs["pick"] else "normal",
            )
            if hs["pick"]:
                for spine in ax.spines.values():
                    spine.set_color("#c0392b")
                    spine.set_linewidth(2.2)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == n_col - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Part axis vs hops at fixed w=13. Flat / white = head and torso became one vector.",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _attach_edges(fr):
    g = fr["gt_grid"].numpy()
    edge = np.zeros(g.shape, dtype=bool)
    edge[1:] |= g[1:] != g[:-1]
    edge[:, 1:] |= g[:, 1:] != g[:, :-1]
    h, w = fr["rgb"].shape[:2]
    t = torch.from_numpy(edge.astype(np.float32))[None, None]
    fr["edge"] = F.interpolate(t, size=(h, w), mode="nearest")[0, 0].numpy() > 0.5
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--dataset", choices=("ytvis", "movi_c"), default="ytvis")
    ap.add_argument("--min-patches", type=int, default=16)
    ap.add_argument("--max-clips", type=int, default=40)
    ap.add_argument("--out-dir", default="event_analysis/featcur_window_vis")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device={device} dataset={args.dataset}", flush=True)
    print(
        "windows:",
        {s: w_at(s) for s in SCHEDULE_STEPS},
        "s:",
        {s: f"{s_at(s):.3f}" for s in SCHEDULE_STEPS},
        flush=True,
    )

    if args.dataset == "movi_c":
        from featcur_dino_vis_movi import collect_movi

        frames = collect_movi(args.data_dir, device, args.max_clips, args.min_patches, n_keep=6)
        for fr in frames:
            fr["bg_id"] = 0
            if "edge" not in fr:
                _attach_edges(fr)
    else:
        import featcur_dino_vis_parts as vp

        vp.TARGET_CLIPS = {"ytvis": [4, 50, 23, 18, 14, 8]}
        frames = vp.collect_targets(args.data_dir, device, args.min_patches)
        frames = [_attach_edges(fr) for fr in frames]
        for fr in frames:
            fr["bg_id"] = -1
    frames = [attach_schedule(fr, device) for fr in frames]

    print("\npart-cos / inst-cos along window clock:", flush=True)
    for fr in frames:
        print(
            f" clip {fr['clip']}: "
            + ", ".join(
                f"{s // 1000}k(w={w_at(s)}→{fr['stats'][s]['weff']})="
                f"p{fr['scores'][s]:.3f}/i{fr['stats'][s]['inst_cos']:.3f}"
                for s in SCHEDULE_STEPS
            ),
            flush=True,
        )
    print("\ncollapse (var_ratio / cos-to-centroid) at w=13 vs hops:", flush=True)
    for fr in frames:
        bits = []
        for hs in HOP_SETTINGS:
            st = fr["hop_stats"][hs["name"]]
            bits.append(
                f"{hs['name']}: var={st['var_ratio']:.3f} mid={st['mean_cos_to_centroid']:.3f} "
                f"part={st['part_cos']:.3f} inst={st['inst_cos']:.3f}"
            )
        print(f" clip {fr['clip']}:\n  " + "\n  ".join(bits), flush=True)

    summary = {
        "dataset": args.dataset,
        "windows": {str(s): w_at(s) for s in SCHEDULE_STEPS},
        "clips": {
            str(fr["clip"]): {
                "schedule": {str(s): fr["stats"][s] for s in SCHEDULE_STEPS},
                "hops": fr["hop_stats"],
            }
            for fr in frames
        },
    }
    tag = "movi" if args.dataset == "movi_c" else "ytvis"
    with open(os.path.join(args.out_dir, f"collapse_stats_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    note = ""
    if frames and frames[0].get("weffs"):
        we0 = frames[0]["weffs"][SCHEDULE_STEPS[0]]
        if we0 != W0:
            note = f"  Grid-scaled: w0=13 → weff={we0} on this token grid."
    if args.dataset == "movi_c":
        part_frames = frames[:4]
        prefix = "dino_v34_movi"
    else:
        part_frames = [fr for fr in frames if fr["clip"] in (4, 50, 23, 18)]
        prefix = "dino_v34"
    draw_part_axis(
        part_frames, os.path.join(args.out_dir, f"{prefix}_schedule_part_axis.png"), note
    )
    draw_pca(frames, os.path.join(args.out_dir, f"{prefix}_schedule_pca.png"), note)
    draw_hops_pca(frames, os.path.join(args.out_dir, f"{prefix}_hops_pca.png"))
    draw_hops_part(part_frames, os.path.join(args.out_dir, f"{prefix}_hops_part_axis.png"))
    print(
        f"wrote {prefix}_schedule_part_axis.png\n"
        f"wrote {prefix}_schedule_pca.png\n"
        f"wrote {prefix}_hops_pca.png\n"
        f"wrote {prefix}_hops_part_axis.png",
        flush=True,
    )


if __name__ == "__main__":
    main()
