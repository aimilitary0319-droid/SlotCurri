"""Probe the connected-component "split score" premise for the unimodality loss.

Question: when one slot merges several similar objects, is its claimed territory
actually multi-blob -- and is a clean single-object slot (large, elongated, or
occlusion-fragmented) safe from the same score?

Per (frame, slot) with A the last-iteration attention (softmax over slots):
  winner_f  = argmax_s A_{s,f}                      (patch ownership, 37x37 grid)
  blobs     = 8-connected components of the won set (specks < min patches dropped)
  split_s   = 1 - mass(largest blob) / mass(all kept blobs)   in [0, 1)
so one blob -> 0, two objects held together -> ~0.5. This is exactly the quantity
the proposed loss would penalize (gate-weighted, partition detached).

Slots are labeled from GT segmentations (nearest-downsampled to the patch grid):
  merged      : slot owns >= 50% of >= 2 GT objects        (should score HIGH)
  single      : owns exactly 1 GT object, contiguous       (should score LOW)
  single_frag : owns 1 GT object that is itself multi-part (occlusion trap; any
                purely spatial score is expected to fire here -- measure how often)
  none        : owns only background                        (reported, not scored)

Outputs: per-row CSV, group summary + Mann-Whitney AUC, and per-sample panels
(RGB / GT / ownership / per-slot attention with g, win%, split, #objs).

Usage (inside the slotcurri container):
  python event_analysis/split_score_probe.py --data-dir /workspace/dataset
"""
import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

from slotcurri import configuration, data, models
from slotcurri.data.transforms import Denormalize

STRUCT8 = np.ones((3, 3), dtype=int)


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(pos > neg). 1.0 = perfectly separable."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort().astype(np.float64) + 1.0
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def parse_sample_spec(spec: str):
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def frame_split_scores(att_fr: np.ndarray, grid: int, min_comp: int):
    """att_fr: (S, F) softmax over slots. Returns winner (F,), win_frac (S,), split (S,)."""
    n_slots, n_feat = att_fr.shape
    winner = att_fr.argmax(axis=0)
    win_frac = np.zeros(n_slots)
    split = np.full(n_slots, np.nan)
    for s in range(n_slots):
        won = winner == s
        win_frac[s] = won.mean()
        if not won.any():
            continue
        lab, n = ndimage.label(won.reshape(grid, grid), structure=STRUCT8)
        if n == 1:
            split[s] = 0.0
            continue
        masses, sizes = [], []
        att_map = att_fr[s].reshape(grid, grid)
        for ci in range(1, n + 1):
            comp = lab == ci
            sizes.append(int(comp.sum()))
            masses.append(float(att_map[comp].sum()))
        masses = np.array(masses)
        keep = np.array(sizes) >= min_comp
        if keep.sum() <= 1:
            split[s] = 0.0
        else:
            kept = masses[keep]
            split[s] = float(1.0 - kept.max() / kept.sum())
    return winner, win_frac, split


def frame_slot_labels(seg_fr: np.ndarray, winner: np.ndarray, n_slots: int, grid: int,
                      min_obj: int):
    """seg_fr: (F,) GT ids (0 = background). Returns n_objs (S,), frag (S,)."""
    n_objs = np.zeros(n_slots, dtype=int)
    frag = np.zeros(n_slots, dtype=bool)
    for oid in np.unique(seg_fr):
        if oid == 0:
            continue
        obj = seg_fr == oid
        if obj.sum() < min_obj:
            continue
        owner_counts = np.bincount(winner[obj], minlength=n_slots)
        s = int(owner_counts.argmax())
        if owner_counts[s] < 0.5 * obj.sum():
            continue  # no slot owns this object
        n_objs[s] += 1
        lab, n = ndimage.label(obj.reshape(grid, grid), structure=STRUCT8)
        if n > 1:
            sizes = np.bincount(lab.ravel())[1:]
            if int((sizes >= 2).sum()) >= 2:
                frag[s] = True
    return n_objs, frag


@torch.no_grad()
def forward_sample(model, batch, device):
    batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    outputs = model.forward(batch_dev, train=False, cycle=model.cyclic_inference)
    att = outputs["processor"]["state_attn_mask"].float()[0].cpu().numpy()  # (T, S, F)
    gate = outputs["processor"]["active_mask"].float()[0].cpu().numpy()  # (T, S)
    return att, gate


def visualize_sample(si, video, seg_small, att, gate, win_frac, split, n_objs, grid,
                     out_dir, frames_to_show=3):
    t_total, n_slots, _ = att.shape
    ts = sorted(set([0, t_total // 2, t_total - 1]))[:frames_to_show]
    denorm = Denormalize(input_type="video")
    rgb = denorm(video[0].cpu()).clamp(0, 1).permute(0, 2, 3, 1).numpy()  # (T, H, W, 3)

    ncols = 3 + n_slots
    fig, axes = plt.subplots(len(ts), ncols, figsize=(2.1 * ncols, 2.4 * len(ts)))
    if len(ts) == 1:
        axes = axes[None, :]
    cmap = plt.get_cmap("tab10")
    for ri, t in enumerate(ts):
        ax = axes[ri, 0]
        ax.imshow(rgb[t])
        ax.set_title(f"t={t}", fontsize=8)

        ax = axes[ri, 1]
        seg_map = seg_small[t].reshape(grid, grid)
        ax.imshow(seg_map, cmap="tab20", interpolation="nearest",
                  vmin=0, vmax=max(seg_small.max(), 1))
        ax.set_title("GT ids", fontsize=8)

        ax = axes[ri, 2]
        winner = att[t].argmax(axis=0).reshape(grid, grid)
        own = cmap(winner % 10)[..., :3]
        ax.imshow(own, interpolation="nearest")
        ax.set_title("ownership", fontsize=8)

        for s in range(n_slots):
            ax = axes[ri, 3 + s]
            ax.imshow(att[t, s].reshape(grid, grid), cmap="viridis", interpolation="nearest")
            sp = split[t, s]
            sp_txt = f"{sp:.2f}" if np.isfinite(sp) else "-"
            ax.set_title(
                f"s{s} g={gate[t, s]:.2f}\nwin={win_frac[t, s]:.0%} split={sp_txt} "
                f"obj={n_objs[t, s]}",
                fontsize=7,
                color=("tab:red" if np.isfinite(sp) and sp > 0.2 else "black"),
            )
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"sample {si:03d}  (split = 1 - largest-blob mass share of won territory)")
    fig.tight_layout()
    path = out_dir / f"sample{si:03d}_split.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings",
                    default="logs/_ytvis_attnmass_v25/settings/slotcurri/settings.yaml")
    ap.add_argument("--ckpt",
                    default="logs/_ytvis_attnmass_v25/checkpoints/"
                            "slotcurri_step=step=100000-v1.ckpt")
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--device", default="cuda:0")
    # normals 0-29 plus the vis_v10_loses_to_baseline failure clips
    ap.add_argument("--samples",
                    default="0-29,33,38,57,100,101,118,129,130,134,153,157,161,183,209")
    ap.add_argument("--vis-samples", default="0,1,22,33,100,129,130")
    ap.add_argument("--out-dir", default="logs/split_score_probe")
    ap.add_argument("--min-comp-patches", type=int, default=2,
                    help="ignore argmax specks smaller than this when scoring blobs")
    ap.add_argument("--min-obj-patches", type=int, default=4,
                    help="ignore GT objects smaller than this on the patch grid")
    ap.add_argument("--gate-min", type=float, default=0.5,
                    help="stats restricted to slots the gate would actually weight")
    ap.add_argument("--large-win-frac", type=float, default=0.10)
    args = ap.parse_args()

    wanted = parse_sample_spec(args.samples)
    vis_wanted = parse_sample_spec(args.vis_samples)
    wanted |= vis_wanted
    max_idx = max(wanted)

    root = Path("/workspace/SlotCurri")
    if not (root / "slotcurri").exists():
        root = Path("/mnt/ssd2/hmlee/SlotCurri")
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = configuration.load_config(str(root / args.settings))
    config.model.visualize = False
    config.dataset.num_val_workers = 0
    config.dataset.val_batch_size = 1

    model = models.build(config.model, config.optimizer)
    model.load_weights_from_checkpoint(str(root / args.ckpt))
    model.to(device).eval()

    dm = data.build(config.dataset, data_dir=args.data_dir)
    dm.setup("validate")
    loader = dm.val_dataloader()

    rows = []
    print(f"probing {len(wanted)} samples (up to index {max_idx}) with {args.ckpt}")
    for si, batch in enumerate(loader):
        if si > max_idx:
            break
        if si not in wanted:
            continue
        if "batch_padding_mask" in batch:
            batch = model._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                continue

        att, gate = forward_sample(model, batch, device)  # (T, S, F), (T, S)
        t_total, n_slots, n_feat = att.shape
        grid = int(round(n_feat ** 0.5))

        seg = batch["segmentations"][0]  # (T, H, W) ids or (T, C, H, W) one-hot
        if seg.ndim == 4:
            seg = seg.float().argmax(dim=1)  # all-zero pixels -> 0 = background
        seg_small = torch.nn.functional.interpolate(
            seg[:, None].float(), size=(grid, grid), mode="nearest"
        )[:, 0].long().numpy().reshape(t_total, -1)  # (T, F)

        win_frac = np.zeros((t_total, n_slots))
        split = np.full((t_total, n_slots), np.nan)
        n_objs = np.zeros((t_total, n_slots), dtype=int)
        frag = np.zeros((t_total, n_slots), dtype=bool)
        for t in range(t_total):
            winner, wf, sp = frame_split_scores(att[t], grid, args.min_comp_patches)
            no, fr = frame_slot_labels(seg_small[t], winner, n_slots, grid,
                                       args.min_obj_patches)
            win_frac[t], split[t], n_objs[t], frag[t] = wf, sp, no, fr
            for s in range(n_slots):
                rows.append({
                    "sample": si, "t": t, "slot": s,
                    "gate": float(gate[t, s]), "win_frac": float(wf[s]),
                    "split": float(sp[s]) if np.isfinite(sp[s]) else "",
                    "n_objs": int(no[s]), "frag": bool(fr[s]),
                })

        n_merged = int(((n_objs >= 2) & (gate > args.gate_min)).sum())
        print(f"  sample {si:03d}: T={t_total} merged-slot-frames={n_merged}")

        if si in vis_wanted:
            path = visualize_sample(si, batch["video"], seg_small, att, gate, win_frac,
                                    split, n_objs, grid, out_dir)
            print(f"    viz -> {path}")

    csv_path = out_dir / "split_scores.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")

    # ---- summary over gated, territory-owning slots ----
    def col(key, cast=float):
        return np.array([cast(r[key]) if r[key] != "" else np.nan for r in rows])

    gate_v = col("gate")
    win_v = col("win_frac")
    split_v = col("split")
    objs_v = col("n_objs", int)
    frag_v = np.array([r["frag"] for r in rows])

    valid = (gate_v > args.gate_min) & (win_v > 0) & np.isfinite(split_v)
    merged = valid & (objs_v >= 2)
    single = valid & (objs_v == 1) & ~frag_v
    single_frag = valid & (objs_v == 1) & frag_v
    single_large = single & (win_v >= args.large_win_frac)
    none_grp = valid & (objs_v == 0)

    print()
    print(f"rows kept for stats (gate>{args.gate_min}, wins>0): {int(valid.sum())}")
    header = f"{'group':>14s} {'n':>6s} {'split med':>10s} {'q10':>7s} {'q90':>7s} {'frac>0.2':>9s}"
    print(header)
    print("-" * len(header))
    for name, mask in (("merged", merged), ("single", single),
                       ("single_large", single_large), ("single_frag", single_frag),
                       ("none(bg)", none_grp)):
        v = split_v[mask]
        if v.size == 0:
            print(f"{name:>14s} {0:>6d}")
            continue
        print(f"{name:>14s} {v.size:>6d} {np.median(v):>10.3f} {np.quantile(v, 0.1):>7.3f}"
              f" {np.quantile(v, 0.9):>7.3f} {float((v > 0.2).mean()):>9.1%}")

    auc_single = auc_rank(split_v[merged], split_v[single])
    auc_large = auc_rank(split_v[merged], split_v[single_large])
    print()
    print(f"AUC merged-vs-single:       {auc_single:.3f}")
    print(f"AUC merged-vs-single_large: {auc_large:.3f}")

    if merged.sum() > 0 and single.sum() > 0:
        thresh = np.quantile(split_v[merged], 0.2)  # 80% recall on merged
        print(f"threshold @80% merged recall: split > {thresh:.3f}")
        for name, mask in (("single", single), ("single_large", single_large),
                           ("single_frag", single_frag)):
            v = split_v[mask]
            if v.size:
                print(f"  false-positive rate {name:>12s}: {float((v > thresh).mean()):.1%}")

    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
