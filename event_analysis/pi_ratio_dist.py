#!/usr/bin/env python3
"""Histogram of π / max_s(π) on a trained v39lam1gu checkpoint.

π is the decoder gate (active_mask): for v39lam1gu, max(λ1, 0) of the
8-neighbor graph. The temporal mix uses this ratio (state_max_norm).

Usage (slotcurri image):
  python event_analysis/pi_ratio_dist.py --max-clips 48 --max-frames 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from slotcurri import configuration, data, models


class _FakeTrainer:
    def __init__(self, step: int):
        self.global_step = step


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _slice_time(batch, max_frames: int):
    vid = batch.get("video")
    if vid is None or vid.ndim < 2 or vid.shape[1] <= max_frames:
        return batch
    batch = dict(batch)
    t = vid.shape[1]
    # Evenly spaced frames so a long clip is not only its opening.
    idx = np.linspace(0, t - 1, max_frames).round().astype(int)
    idx_t = torch.as_tensor(idx, device=vid.device)
    batch["video"] = vid.index_select(1, idx_t)
    seg = batch.get("segmentations")
    if torch.is_tensor(seg) and seg.ndim >= 2 and seg.shape[1] == t:
        batch["segmentations"] = seg.index_select(1, idx_t.to(seg.device))
    return batch


def _summarize(xs: np.ndarray) -> dict:
    if xs.size == 0:
        return {"n": 0}
    return {
        "n": int(xs.size),
        "mean": float(xs.mean()),
        "std": float(xs.std()),
        "min": float(xs.min()),
        "p10": float(np.percentile(xs, 10)),
        "p25": float(np.percentile(xs, 25)),
        "p50": float(np.percentile(xs, 50)),
        "p75": float(np.percentile(xs, 75)),
        "p90": float(np.percentile(xs, 90)),
        "max": float(xs.max()),
        "frac_ge_0.5": float((xs >= 0.5).mean()),
        "frac_ge_0.8": float((xs >= 0.8).mean()),
        "frac_lt_0.2": float((xs < 0.2).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="logs/_ytvis_attnmass_v39lam1gu/settings/slotcurri/settings.yaml",
    )
    ap.add_argument(
        "--ckpt",
        default="logs/_ytvis_attnmass_v39lam1gu/checkpoints/slotcurri_step=step=100000-v1.ckpt",
    )
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--max-clips", type=int, default=48)
    ap.add_argument("--max-frames", type=int, default=4)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="logs/pi_ratio_ytvis_v39lam1gu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = configuration.load_config(args.config)
    config.model.visualize = False
    config.dataset.val_batch_size = 1
    config.dataset.num_val_workers = 0

    model = models.build(config.model, config.optimizer)
    model.load_weights_from_checkpoint(args.ckpt)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    model._trainer = _FakeTrainer(args.step)

    dataset = data.build(config.dataset, data_dir=args.data_dir)
    dataset.setup("validate")
    loader = dataset.val_dataloader()

    ratios = []
    ratios_rest = []
    pi_vals = []
    n_clips = 0
    n_frames = 0

    print(
        f"ckpt={args.ckpt} device={device} max_clips={args.max_clips} "
        f"max_frames={args.max_frames}",
        flush=True,
    )

    for batch in loader:
        if "batch_padding_mask" in batch:
            batch = model._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                continue
        batch = _slice_time(batch, args.max_frames)
        batch = _to_device(batch, device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model.forward(batch, train=False, cycle=False)
        pi = outputs["processor"]["active_mask"].float()
        if pi.ndim == 2:
            pi = pi.unsqueeze(1)
        # (B, T, S)
        mx = pi.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        ratio = (pi / mx).detach().cpu().numpy()
        pi_np = pi.detach().cpu().numpy()
        winner = np.argmax(ratio, axis=-1)
        for b in range(ratio.shape[0]):
            for t in range(ratio.shape[1]):
                row = ratio[b, t]
                ratios.append(row)
                rest = np.delete(row, int(winner[b, t]))
                ratios_rest.append(rest)
                pi_vals.append(pi_np[b, t])
                n_frames += 1
        n_clips += 1
        if n_clips % 8 == 0:
            print(f"  clips {n_clips}/{args.max_clips}  frames={n_frames}", flush=True)
        if n_clips >= args.max_clips:
            break
        del outputs

    all_r = np.concatenate(ratios) if ratios else np.zeros(0)
    rest_r = np.concatenate(ratios_rest) if ratios_rest else np.zeros(0)
    all_pi = np.concatenate(pi_vals) if pi_vals else np.zeros(0)

    edges = np.linspace(0.0, 1.0, args.bins + 1)
    hist_all, _ = np.histogram(all_r, bins=edges)
    hist_rest, _ = np.histogram(rest_r, bins=edges)
    centers = (0.5 * (edges[:-1] + edges[1:])).tolist()

    report = {
        "ckpt": args.ckpt,
        "n_clips": n_clips,
        "n_frames": n_frames,
        "n_slots": int(all_r.size // max(n_frames, 1)),
        "max_frames": args.max_frames,
        "note": (
            "Eval forward, train=False (raw Keys). "
            "Frames subsampled evenly when a clip is longer than max_frames. "
            "ratio = π / max_s(π) per frame. rest excludes one argmax slot."
        ),
        "pi": _summarize(all_pi),
        "ratio_all_slots": _summarize(all_r),
        "ratio_nonmax": _summarize(rest_r),
        "bin_edges": edges.tolist(),
        "bin_centers": centers,
        "hist_all": hist_all.astype(int).tolist(),
        "hist_nonmax": hist_rest.astype(int).tolist(),
        "hist_all_density": (hist_all / max(all_r.size, 1)).tolist(),
        "hist_nonmax_density": (hist_rest / max(rest_r.size, 1)).tolist(),
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=False)
    labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(args.bins)]
    x = np.arange(args.bins)
    axes[0].bar(x, hist_all / max(all_r.size, 1), color="#1f77b4", width=0.9)
    axes[0].set_title("all slots  (winner is 1)")
    axes[1].bar(x, hist_rest / max(rest_r.size, 1), color="#ff7f0e", width=0.9)
    axes[1].set_title("non-max slots only")
    for ax in axes:
        ax.set_xticks(x[::2])
        ax.set_xticklabels(labels[::2], rotation=40, ha="right", fontsize=8)
        ax.set_xlabel(r"$\pi / \max_s \pi$")
        ax.set_ylabel("fraction of slots")
    fig.suptitle(
        f"YTVIS v39lam1gu  step 100k  n={n_clips} clips, {n_frames} frames",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "hist.png", dpi=140)
    plt.close(fig)

    print(json.dumps({k: report[k] for k in ("n_clips", "n_frames", "pi", "ratio_all_slots", "ratio_nonmax")}, indent=2))
    print(f"wrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
