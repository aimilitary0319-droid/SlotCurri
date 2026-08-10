#!/usr/bin/env python3
"""Visualize clips where v10 beats baseline (uses existing all_val_metrics.csv)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from event_analysis.vis_v10_vs_baseline_losses import (
    hstack_labeled,
    load_model,
    one_hot_segmentations,
    overlay,
    prep_masks,
    score_and_masks,
    to_uint8_video,
)
from slotcurri import configuration, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument(
        "--metrics-csv",
        default="logs/vis_v10_loses_to_baseline/all_val_metrics.csv",
    )
    ap.add_argument("--out-dir", default="logs/vis_v10_beats_baseline")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--frame-stride", type=int, default=2)
    args = ap.parse_args()

    root = Path("/workspace/SlotCurri")
    if not (root / "slotcurri").exists():
        root = Path("/mnt/ssd2/hmlee/SlotCurri")

    metrics_path = root / args.metrics_csv
    rows = list(csv.DictReader(open(metrics_path)))
    for r in rows:
        r["sample"] = int(r["sample"])
        for k in (
            "baseline_ari",
            "baseline_mbo",
            "v10_ari",
            "v10_mbo",
            "d_ari",
            "d_mbo",
        ):
            r[k] = float(r[k])
        # d = baseline - v10; win when negative
        r["win_ari"] = -r["d_ari"]
        r["win_mbo"] = -r["d_mbo"]
        r["win_score"] = max(0.0, r["win_ari"]) + max(0.0, r["win_mbo"])

    chosen = [r for r in sorted(rows, key=lambda x: -x["win_score"]) if r["win_score"] > 0][
        : args.top_k
    ]
    want = {r["sample"] for r in chosen}

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "winners_table.csv").write_text("")
    with open(out_dir / "winners_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "rank",
                "sample",
                "win_score",
                "win_ari",
                "win_mbo",
                "baseline_ari",
                "v10_ari",
                "baseline_mbo",
                "v10_mbo",
            ],
        )
        w.writeheader()
        for i, r in enumerate(chosen):
            w.writerow(
                {
                    "rank": i,
                    "sample": r["sample"],
                    "win_score": r["win_score"],
                    "win_ari": r["win_ari"],
                    "win_mbo": r["win_mbo"],
                    "baseline_ari": r["baseline_ari"],
                    "v10_ari": r["v10_ari"],
                    "baseline_mbo": r["baseline_mbo"],
                    "v10_mbo": r["v10_mbo"],
                }
            )

    summary = {
        "n_val": len(rows),
        "n_win_ari_ge_0.05": sum(1 for r in rows if r["win_ari"] >= 0.05),
        "n_win_mbo_ge_0.05": sum(1 for r in rows if r["win_mbo"] >= 0.05),
        "n_win_both_ge_0.05": sum(
            1 for r in rows if r["win_ari"] >= 0.05 and r["win_mbo"] >= 0.05
        ),
        "mean_win_ari": float(np.mean([r["win_ari"] for r in rows])),
        "mean_win_mbo": float(np.mean([r["win_mbo"] for r in rows])),
        "chosen_samples": [r["sample"] for r in chosen],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    base_settings = root / "logs/_ytvis/settings/slotcurri/settings.yaml"
    base_ckpt = root / "logs/_ytvis/checkpoints/slotcurri_step=step=100000-v1.ckpt"
    v10_settings = root / "logs/_ytvis_attnmass_v10/settings/slotcurri/settings.yaml"
    v10_ckpt = root / "logs/_ytvis_attnmass_v10/checkpoints/slotcurri_step=step=100000-v1.ckpt"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = configuration.load_config(str(base_settings))
    cfg.dataset.num_val_workers = 0
    cfg.dataset.val_batch_size = 1
    dm = data.build(cfg.dataset, data_dir=args.data_dir)
    dm.setup("fit")
    loader = dm.val_dataloader()

    print("Loading models...")
    base = load_model(str(base_settings), str(base_ckpt), device)
    v10 = load_model(str(v10_settings), str(v10_ckpt), device)

    print(f"Visualizing {len(want)} winner clips:", sorted(want))
    for si, batch in enumerate(loader):
        if si not in want:
            continue
        meta = next(r for r in chosen if r["sample"] == si)
        rank_i = next(i for i, r in enumerate(chosen) if r["sample"] == si)
        print(
            f"viz sample {si}: winARI={meta['win_ari']:+.3f} winMBO={meta['win_mbo']:+.3f} "
            f"score={meta['win_score']:.3f}"
        )
        _, masks_b = score_and_masks(base, batch, device)
        _, masks_v = score_and_masks(v10, batch, device)
        video = to_uint8_video(batch["video"])
        spatial = video.shape[-2:]

        overlays = {}
        if "segmentations" in batch:
            seg = batch["segmentations"].cpu()
            if seg.ndim == 4:
                ncls = int(seg.max().item()) + 1
                gt = one_hot_segmentations(seg, max_classes=max(ncls, 2))
            else:
                gt = seg.bool()
            gt = prep_masks(gt.float(), spatial)
            overlays["gt"] = overlay(video, gt)

        overlays["baseline"] = overlay(video, prep_masks(masks_b, spatial))
        overlays["v10"] = overlay(video, prep_masks(masks_v, spatial))

        # label d as baseline-v10 (same as losers); also show win = -d
        labels = [
            "gt",
            f"baseline ARI={meta['baseline_ari']:.3f} mBO={meta['baseline_mbo']:.3f}",
            f"v10 ARI={meta['v10_ari']:.3f} mBO={meta['v10_mbo']:.3f}  "
            f"winARI={meta['win_ari']:+.3f} winMBO={meta['win_mbo']:+.3f}",
        ]
        order = ["gt", "baseline", "v10"]
        frames = hstack_labeled([overlays[k] for k in order], labels)
        frames = frames[:: max(args.frame_stride, 1)]
        mid = frames[len(frames) // 2]
        stem = (
            f"rank{rank_i:02d}_sample{si:03d}_"
            f"winARI{meta['win_ari']:+.3f}_winMBO{meta['win_mbo']:+.3f}"
        )
        Image.fromarray(mid).save(out_dir / f"{stem}_mid.png")
        try:
            import imageio

            imageio.mimsave(out_dir / f"{stem}.gif", frames, fps=4)
        except Exception as e:
            print("gif skip:", e)

    print("done ->", out_dir)


if __name__ == "__main__":
    main()
