#!/usr/bin/env python3
"""Visualize clips where a method beats SlotCurri baseline (uses existing all_val_metrics.csv).

Default --method v10 --dataset ytvis reproduces logs/vis_v10_beats_baseline.
  python event_analysis/vis_v10_vs_baseline_wins.py --method v26
reads logs/vis_v26_loses_to_baseline/all_val_metrics.csv and writes
logs/vis_v26_beats_baseline.

  python event_analysis/vis_v10_vs_baseline_wins.py --method v39 --dataset movi_c
reads logs/vis_v39_movi_c_loses_to_baseline/all_val_metrics.csv and writes
logs/vis_v39_movi_c_beats_baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from event_analysis.vis_v10_vs_baseline_losses import (
    hstack_labeled,
    load_model,
    one_hot_segmentations,
    overlay,
    prep_masks,
    project_root,
    resolve_run,
    save_clip,
    score_and_masks,
    to_uint8_video,
)
from slotcurri import configuration, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--method", default="v10", help="attn-mass run tag, e.g. v10 or v26")
    ap.add_argument(
        "--dataset",
        default="ytvis",
        choices=("ytvis", "movi_c"),
        help="must match the losses scan that wrote all_val_metrics.csv",
    )
    ap.add_argument("--metrics-csv", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--frame-stride", type=int, default=2)
    args = ap.parse_args()

    method = args.method
    root = project_root()
    spec = resolve_run(root, method, args.dataset)
    ign_bg = spec["ignore_background"]
    metrics_path = root / (
        args.metrics_csv or f"logs/vis_{spec['out_stem']}_loses_to_baseline/all_val_metrics.csv"
    )
    out_dir = root / (args.out_dir or f"logs/vis_{spec['out_stem']}_beats_baseline")
    ari_key = f"{method}_ari"
    mbo_key = f"{method}_mbo"

    rows = list(csv.DictReader(open(metrics_path)))
    for r in rows:
        r["sample"] = int(r["sample"])
        for k in (
            "baseline_ari",
            "baseline_mbo",
            ari_key,
            mbo_key,
            "d_ari",
            "d_mbo",
        ):
            r[k] = float(r[k])
        # d = baseline - method; win when negative
        r["win_ari"] = -r["d_ari"]
        r["win_mbo"] = -r["d_mbo"]
        r["win_score"] = max(0.0, r["win_ari"]) + max(0.0, r["win_mbo"])

    chosen = [r for r in sorted(rows, key=lambda x: -x["win_score"]) if r["win_score"] > 0][
        : args.top_k
    ]
    want = {r["sample"] for r in chosen}

    out_dir.mkdir(parents=True, exist_ok=True)
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
                ari_key,
                "baseline_mbo",
                mbo_key,
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
                    ari_key: r[ari_key],
                    "baseline_mbo": r["baseline_mbo"],
                    mbo_key: r[mbo_key],
                }
            )

    summary = {
        "n_val": len(rows),
        "method": method,
        "dataset": spec["dataset"],
        "baseline": spec["baseline_name"],
        "baseline_ckpt": str(spec["baseline_ckpt"]),
        "ignore_background": ign_bg,
        "n_win_ari_ge_0.05": sum(1 for r in rows if r["win_ari"] >= 0.05),
        "n_win_mbo_ge_0.05": sum(1 for r in rows if r["win_mbo"] >= 0.05),
        "n_win_both_ge_0.05": sum(
            1 for r in rows if r["win_ari"] >= 0.05 and r["win_mbo"] >= 0.05
        ),
        "mean_win_ari": float(np.nanmean([r["win_ari"] for r in rows])),
        "mean_win_mbo": float(np.nanmean([r["win_mbo"] for r in rows])),
        "chosen_samples": [r["sample"] for r in chosen],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = configuration.load_config(str(spec["data_settings"]))
    cfg.dataset.num_val_workers = 0
    cfg.dataset.val_batch_size = 1
    dm = data.build(cfg.dataset, data_dir=args.data_dir)
    dm.setup("fit")
    loader = dm.val_dataloader()

    print("Loading models...")
    print(f"  dataset={spec['dataset']} baseline={spec['baseline_name']} {spec['baseline_ckpt']}")
    base = load_model(str(spec["baseline_settings"]), str(spec["baseline_ckpt"]), device)
    meth = load_model(str(spec["method_settings"]), str(spec["method_ckpt"]), device)
    print(
        f"  {method} src_gate={getattr(meth, 'amc_predictor_src_gate', None)}  "
        f"eval_src_gate={getattr(meth, 'amc_eval_predictor_src_gate', None)}  "
        f"eval_perron={getattr(meth, 'amc_eval_perron_readout', None)}  "
        f"eval_pair_iso={getattr(meth, 'amc_eval_predictor_pair_isolate', None)}"
    )

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
        _, masks_b = score_and_masks(base, batch, device, ign_bg)
        _, masks_v = score_and_masks(meth, batch, device, ign_bg)
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
        overlays[method] = overlay(video, prep_masks(masks_v, spatial))

        labels = [
            "gt",
            f"{spec['baseline_name']} ARI={meta['baseline_ari']:.3f} mBO={meta['baseline_mbo']:.3f}",
            f"{method} ARI={meta[ari_key]:.3f} mBO={meta[mbo_key]:.3f}  "
            f"winARI={meta['win_ari']:+.3f} winMBO={meta['win_mbo']:+.3f}",
        ]
        order = ["gt", "baseline", method]
        frames = hstack_labeled([overlays[k] for k in order], labels)
        frames = frames[:: max(args.frame_stride, 1)]
        stem = (
            f"rank{rank_i:02d}_sample{si:03d}_"
            f"winARI{meta['win_ari']:+.3f}_winMBO{meta['win_mbo']:+.3f}"
        )
        save_clip(frames, out_dir / stem)

    print("done ->", out_dir)


if __name__ == "__main__":
    main()
