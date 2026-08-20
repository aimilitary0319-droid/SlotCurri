"""Slot-level temporal analysis of the split score (uses split_score_probe CSV).

Frame-level split failed (merged territories are contiguous). Test the temporal
variant: a merged slot holds two INSTANCES, which sometimes separate spatially --
does max-over-time split (per slot, per clip) distinguish merged slots after all?

Slot label per (sample, slot), over frames where the slot is gated (g > 0.5)
and owns territory:
  merged: n_objs >= 2 in >= 25% of those frames
  single: n_objs == 1 in >= 75% and never >= 2
Scores: max split over frames, and 90th-percentile split (robust to single-frame
argmax noise).
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def auc_rank(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort().astype(np.float64) + 1.0
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="logs/split_score_probe/split_scores.csv")
    ap.add_argument("--gate-min", type=float, default=0.5)
    args = ap.parse_args()

    root = Path("/workspace/SlotCurri")
    if not (root / "slotcurri").exists():
        root = Path("/mnt/ssd2/hmlee/SlotCurri")

    groups = defaultdict(list)
    with open(root / args.csv) as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "":
                continue
            if float(r["gate"]) <= args.gate_min or float(r["win_frac"]) <= 0:
                continue
            groups[(int(r["sample"]), int(r["slot"]))].append(
                (float(r["split"]), int(r["n_objs"]))
            )

    merged_max, single_max = [], []
    merged_q90, single_q90 = [], []
    merged_keys, single_n = [], 0
    for key, vals in groups.items():
        if len(vals) < 4:
            continue
        splits = np.array([v[0] for v in vals])
        objs = np.array([v[1] for v in vals])
        frac2 = (objs >= 2).mean()
        frac1 = (objs == 1).mean()
        if frac2 >= 0.25:
            merged_max.append(splits.max())
            merged_q90.append(np.quantile(splits, 0.9))
            merged_keys.append((key, float(splits.max()), float(frac2)))
        elif frac1 >= 0.75 and (objs >= 2).sum() == 0:
            single_max.append(splits.max())
            single_q90.append(np.quantile(splits, 0.9))
            single_n += 1

    merged_max, single_max = np.array(merged_max), np.array(single_max)
    merged_q90, single_q90 = np.array(merged_q90), np.array(single_q90)

    print(f"slot-clips: merged={len(merged_max)}  single={single_n}")
    print(f"max-split  : merged med={np.median(merged_max):.3f} "
          f"single med={np.median(single_max):.3f}  AUC={auc_rank(merged_max, single_max):.3f}")
    print(f"q90-split  : merged med={np.median(merged_q90):.3f} "
          f"single med={np.median(single_q90):.3f}  AUC={auc_rank(merged_q90, single_q90):.3f}")

    if len(merged_max) and len(single_max):
        thresh = np.quantile(merged_max, 0.2)  # 80% recall on merged
        print(f"threshold @80% merged recall (max-split > {thresh:.3f}): "
              f"single FP rate = {float((single_max > thresh).mean()):.1%}")

    print("\nmerged slot-clips (sample, slot) with max split / merged-frame fraction:")
    for (key, mx, fr) in sorted(merged_keys, key=lambda x: -x[1])[:15]:
        print(f"  sample {key[0]:3d} slot {key[1]}: max split={mx:.3f}  merged frames={fr:.0%}")


if __name__ == "__main__":
    main()
