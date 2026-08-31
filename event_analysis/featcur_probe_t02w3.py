"""Measure tau=0.2, window=3, n_steps=5 vs the previous pick (tau=0.1 s5 w5)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_mix_sweep import setting, with_scores  # noqa: E402
from featcur_strength_probe import DATASETS, _nanmean, run_dataset  # noqa: E402

SETTINGS = [
    setting(None, 0, None),
    setting(0.1, 5, 5),
    setting(0.1, 5, 3),
    setting(0.2, 1, 3),
    setting(0.2, 3, 3),
    setting(0.2, 5, 3),
    setting(0.2, 5, 5),
    setting(0.15, 5, 3),
]


def agg_rows(all_rows: List[dict]) -> List[dict]:
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
        "var_obj_ratio",
        "pair_cos_raw",
        "pair_cos_sm",
        "contact_cos_raw",
        "contact_cos_sm",
    ]
    out = []
    for ds in sorted(set(r["dataset"] for r in all_rows)):
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
            out.append(rec)
    return with_scores(out)


def hard_subset(all_rows: List[dict], dataset: str, setting_name: str) -> dict:
    sub = [
        r
        for r in all_rows
        if r["dataset"] == dataset
        and r["setting"] == setting_name
        and r.get("kmeans_cos_raw") == r.get("kmeans_cos_raw")
        and float(r["kmeans_cos_raw"]) < 0.55
        and int(r["n_patches"]) >= 40
    ]
    if not sub:
        return {"n": 0}
    return {
        "n": len(sub),
        "part0": _nanmean([r["kmeans_cos_raw"] for r in sub]),
        "part1": _nanmean([r["kmeans_cos_sm"] for r in sub]),
        "mix": _nanmean([r["kmeans_mix"] for r in sub]),
        "obj1": _nanmean([r["pair_cos_sm"] for r in sub]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--max-clips", type=int, default=50)
    ap.add_argument("--frames-per-clip", type=int, default=2)
    ap.add_argument("--min-patches", type=int, default=16)
    ap.add_argument("--out-dir", default="event_analysis/featcur_mix_sweep")
    args = ap.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print("settings:", [s["name"] for s in SETTINGS], flush=True)

    all_rows: List[dict] = []
    for ds in ("ytvis", "movi_c"):
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

    scored = agg_rows(all_rows)
    path = os.path.join(args.out_dir, "t02w3_summary.json")
    with open(path, "w") as f:
        json.dump(scored, f, indent=2)
    with open(os.path.join(args.out_dir, "t02w3_per_object.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    header = (
        f"{'dataset':8} {'setting':14} {'part':>6} {'spat':>6} {'marg':>6} "
        f"{'imix':>6} {'cmix':>6} {'score':>6} {'Ppart':>6} {'Pobj':>5} {'Pbg':>5}"
    )
    print("\n" + header)
    for r in scored:
        print(
            f"{r['dataset']:8} {r['setting']:14} {r['kmeans_mix']:6.3f} {r['spatial_mix']:6.3f} "
            f"{r['margin']:6.3f} {r['inst_mix']:6.3f} {r['contact_mix']:6.3f} {r['score']:6.3f} "
            f"{r['p_kmeans_other_part']:6.3f} {r['p_kmeans_other_obj']:5.3f} {r['p_kmeans_bg']:5.3f}"
        )

    print("\n===== YTVIS hard parts (raw cos<0.55, n>=40) =====")
    print(f"{'setting':14} {'n':>3} {'part0':>6} {'part1':>6} {'mix':>6} {'obj1':>6}")
    for st in SETTINGS:
        h = hard_subset(all_rows, "ytvis", st["name"])
        if h["n"] == 0:
            continue
        print(
            f"{st['name']:14} {h['n']:3d} {h['part0']:6.3f} {h['part1']:6.3f} "
            f"{h['mix']:6.3f} {h['obj1']:6.3f}"
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
