"""Training status across runs: progress, validation curve, and gate statistics.

Reads the Lightning CSV logger output that every run writes to
logs/_<experiment_name>/metrics/slotcurri/metrics.csv. Validation rows and training rows are
logged separately (each row has only the columns for its own phase), so the two are collected
independently rather than joined.

Usage:
  python event_analysis/run_status.py [experiment_name ...]
"""

import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

LOG_ROOT = "logs"
VAL_KEYS = ["val/ari", "val/image_ari", "val/mbo", "val/image_mbo", "val/loss"]
GATE_KEYS = ["train/active_slots", "train/gate_p", "train/gate_tau", "train/gate_max",
             "train/gate_top2", "train/gate_entropy", "train/gate_n_half", "train/loss"]


def num(row: Dict[str, str], key: str) -> Optional[float]:
    v = row.get(key, "")
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read(name: str) -> Tuple[List[Dict], List[Dict]]:
    path = os.path.join(LOG_ROOT, f"_{name}", "metrics", "slotcurri", "metrics.csv")
    if not os.path.exists(path):
        return [], []
    val, train = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            step = num(row, "step")
            if step is None:
                continue
            if num(row, "val/ari") is not None:
                val.append({"step": int(step), **{k: num(row, k) for k in VAL_KEYS}})
            elif num(row, "train/loss") is not None:
                train.append({"step": int(step), **{k: num(row, k) for k in GATE_KEYS}})
    return val, train


def fmt(v: Optional[float], w: int = 8, p: int = 4) -> str:
    return f"{v:>{w}.{p}f}" if v is not None else " " * (w - 1) + "-"


def report(name: str) -> None:
    val, train = read(name)
    if not train and not val:
        print(f"\n{name}: no metrics found")
        return
    last_step = max([r["step"] for r in train] + [r["step"] for r in val])
    print(f"\n{'=' * 92}\n{name}   last logged step {last_step:,}\n{'=' * 92}")

    if val:
        print("  validation")
        print(f"    {'step':>8}{'ari':>9}{'image_ari':>11}{'mbo':>9}{'image_mbo':>11}{'loss':>9}")
        # every validation point, plus which one is best on each metric
        for r in val:
            print(f"    {r['step']:>8,}{fmt(r['val/ari'], 9)}{fmt(r['val/image_ari'], 11)}"
                  f"{fmt(r['val/mbo'], 9)}{fmt(r['val/image_mbo'], 11)}{fmt(r['val/loss'], 9)}")
        print("    best:", ", ".join(
            f"{k.split('/')[1]} {max(r[k] for r in val if r[k] is not None):.4f}"
            f"@{max((r for r in val if r[k] is not None), key=lambda r: r[k])['step']:,}"
            for k in VAL_KEYS if k != "val/loss" and any(r[k] is not None for r in val)))

    if train:
        print("  gate statistics (train, sampled)")
        print(f"    {'step':>8}{'active':>8}{'p':>8}{'tau':>8}{'g_max':>8}{'g_top2':>8}"
              f"{'H(g)':>8}{'n_half':>8}{'loss':>8}")
        idx = sorted({0, len(train) // 4, len(train) // 2, 3 * len(train) // 4, len(train) - 1})
        for i in idx:
            r = train[i]
            print(f"    {r['step']:>8,}{fmt(r['train/active_slots'], 8, 3)}"
                  f"{fmt(r['train/gate_p'], 8, 4)}{fmt(r['train/gate_tau'], 8, 4)}"
                  f"{fmt(r['train/gate_max'], 8, 3)}{fmt(r['train/gate_top2'], 8, 3)}"
                  f"{fmt(r['train/gate_entropy'], 8, 3)}{fmt(r['train/gate_n_half'], 8, 2)}"
                  f"{fmt(r['train/loss'], 8, 3)}")


def main() -> int:
    names = sys.argv[1:] or [
        d[1:] for d in sorted(os.listdir(LOG_ROOT))
        if d.startswith("_ytvis") and os.path.isdir(os.path.join(LOG_ROOT, d))
    ]
    for name in names:
        report(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
