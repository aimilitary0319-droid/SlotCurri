#!/usr/bin/env python3
"""Eval-only temporal ablations on a trained v39 checkpoint.

Runs (no retraining):
  gated_single     reported setting: π gate on decoder + temporal mix, cycle=False
  gated_last       SlotCurri cyclic: backward sweep from last frame
  gated_evidence   EABI, anchor = argmax_t sum_s g * ownership-c
  ungated_single   predictor_ungated=True (decoder still sees π), cycle=False
  ungated_last     predictor_ungated=True + cyclic

Usage (inside the slotcurri container):
  python event_analysis/v39_temporal_eval.py --runs gated_single gated_last gated_evidence
  python event_analysis/v39_temporal_eval.py --runs ungated_single ungated_last
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from slotcurri import configuration, data, metrics, models


RUNS = {
    "gated_single": {"cycle": False, "ungated": False},
    "gated_last": {"cycle": True, "ungated": False},
    "gated_evidence": {"cycle": "evidence", "ungated": False},
    "ungated_single": {"cycle": False, "ungated": True},
    "ungated_last": {"cycle": True, "ungated": True},
    # Temporal-mix only (decoder keeps instantaneous π):
    #   π̃_t = m π_t + (1-m) π̃_{t-1}, then max(π̃_t, hold * π̃_{t-1}).
    "ema_05": {"cycle": False, "ungated": False, "ema": 0.5},
    "ema_03": {"cycle": False, "ungated": False, "ema": 0.3},
    "ema_02": {"cycle": False, "ungated": False, "ema": 0.2},
    "hold_09": {"cycle": False, "ungated": False, "hold": 0.9},
    "ema_05_hold_09": {"cycle": False, "ungated": False, "ema": 0.5, "hold": 0.9},
    # Leaky max on π before decoder AND temporal (hits masks too).
    "hyst70_single": {"cycle": False, "ungated": False, "hyst": 0.70},
    "hyst85_single": {"cycle": False, "ungated": False, "hyst": 0.85},
    "hyst95_single": {"cycle": False, "ungated": False, "hyst": 0.95},
    "hyst85_evidence": {"cycle": "evidence", "ungated": False, "hyst": 0.85},
    "hyst95_evidence": {"cycle": "evidence", "ungated": False, "hyst": 0.95},
}


class _FakeTrainer:
    def __init__(self, step: int):
        self.global_step = step


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _metric_to_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


@torch.no_grad()
def run_one(
    model,
    loader,
    device: torch.device,
    cycle,
    ungated: bool,
    max_clips: int,
    metric_cfgs: Dict[str, Any],
    hyst: float = 0.0,
    ema: float = 1.0,
    hold: float = 0.0,
) -> Dict[str, Any]:
    model.amc_predictor_ungated = bool(ungated)
    model.amc_gate_hysteresis = float(hyst)
    model.amc_state_gate_ema = float(ema)
    model.amc_state_gate_hold = float(hold)
    built = {name: metrics.build(cfg).to(device) for name, cfg in metric_cfgs.items()}
    for m in built.values():
        m.reset()

    per_clip_ari: List[float] = []
    per_clip_iari: List[float] = []
    n_clips = 0
    clip_ari = metrics.build(metric_cfgs["ari"]).to(device)
    clip_iari = (
        metrics.build(metric_cfgs["image_ari"]).to(device)
        if "image_ari" in metric_cfgs
        else None
    )

    for batch in loader:
        if "batch_padding_mask" in batch:
            batch = model._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                continue
        batch = _to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model.forward(batch, train=False, cycle=cycle)
            aux = model.aux_forward(batch, outputs)
        for m in built.values():
            m.update(**batch, **outputs, **aux)

        bsz = int(outputs["batch_size"])
        pred = aux.get("decoder_masks_hard")
        true = batch.get("segmentations")
        for i in range(bsz):
            clip_kw = {
                "decoder_masks_hard": pred[i : i + 1],
                "segmentations": true[i : i + 1],
            }
            clip_ari.reset()
            clip_ari.update(**clip_kw)
            per_clip_ari.append(_metric_to_float(clip_ari.compute()))
            if clip_iari is not None:
                clip_iari.reset()
                clip_iari.update(**clip_kw)
                per_clip_iari.append(_metric_to_float(clip_iari.compute()))

        n_clips += bsz
        if n_clips % 16 == 0 or n_clips >= max_clips:
            print(f"  {n_clips} clips ...", flush=True)
        if n_clips >= max_clips:
            break

    agg = {name: _metric_to_float(m.compute()) for name, m in built.items()}
    return {
        "agg": agg,
        "n_clips": n_clips,
        "per_clip_ari": per_clip_ari,
        "per_clip_image_ari": per_clip_iari,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/slotcurri/movi_c_attnmass_v39.yaml")
    ap.add_argument(
        "--ckpt",
        default="logs/_movi_c_attnmass_v39/checkpoints/slotcurri_step=step=100000-v1.ckpt",
    )
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--max-clips", type=int, default=250)
    ap.add_argument("--val-batch-size", type=int, default=4)
    ap.add_argument("--runs", nargs="+", default=["gated_single", "gated_last"])
    ap.add_argument("--out", default="logs/v39_temporal_eval")
    ap.add_argument("--step", type=int, default=100000)
    args = ap.parse_args()

    unknown = [r for r in args.runs if r not in RUNS]
    if unknown:
        raise SystemExit(f"unknown runs {unknown}; choose from {sorted(RUNS)}")

    config = configuration.load_config(args.config)
    config.model.visualize = False
    config.dataset.val_batch_size = int(args.val_batch_size)
    config.dataset.num_val_workers = 0

    built = {n: metrics.build(c) for n, c in config.val_metrics.items()}
    model = models.build(config.model, config.optimizer, None, built)
    model.load_weights_from_checkpoint(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    model._trainer = _FakeTrainer(args.step)

    dataset = data.build(config.dataset, data_dir=args.data_dir)
    dataset.setup("validate")

    print(
        f"ckpt={args.ckpt}\n"
        f"device={device}  max_clips={args.max_clips}  "
        f"val_batch_size={config.dataset.val_batch_size}",
        flush=True,
    )
    print(f"runs={args.runs}", flush=True)

    all_runs: Dict[str, Dict[str, Any]] = {}
    for name in args.runs:
        spec = RUNS[name]
        print(
            f"\n=== {name}  cycle={spec['cycle']!r}  ungated={spec['ungated']}  "
            f"ema={spec.get('ema', 1.0)}  hold={spec.get('hold', 0.0)}  "
            f"hyst={spec.get('hyst', 0.0)} ===",
            flush=True,
        )
        # new loader each run so every condition sees the same clip order
        loader = dataset.val_dataloader()
        all_runs[name] = run_one(
            model,
            loader,
            device,
            spec["cycle"],
            spec["ungated"],
            args.max_clips,
            config.val_metrics,
            hyst=spec.get("hyst", 0.0),
            ema=spec.get("ema", 1.0),
            hold=spec.get("hold", 0.0),
        )
        agg = all_runs[name]["agg"]
        n = all_runs[name]["n_clips"]
        print(
            f"[{name}] n={n}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in agg.items()),
            flush=True,
        )

    names = list(config.val_metrics.keys())
    header = f"{'run':>16s} | " + " | ".join(f"{n:>10s}" for n in names)
    print("\n" + header)
    print("-" * len(header))
    for name in args.runs:
        row = f"{name:>16s} | " + " | ".join(
            f"{all_runs[name]['agg'][n]:10.4f}" for n in names
        )
        print(row)

    if "gated_single" in all_runs:
        base = all_runs["gated_single"]["agg"]
        print("\ndeltas vs gated_single (+ = better than reported v39):")
        for name in args.runs:
            if name == "gated_single":
                continue
            parts = []
            for k in names:
                parts.append(f"{k}={all_runs[name]['agg'][k] - base[k]:+.4f}")
            print(f"  {name:>16s}  " + "  ".join(parts))

    # paired per-clip ARI
    if len(args.runs) >= 2:
        print("\nper-clip video ARI:")
        ref = args.runs[0]
        a_ref = np.asarray(all_runs[ref]["per_clip_ari"])
        for name in args.runs[1:]:
            a = np.asarray(all_runs[name]["per_clip_ari"])
            n = min(len(a_ref), len(a))
            d = a[:n] - a_ref[:n]
            print(
                f"  {name} - {ref}: mean {d.mean():+.4f}  "
                f"median {np.median(d):+.4f}  "
                f"improved {(d > 0).mean():.1%}  worsened {(d < 0).mean():.1%}"
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "ckpt": args.ckpt,
        "config": args.config,
        "max_clips": args.max_clips,
        "runs": {
            name: {
                "cycle": RUNS[name]["cycle"]
                if not isinstance(RUNS[name]["cycle"], bool)
                else bool(RUNS[name]["cycle"]),
                "ungated": RUNS[name]["ungated"],
                "ema": RUNS[name].get("ema", 1.0),
                "hold": RUNS[name].get("hold", 0.0),
                "hyst": RUNS[name].get("hyst", 0.0),
                "n_clips": all_runs[name]["n_clips"],
                "agg": all_runs[name]["agg"],
            }
            for name in args.runs
        },
    }
    with open(f"{args.out}.json", "w") as f:
        json.dump(payload, f, indent=2)
    np.savez_compressed(
        f"{args.out}.npz",
        **{
            f"{name}_ari": np.asarray(all_runs[name]["per_clip_ari"])
            for name in args.runs
        },
        **{
            f"{name}_image_ari": np.asarray(all_runs[name]["per_clip_image_ari"])
            for name in args.runs
        },
    )
    print(f"\nsaved {args.out}.json  {args.out}.npz", flush=True)


if __name__ == "__main__":
    main()
