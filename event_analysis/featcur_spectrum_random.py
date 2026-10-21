"""Leading spectrum of X^rel on one random frame per training video.

Reports the share of λ1 and λ2, λ1/λ2, and the participation ratio.
Seed 0 makes the frame draw reproducible. Center crop matches val resolution.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import webdataset as wds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slotcurri import configuration
from slotcurri.data import transforms
from slotcurri.featcur_start import _batch_size, _random_frame, _shard_paths
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor

DATASETS = {
    "ytvis": "configs/slotcurri/ytvis2021_attnmass_v39lam1gu.yaml",
    "movi_c": "configs/slotcurri/movi_c_attnmass_v39lam1gu.yaml",
    "movi_e": "configs/slotcurri/movi_e_attnmass_v39lam1gu.yaml",
}


def spectrum(rel: torch.Tensor) -> dict:
    xc = rel.float() - rel.float().mean(0, keepdim=True)
    energy = torch.linalg.svdvals(xc).square()
    tot = energy.sum().clamp_min(1e-12)
    p = energy / tot
    pr = float((tot.square() / energy.square().sum().clamp_min(1e-12)).item())
    l1 = float(p[0].item())
    l2 = float(p[1].item()) if p.numel() > 1 else 0.0
    return {
        "l1": l1,
        "l2": l2,
        "l1_over_l2": l1 / l2 if l2 > 1e-8 else float("inf"),
        "pr": pr,
    }


def pack(rows, key):
    a = np.asarray([r[key] for r in rows], dtype=np.float64)
    a = a[np.isfinite(a)]
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p25": float(np.quantile(a, 0.25)),
        "p75": float(np.quantile(a, 0.75)),
    }


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--data-dir", default="/workspace/dataset")
    p.add_argument("--out-dir", default="event_analysis/featcur_s0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--mem-fraction", type=float, default=0.30)
    args = p.parse_args()

    if torch.cuda.is_available() and args.mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)

    cfg = configuration.load_config(DATASETS[args.dataset])
    tf = transforms.build(cfg.dataset.val_pipeline.transforms)["video"]
    bb = cfg.model.encoder.backbone
    input_size = int(cfg.dataset.val_pipeline.transforms.input_size)
    batch_size = args.batch_size or _batch_size(bb.model, input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = TimmExtractor(
        model=bb.model,
        pretrained=True,
        frozen=True,
        features=bb.features,
        model_kwargs=dict(bb.get("model_kwargs") or {}) or None,
    ).to(device).eval()
    level = NcutRelationalLeveling(chunk_size=4, barrier=False).to(device)
    paths = _shard_paths(str(cfg.dataset.train_shards), args.data_dir)
    rng = np.random.default_rng(int(args.seed))
    print(
        f"{args.dataset}: {len(paths)} shards, batch {batch_size}, seed {args.seed}",
        flush=True,
    )
    ds = wds.WebDataset(paths, shardshuffle=False, nodesplitter=lambda urls: urls)
    ds = ds.map(
        lambda sample: {
            k: v for k, v in sample.items() if "video" in k or str(k).startswith("__")
        }
    )
    ds = ds.decode("rgb")

    rows = []
    batch = []
    n_seen = 0
    t0 = time.time()

    def flush():
        nonlocal batch
        if not batch:
            return
        frames = torch.stack(batch, 0).to(device)
        tokens = backbone(frames)
        if isinstance(tokens, dict):
            tokens = tokens[bb.features]
        rel = level(tokens.float(), 0.0)
        for i in range(rel.shape[0]):
            rows.append(spectrum(rel[i]))
        batch = []

    for sample in ds:
        video = next(
            sample[k] for k in sample if "video" in k and not str(k).startswith("__")
        )
        clip = _random_frame(np.asarray(video), rng)
        transformed = tf(clip)
        if transformed.ndim == 4:
            transformed = transformed[0]
        batch.append(transformed)
        n_seen += 1
        if len(batch) >= batch_size:
            flush()
            if n_seen % (batch_size * 20) == 0:
                print(
                    f"  {args.dataset} n={len(rows)} "
                    f"l1={np.median([r['l1'] for r in rows]):.3f} "
                    f"l2={np.median([r['l2'] for r in rows]):.3f} "
                    f"pr={np.median([r['pr'] for r in rows]):.2f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
    flush()
    summary = {
        "dataset": args.dataset,
        "seed": int(args.seed),
        "n": len(rows),
        "l1": pack(rows, "l1"),
        "l2": pack(rows, "l2"),
        "l1_over_l2": pack(rows, "l1_over_l2"),
        "pr": pack(rows, "pr"),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"spectrum_random_{args.dataset}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(
        os.path.join(args.out_dir, f"spectrum_random_{args.dataset}.npz"),
        l1=np.asarray([r["l1"] for r in rows]),
        l2=np.asarray([r["l2"] for r in rows]),
        pr=np.asarray([r["pr"] for r in rows]),
    )
    print(
        f"{args.dataset}: n={summary['n']} "
        f"l1={summary['l1']['median']:.3f} l2={summary['l2']['median']:.3f} "
        f"l1/l2={summary['l1_over_l2']['median']:.2f} pr={summary['pr']['median']:.2f}",
        flush=True,
    )
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
