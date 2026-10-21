"""One middle frame per training video, then the median start mix.

s(x) is the closed-form root of q(s) = T(s) / 2 along the leading axis of X^rel.
s0 is the median over training videos. Deterministic center-crop at the
dataset's val input size (same resolution as training).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import webdataset as wds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_s_closed import closed_form  # noqa: E402
from featcur_strength_probe import extract_tokens  # noqa: E402
from slotcurri import configuration
from slotcurri.data import transforms
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor

DATASETS = {
    "ytvis": "configs/slotcurri/ytvis2021_attnmass_v39lam1gu.yaml",
    "movi_c": "configs/slotcurri/movi_c_attnmass_v39lam1gu.yaml",
    "movi_e": "configs/slotcurri/movi_e_attnmass_v39lam1gu.yaml",
}


def shard_paths(pattern: str, data_dir: str) -> list:
    # pattern like "movi_c/movi_c-train-{000000..000304}.tar"
    if "{" in pattern:
        import braceexpand
        rels = list(braceexpand.braceexpand(pattern))
    else:
        rels = [pattern]
    return [os.path.join(data_dir, rel) for rel in rels]


def middle_frame(video: np.ndarray) -> np.ndarray:
    """(T,H,W,C) or (T,C,H,W) -> (1,H,W,C) uint8."""
    if video.ndim != 4:
        raise ValueError(f"expected 4D video, got {video.shape}")
    t = video.shape[0]
    frame = video[t // 2]
    if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    return frame[None]


def summarize(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "frac_zero": float(np.mean(values <= 1e-8)),
    }


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--data-dir", default="/workspace/dataset")
    p.add_argument("--out-dir", default="event_analysis/featcur_s0")
    p.add_argument("--batch-size", type=int, default=0)
    args = p.parse_args()

    cfg = configuration.load_config(DATASETS[args.dataset])
    tf = transforms.build(cfg.dataset.val_pipeline.transforms)["video"]
    bb = cfg.model.encoder.backbone
    batch_size = args.batch_size or (8 if args.dataset == "ytvis" else 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = TimmExtractor(
        model=bb.model,
        pretrained=True,
        frozen=True,
        features=bb.features,
        model_kwargs=dict(bb.get("model_kwargs") or {}) or None,
    ).to(device).eval()
    level = NcutRelationalLeveling(chunk_size=4, barrier=False).to(device)
    paths = shard_paths(cfg.dataset.train_shards, args.data_dir)
    print(f"{args.dataset}: {len(paths)} shards, batch {batch_size}, device {device}", flush=True)

    ds = wds.WebDataset(paths, shardshuffle=False, nodesplitter=lambda urls: urls)
    ds = ds.map(lambda sample: {k: v for k, v in sample.items() if "video" in k or k.startswith("__")})
    ds = ds.decode("rgb")

    os.makedirs(args.out_dir, exist_ok=True)
    scores = []
    batch = []
    n_seen = 0

    def flush():
        nonlocal batch
        if not batch:
            return
        frames = torch.stack(batch, 0).to(device, non_blocking=True)
        tokens = extract_tokens(backbone, frames, bb.features).float()
        rel = level(tokens, 0.0)
        for i in range(tokens.shape[0]):
            scores.append(closed_form(tokens[i], rel[i]))
        batch = []

    for sample in ds:
        video = next(sample[k] for k in sample if "video" in k and not k.startswith("__"))
        clip = middle_frame(np.asarray(video))
        transformed = tf(clip)
        if transformed.ndim == 4:
            transformed = transformed[0]
        batch.append(transformed.cpu())
        n_seen += 1
        if len(batch) >= batch_size:
            flush()
            if n_seen % (batch_size * 20) == 0:
                arr = np.asarray(scores, dtype=np.float64)
                print(
                    f"  {args.dataset} n={arr.size} median={np.median(arr):.4f}",
                    flush=True,
                )
    flush()
    arr = np.asarray(scores, dtype=np.float64)
    summary = {"dataset": args.dataset, **summarize(arr)}
    out_path = os.path.join(args.out_dir, f"dataset_s0_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    np.save(os.path.join(args.out_dir, f"dataset_s0_{args.dataset}.npy"), arr)
    print(
        f"{args.dataset}: n={summary['n']} median={summary['median']:.4f} "
        f"mean={summary['mean']:.4f} p25={summary['p25']:.4f} p75={summary['p75']:.4f} "
        f"frac0={summary['frac_zero']:.3f}",
        flush=True,
    )
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
