"""Dataset start mix for the feature curriculum.

One random frame per training video. s(x) is the closed-form root of
q(s) = T(s) / 2 along the leading axis of centered X^rel. s0 is the median.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np
import torch
import webdataset as wds

from slotcurri.data import transforms
from slotcurri.data.utils import get_data_root_dir
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor


def closed_form_s(raw: torch.Tensor, rel: torch.Tensor) -> float:
    """Smallest s in [0, 1] with energy along v at most half of total energy.

    v is the leading right singular vector of centered `rel`. q and T are
    quadratic in s, so the crossing is a quadratic root. s = 0 when the
    leveled features are already at or under half.
    """
    raw_c = raw.float() - raw.float().mean(0, keepdim=True)
    rel_c = rel.float() - rel.float().mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(rel_c, full_matrices=False)
    v = vh[0]
    a = rel_c @ v
    b = raw_c @ v
    qa, qb, qc = (a * a).sum(), (b * b).sum(), (a * b).sum()
    ta = (rel_c * rel_c).sum()
    tb = (raw_c * raw_c).sum()
    tc = (rel_c * raw_c).sum()
    # 2 q(s) - T(s) = A s^2 + B s + C
    A = 2 * (qa + qb - 2 * qc) - (ta + tb - 2 * tc)
    B = 2 * (-2 * qa + 2 * qc) - (-2 * ta + 2 * tc)
    C = 2 * qa - ta
    if float(C) <= 0:
        return 0.0
    disc = B * B - 4 * A * C
    roots = []
    if float(A.abs()) < 1e-12:
        if float(B.abs()) > 1e-12:
            roots.append(float((-C / B).item()))
    elif float(disc) >= 0:
        sd = torch.sqrt(disc)
        roots.append(float(((-B + sd) / (2 * A)).item()))
        roots.append(float(((-B - sd) / (2 * A)).item()))
    hits = [r for r in roots if 0.0 <= r <= 1.0]
    return min(hits) if hits else 1.0


def _shard_paths(pattern: str, data_dir: str) -> list:
    if "{" in pattern:
        import braceexpand

        rels = list(braceexpand.braceexpand(pattern))
    else:
        rels = [pattern]
    return [os.path.join(data_dir, rel) for rel in rels]


def _random_frame(video: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if video.ndim != 4:
        raise ValueError(f"expected a 4D video, got {video.shape}")
    index = int(rng.integers(0, video.shape[0]))
    frame = video[index]
    if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    return frame[None]


def _batch_size(model_name: str, input_size: int) -> int:
    name = str(model_name)
    if "vit_base" in name and int(input_size) >= 500:
        return 8
    if "vit_base" in name:
        return 16
    return 32


@torch.no_grad()
def estimate_mix_start(config, data_dir: Optional[str] = None, seed: int = 0) -> float:
    """Median s over one random frame from each training video."""
    data_dir = data_dir or get_data_root_dir(error_on_missing_path=True)
    tf = transforms.build(config.dataset.val_pipeline.transforms)["video"]
    bb = config.model.encoder.backbone
    input_size = int(config.dataset.val_pipeline.transforms.input_size)
    batch_size = _batch_size(bb.model, input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = (
        TimmExtractor(
            model=bb.model,
            pretrained=True,
            frozen=True,
            features=bb.features,
            model_kwargs=dict(bb.get("model_kwargs") or {}) or None,
        )
        .to(device)
        .eval()
    )
    level = NcutRelationalLeveling(chunk_size=4, barrier=False).to(device)
    paths = _shard_paths(str(config.dataset.train_shards), data_dir)
    rng = np.random.default_rng(int(seed))
    print(
        f"Estimating feature-curriculum s0 on {len(paths)} train shards "
        f"(one random frame per video, seed {int(seed)})",
        flush=True,
    )
    ds = wds.WebDataset(paths, shardshuffle=False, nodesplitter=lambda urls: urls)
    ds = ds.map(
        lambda sample: {
            k: v for k, v in sample.items() if "video" in k or str(k).startswith("__")
        }
    )
    ds = ds.decode("rgb")

    scores = []
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
        tokens = tokens.float()
        rel = level(tokens, 0.0)
        for i in range(tokens.shape[0]):
            scores.append(closed_form_s(tokens[i], rel[i]))
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
                    f"  s0 scan n={len(scores)} "
                    f"median={float(np.median(scores)):.4f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
    flush()
    if not scores:
        raise RuntimeError("feature_curriculum.mix_start='auto' saw no training videos")
    s0 = float(np.median(np.asarray(scores, dtype=np.float64)))
    print(
        f"feature-curriculum s0 = {s0:.6f} "
        f"(median of {len(scores)} videos, {time.time() - t0:.0f}s)",
        flush=True,
    )
    return s0


def load_or_estimate_mix_start(
    config,
    log_path,
    data_dir: Optional[str],
    seed: int,
) -> float:
    """Reuse a saved s0 in this log dir, otherwise estimate and save it."""
    path = None if log_path is None else log_path / "featcur_mix_start.json"
    if path is not None and path.is_file():
        with open(path) as f:
            saved = json.load(f)
        s0 = float(saved["mix_start"])
        print(f"Using saved feature-curriculum s0 = {s0:.6f} from {path}", flush=True)
        return s0
    s0 = estimate_mix_start(config, data_dir=data_dir, seed=seed)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"mix_start": s0, "seed": int(seed)}, f, indent=2)
        print(f"Wrote {path}", flush=True)
    return s0
