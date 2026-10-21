"""Is the single-slot takeover an Adam drift along a loss-flat direction?

Real model at init (seed 0), FC mix s=0, train-mode forward and the real loss.
For M mini-batches of 4-frame clips the grads of the Slot Attention path
(initializer, encoder output_transform, corrector) are collected at the same θ.

  margin M(θ): frame-0, iteration-1 logit of the batch-mode leader minus the best
               rival, averaged over clips and patches of a fixed probe batch.
  adam dir   : u = -mean(g) / sqrt(mean(g²))  (bias-corrected Adam with β→1)
  sgd dir    : -mean(g), rescaled to the same L2 norm as u
  ΔM         : M(θ + η·dir) - M(θ)
  sign_cons  : mean over coordinates of |mean_b sign(g_b)|
  cos(g,∇M)  : how much the loss itself cares about the margin

Data × head crossed; gate on (as trained) or replaced by ones (SlotContrast).
DINO tokens are computed once per dataset and replayed through a stub backbone.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import webdataset as wds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slotcurri import configuration, models
from slotcurri.data import transforms
from slotcurri.featcur_start import _shard_paths

CONFIGS = {
    "movi_e": "configs/slotcurri/movi_e_attnmass_v39lam1gu.yaml",
    "ytvis": "configs/slotcurri/ytvis2021_attnmass_v39lam1gu.yaml",
}
HEADS = {"k15d128": (15, 128), "k7d64": (7, 64)}
SA_PREFIXES = ("initializer.", "encoder.module.output_transform.", "processor.module.corrector.")


def load_clips(cfg, data_dir, n_clips, rng, n_frames=4):
    tf = transforms.build(cfg.dataset.val_pipeline.transforms)["video"]
    paths = list(_shard_paths(str(cfg.dataset.train_shards), data_dir))
    paths = [paths[i] for i in rng.permutation(len(paths))]
    clips = []
    for path in paths:
        ds = wds.WebDataset([path], shardshuffle=False, nodesplitter=lambda u: u)
        ds = ds.map(
            lambda s: {k: v for k, v in s.items() if "video" in k or str(k).startswith("__")}
        ).decode("rgb")
        for sample in ds:
            video = np.asarray(
                next(v for k, v in sample.items() if "video" in k and not str(k).startswith("__"))
            )
            if video.shape[0] < n_frames:
                continue
            start = int(rng.integers(0, video.shape[0] - n_frames + 1))
            clips.append(tf(video[start : start + n_frames]))
            break
        if len(clips) >= n_clips:
            break
    return clips


class CachedBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.current = None

    def forward(self, images):
        return self.current


def sa_params(model):
    return [(n, p) for n, p in model.named_parameters() if n.startswith(SA_PREFIXES) and p.requires_grad]


@torch.no_grad()
def margin(model, video, tokens, leader=None):
    model.encoder.module.backbone.current = tokens
    enc = model.encoder(video)
    key = enc.get("features_key", enc["features"])[:, 0]
    corr = model.processor.module.corrector
    slots0 = model.initializer(video.shape[0])
    q = corr.to_q(corr.norm_slots(slots0))
    k = corr.to_k(corr.norm_features(key))
    logit = torch.einsum("bsd,bnd->bsn", q, k) * corr.scale / corr.logit_div
    if leader is None:
        leader = int(torch.bincount(logit.mean(-1).argmax(-1), minlength=logit.shape[1]).argmax())
    lead = logit[:, leader]
    rival = torch.cat([logit[:, :leader], logit[:, leader + 1 :]], 1).amax(1)
    return float((lead - rival).mean()), leader


def probe(dataset, head, gated, clips_tok, args):
    k_slots, d = HEADS[head]
    cfg = configuration.load_config(
        CONFIGS[dataset], [f"globals.NUM_SLOTS={k_slots}", f"globals.SLOT_DIM={d}"]
    )
    torch.manual_seed(0)
    model = models.build(cfg.model, cfg.optimizer)
    model._trainer = SimpleNamespace(global_step=0)
    model._gate_tau = lambda train: None
    if not gated:
        model.slotcontrast_steps = 10**9
    inner = model._frame_encoder()
    inner.backbone = CachedBackbone()
    if getattr(inner, "feature_ncut", None) is not None:
        inner.feature_ncut_mix = 0.0
    model.train()

    params = sa_params(model)
    grads = []
    for video, tok in clips_tok:
        model.zero_grad(set_to_none=True)
        inner.backbone.current = tok
        out = model({"video": video}, train=True)
        loss, _ = model.compute_loss(out)
        loss.backward()
        grads.append(torch.cat([p.grad.detach().flatten() for _, p in params]))
    g = torch.stack(grads)
    gbar = g.mean(0)
    rms = g.square().mean(0).sqrt().clamp_min(1e-20)
    adam = -gbar / rms
    sgd = -gbar / gbar.norm().clamp_min(1e-20) * adam.norm()
    sign_cons = float(g.sign().mean(0).abs().mean())

    video0, tok0 = clips_tok[0]
    theta0 = [p.detach().clone() for _, p in params]
    m0, leader = margin(model, video0, tok0)

    def shifted(direction, eta):
        off = 0
        with torch.no_grad():
            for (_, p), base in zip(params, theta0):
                n = p.numel()
                p.copy_(base + eta * direction[off : off + n].view_as(p))
                off += n
        m, _ = margin(model, video0, tok0, leader)
        with torch.no_grad():
            for (_, p), base in zip(params, theta0):
                p.copy_(base)
        return m - m0

    # ∇M by autograd on the same probe.
    model.zero_grad(set_to_none=True)
    inner.backbone.current = tok0
    with torch.enable_grad():
        enc = model.encoder(video0)
        key = enc.get("features_key", enc["features"])[:, 0]
        corr = model.processor.module.corrector
        slots0 = model.initializer(video0.shape[0])
        q = corr.to_q(corr.norm_slots(slots0))
        kk = corr.to_k(corr.norm_features(key))
        logit = torch.einsum("bsd,bnd->bsn", q, kk) * corr.scale / corr.logit_div
        lead = logit[:, leader]
        rival = torch.cat([logit[:, :leader], logit[:, leader + 1 :]], 1).amax(1)
        (lead - rival).mean().backward()
    grad_m = torch.cat(
        [(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for _, p in params]
    )

    def cos(a, b):
        return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-20))

    row = {
        "margin0": m0,
        "sign_consistency": sign_cons,
        "adam_dM": shifted(adam, args.eta),
        "sgd_dM": shifted(sgd, args.eta),
        "cos_loss_grad_vs_gradM": cos(-gbar, grad_m),
        "cos_adam_vs_gradM": cos(adam, grad_m),
        "batch_grad_pairwise_cos": float(
            torch.nn.functional.normalize(g, dim=1).matmul(
                torch.nn.functional.normalize(g, dim=1).T
            )[~torch.eye(len(g), dtype=torch.bool)].mean()
        ),
    }
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/workspace/dataset")
    p.add_argument("--out", default="event_analysis/adam_drift/adam_drift.json")
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--clips-per-batch", type=int, default=4)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=48)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(0)

    from slotcurri.modules.encoders import TimmExtractor

    results = {}
    for dataset in CONFIGS:
        cfg = configuration.load_config(CONFIGS[dataset])
        bb = cfg.model.encoder.backbone
        backbone = TimmExtractor(
            model=bb.model, pretrained=True, frozen=True, features=bb.features,
            model_kwargs=dict(bb.get("model_kwargs") or {}) or None,
        ).eval()
        t0 = time.time()
        clips = load_clips(cfg, args.data_dir, args.n_batches * args.clips_per_batch, rng)
        clips_tok = []
        with torch.no_grad():
            for b0 in range(0, len(clips), args.clips_per_batch):
                video = torch.stack(clips[b0 : b0 + args.clips_per_batch], 0)
                tok = backbone(video.flatten(0, 1))
                clips_tok.append((video, tok))
        print(f"{dataset}: {len(clips)} clips tokenized ({time.time() - t0:.0f}s)", flush=True)
        for head in HEADS:
            for gated in (True, False):
                t1 = time.time()
                row = probe(dataset, head, gated, clips_tok, args)
                key = f"{dataset}|{head}|{'gated' if gated else 'ungated'}"
                results[key] = row
                print(
                    f"  {key:28s} sign_cons {row['sign_consistency']:.3f}"
                    f"  batch-cos {row['batch_grad_pairwise_cos']:.3f}"
                    f"  ΔM adam {row['adam_dM']:+.4f} sgd {row['sgd_dM']:+.4f}"
                    f"  cos(-g,∇M) {row['cos_loss_grad_vs_gradM']:+.4f}"
                    f"  cos(adam,∇M) {row['cos_adam_vs_gradM']:+.4f}"
                    f"  M0 {row['margin0']:.3f} ({time.time() - t1:.0f}s)",
                    flush=True,
                )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
