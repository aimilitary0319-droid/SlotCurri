"""Within-frame and cross-video LN+L2 cosine for PX, xvid, xframe, and
X^rel = PX - m_f + m_v.

m_f is the frame mean of PX. m_v is the mean of m_f over T frames of the same
video. Several frames per video are required; with one frame the new term is PX.

Same cosine as featcur_xvid_probe: within-frame is the mean pairwise cosine of
LN+L2 tokens. Cross-video averages frame-mean dots only across different videos.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/workspace/SlotCurri/event_analysis")
sys.path.insert(0, "/workspace/SlotCurri")

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from featcur_xvid_probe import xframe_rescale, _row_level
from global_winner_probe import CONFIGS
from slotcurri import configuration
from slotcurri.data import transforms
from slotcurri.featcur_start import _shard_paths
from slotcurri.modules.encoders import TimmExtractor


def _hwc(frame: np.ndarray) -> np.ndarray:
    if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    return frame


def load_clips(cfg, data_dir, n_videos, n_frames, per_shard, rng):
    tf = transforms.build(cfg.dataset.val_pipeline.transforms)["video"]
    paths = list(_shard_paths(str(cfg.dataset.train_shards), data_dir))
    paths = [paths[i] for i in rng.permutation(len(paths))]
    clips = []
    for path in paths:
        ds = wds.WebDataset([path], shardshuffle=False, nodesplitter=lambda u: u)
        ds = ds.map(
            lambda s: {k: v for k, v in s.items() if "video" in k or str(k).startswith("__")}
        ).decode("rgb")
        taken = 0
        for sample in ds:
            video = next(v for k, v in sample.items() if "video" in k and not str(k).startswith("__"))
            video = np.asarray(video)
            if video.ndim != 4 or video.shape[0] < n_frames:
                continue
            idx = np.linspace(0, video.shape[0] - 1, n_frames).astype(int)
            frames = []
            for i in idx:
                x = tf(_hwc(video[i])[None])
                frames.append(x[0] if x.ndim == 4 else x)
            clips.append(torch.stack(frames, 0))
            taken += 1
            if taken >= per_shard or len(clips) >= n_videos:
                break
        if len(clips) >= n_videos:
            break
    return clips


def dense_px(x: torch.Tensor) -> torch.Tensor:
    z = F.normalize(x, dim=-1)
    return _row_level(torch.bmm(z, z.transpose(1, 2)).clamp_min(0.0), x)


def xframe_video(x: torch.Tensor, rel: torch.Tensor, n_frames: int) -> torch.Tensor:
    """xframe with the per-video mean of raw X in place of the batch mean."""
    b = x.shape[0] // n_frames
    n, d = x.shape[1], x.shape[2]
    xx = x.view(b, n_frames, n, d)
    rr = rel.view(b, n_frames, n, d)
    m_v = xx.mean(dim=(1, 2), keepdim=True)
    m_f = rr.mean(dim=2, keepdim=True)
    resid = rr - m_f
    num = (xx - m_v).square().sum(-1).mean() - resid.square().sum(-1).mean()
    den = (m_f - m_v).square().sum(-1).mean().clamp_min(1e-12)
    g = (num.clamp_min(0.0) / den).sqrt()
    out = m_v + g * (m_f - m_v) + resid
    return out.reshape(x.shape), float(g)


def xvid_video(x: torch.Tensor, rel: torch.Tensor, n_frames: int) -> torch.Tensor:
    """xvid with the per-video mean of raw X in place of the batch mean."""
    b = x.shape[0] // n_frames
    n, d = x.shape[1], x.shape[2]
    xx = x.view(b, n_frames, n, d)
    rr = rel.view(b, n_frames, n, d)
    m_v = xx.mean(dim=(1, 2), keepdim=True)
    num = (xx - m_v).square().sum(-1).mean()
    den = (rr - m_v).square().sum(-1).mean().clamp_min(1e-12)
    g = (num / den).sqrt()
    out = m_v + g * (rr - m_v)
    return out.reshape(x.shape), float(g)


def xvid(x: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=(0, 1), keepdim=True)
    g = ((x - mu).square().sum(-1).mean() / (rel - mu).square().sum(-1).mean().clamp_min(1e-12)).sqrt()
    return mu + g * (rel - mu)


def mean_direction(x: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
    """(PX - m_f) + ||m_f|| u_f, u_f = (m_f - mu) / ||m_f - mu||.

    Frames whose frame mean sits on the batch mean stay at PX.
    """
    mu = x.mean(dim=(0, 1), keepdim=True)
    m_f = rel.mean(dim=1, keepdim=True)
    d = m_f - mu
    nrm = d.norm(dim=-1, keepdim=True)
    u = d / nrm.clamp_min(1e-12)
    m_new = m_f.norm(dim=-1, keepdim=True) * u
    out = rel - m_f + m_new
    return torch.where(nrm > 1e-6, out, rel)


def video_energy(x: torch.Tensor, rel: torch.Tensor, n_frames: int) -> torch.Tensor:
    """PX + (g-1)(m_v - mu). g matches raw energy on the video mean only."""
    b = rel.shape[0] // n_frames
    n, d = rel.shape[1], rel.shape[2]
    r = rel.view(b, n_frames, n, d)
    xx = x.view(b, n_frames, n, d)
    mu = xx.mean(dim=(0, 1, 2), keepdim=True)
    m_v = r.mean(dim=(1, 2), keepdim=True)
    num = (xx - mu).square().sum(-1).mean() - (r - m_v).square().sum(-1).mean()
    den = (m_v - mu).square().sum(-1).mean().clamp_min(1e-12)
    g = (num.clamp_min(0.0) / den).sqrt()
    out = r + (g - 1.0) * (m_v - mu)
    return out.reshape(rel.shape), float(g)


def video_mean(rel: torch.Tensor, n_frames: int) -> torch.Tensor:
    b = rel.shape[0] // n_frames
    r = rel.view(b, n_frames, rel.shape[1], rel.shape[2])
    m_f = r.mean(dim=2, keepdim=True)
    m_v = m_f.mean(dim=1, keepdim=True)
    return (r - m_f + m_v).reshape(rel.shape)


def drop_shared_residual(rel: torch.Tensor) -> torch.Tensor:
    """Remove the top cross-frame direction of r = PX - m_f. Frame mean stays."""
    m_f = rel.mean(dim=1, keepdim=True)
    r = rel - m_f
    # Power iteration on the token covariance of r. r is (F, N, D).
    flat = r.reshape(-1, r.shape[-1])
    v = flat[0]
    v = v / v.norm().clamp_min(1e-12)
    for _ in range(8):
        v = flat.transpose(0, 1) @ (flat @ v)
        v = v / v.norm().clamp_min(1e-12)
    coeff = (r * v).sum(dim=-1, keepdim=True)
    return rel - coeff * v


def mean_nudge(x: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
    """Replace m_f by ||m_f|| (2 m_f - mu) / ||2 m_f - mu||. Residual stays."""
    mu = x.mean(dim=(0, 1), keepdim=True)
    m_f = rel.mean(dim=1, keepdim=True)
    direction = 2.0 * m_f - mu
    nrm = direction.norm(dim=-1, keepdim=True)
    m_new = m_f.norm(dim=-1, keepdim=True) * direction / nrm.clamp_min(1e-12)
    out = rel - m_f + m_new
    return torch.where(nrm > 1e-6, out, rel)


def axis_swap(rel: torch.Tensor) -> torch.Tensor:
    """Swap each token's coordinates on the batch-mean axis and the frame axis."""
    mu = rel.mean(dim=(0, 1), keepdim=True)
    u_hat = mu / mu.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    m_f = rel.mean(dim=1, keepdim=True)
    a = (m_f * u_hat).sum(dim=-1, keepdim=True)
    q = m_f - a * u_hat
    qn = q.norm(dim=-1, keepdim=True)
    u = q / qn.clamp_min(1e-12)
    a_x = (rel * u_hat).sum(dim=-1, keepdim=True)
    b_x = (rel * u).sum(dim=-1, keepdim=True)
    out = rel + (b_x - a_x) * u_hat + (a_x - b_x) * u
    return torch.where(qn > 1e-6, out, rel)


def frame_stats(tokens: torch.Tensor):
    """tokens (F, N, D) -> within-frame cosines (F,), frame means of unit tokens (F, D)."""
    u = F.normalize(F.layer_norm(tokens, (tokens.shape[-1],)), dim=-1)
    m = u.mean(1)
    n = u.shape[1]
    cin = (n * m.square().sum(-1) - 1.0) / (n - 1)
    return cin, m


def main():
    dataset = sys.argv[1]
    n_videos = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    n_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    batch_videos = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    device = torch.device("cpu")
    rng = np.random.default_rng(0)
    cfg = configuration.load_config(CONFIGS[dataset])
    bb = cfg.model.encoder.backbone
    backbone = TimmExtractor(
        model=bb.model, pretrained=True, frozen=True, features=bb.features,
        model_kwargs=dict(bb.get("model_kwargs") or {}) or None,
    ).to(device).eval()
    clips = load_clips(cfg, "/workspace/dataset", n_videos, n_frames, 8, rng)
    print(f"{dataset}: {len(clips)} videos x {n_frames} frames", flush=True)
    names = ("dense", "xframe", "mvmf", "venergy")
    cin = {k: [] for k in names}
    means = {k: [] for k in names}
    to_raw = {k: [] for k in names}
    vids = []
    with torch.no_grad():
        for b0 in range(0, len(clips), batch_videos):
            batch = clips[b0:b0 + batch_videos]
            b = len(batch)
            imgs = torch.stack(batch, 0).reshape(b * n_frames, *batch[0].shape[1:])
            tok = []
            for c0 in range(0, imgs.shape[0], 4):
                t = backbone(imgs[c0:c0 + 4].to(device))
                tok.append((t[bb.features] if isinstance(t, dict) else t).float().cpu())
            tok = torch.cat(tok, 0)
            rel = dense_px(tok)
            xframe, _ = xframe_rescale(tok, rel)
            mvmf, _ = xframe_video(tok, rel, n_frames)
            venergy, _ = video_energy(tok, rel, n_frames)
            feats = {
                "dense": rel,
                "xframe": xframe,
                "mvmf": mvmf,
                "venergy": venergy,
            }
            raw_u = F.normalize(F.layer_norm(tok, (tok.shape[-1],)), dim=-1)
            for name, feat in feats.items():
                c, m = frame_stats(feat)
                cin[name].append(c)
                means[name].append(m)
                feat_u = F.normalize(F.layer_norm(feat, (feat.shape[-1],)), dim=-1)
                to_raw[name].append((feat_u * raw_u).sum(-1).mean())
            vids.append(torch.arange(b0, b0 + b).repeat_interleave(n_frames))
            print(f"  videos {b0 + b}/{len(clips)}", flush=True)
    vid = torch.cat(vids, 0)
    print(f"\n{dataset}  LN+L2 cosine, {len(clips)} videos, {n_frames} frames")
    print(f"{'method':8s} {'within':>8s} {'cross_frame':>12s} {'cross_video':>12s} {'vs_raw':>8s}")
    for name in names:
        c = torch.cat(cin[name], 0)
        m = torch.cat(means[name], 0)
        dot = m @ m.T
        same_v = vid[:, None].eq(vid[None, :])
        eye = torch.eye(m.shape[0], dtype=torch.bool)
        cross_v = dot[~same_v].mean()
        cross_f = dot[same_v & ~eye].mean()
        raw_cos = torch.stack(to_raw[name]).mean()
        print(
            f"{name:8s} {float(c.mean()):8.3f} {float(cross_f):12.3f} "
            f"{float(cross_v):12.3f} {float(raw_cos):8.3f}"
        )


if __name__ == "__main__":
    main()
