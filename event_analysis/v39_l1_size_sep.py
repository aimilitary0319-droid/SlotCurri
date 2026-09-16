"""Can v39 λ1 separate large objects, small objects, and ghosts?

Same win-share buckets as calibrate_tau_g.py (ghost=0 argmax wins, small=1..5%,
large>5%), plus a GT-IoU view. Reports λ1 vs mass vs π on a trained v39 ckpt.

Usage (GPU 6 or 7):
  python event_analysis/v39_l1_size_sep.py --dataset ytvis --max-clips 80
  python event_analysis/v39_l1_size_sep.py --dataset movi --max-clips 80
"""
from __future__ import annotations

import argparse
import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from slotcurri import configuration, data, models
from slotcurri.modules.video import _top2_algebraic_n8

EPS = 1e-6
SMALL_FRAC = 0.05
IOU_GHOST = 0.10
IOU_MATCH = 0.30

RUNS = {
    "ytvis": {
        "config": "configs/slotcurri/ytvis2021_attnmass_v39.yaml",
        "ckpt": "logs/_ytvis_attnmass_v39/checkpoints/slotcurri_step=step=100000-v1.ckpt",
        "ignore_bg": False,
    },
    "movi": {
        "config": "configs/slotcurri/movi_c_attnmass_v39.yaml",
        "ckpt": "logs/_movi_c_attnmass_v39/checkpoints/slotcurri_step=step=100000-v1.ckpt",
        "ignore_bg": True,
    },
}


class _FakeTrainer:
    def __init__(self, step: int):
        self.global_step = step


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort().astype(np.float64) + 1.0
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def summarize(xs: np.ndarray) -> str:
    if xs.size == 0:
        return "n=0"
    return (
        f"n={xs.size:6d}  mean={xs.mean():.4f}  med={np.median(xs):.4f}  "
        f"p10={np.percentile(xs, 10):.4f}  p90={np.percentile(xs, 90):.4f}"
    )


@torch.no_grad()
def n8_spectrum(att: torch.Tensor, feat: torch.Tensor, chunk: int = 8):
    z = F.normalize(feat.float(), dim=-1)
    a = att.float()
    t, s, _ = a.shape
    lam1 = a.new_empty(t, s)
    lam2 = a.new_empty(t, s)
    step = max(int(chunk), 1)
    for i in range(0, t, step):
        l1, l2 = _top2_algebraic_n8(z[i : i + step], a[i : i + step], 16, EPS)
        lam1[i : i + step] = l1
        lam2[i : i + step] = l2
    lam2p = lam2.clamp_min(0.0)
    pi = (lam1 - lam2p).clamp_min(0.0)
    return lam1, lam2, pi


def _one_hot_gt(seg: torch.Tensor) -> torch.Tensor:
    if seg.ndim == 5:
        return seg.bool()
    ncls = int(seg.max().item()) + 1
    b, t, h, w = seg.shape
    oh = torch.zeros(b, t, ncls, h, w, dtype=torch.bool, device=seg.device)
    for c in range(ncls):
        oh[:, :, c] = seg == c
    return oh


def _iou_slot_gt(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # pred (T,S,H,W) bool, gt (T,C,H,W) bool -> (T,S,C)
    p = pred.flatten(-2).float()
    g = gt.flatten(-2).float()
    inter = torch.einsum("tsn,tcn->tsc", p, g)
    area_p = p.sum(-1).unsqueeze(-1)
    area_g = g.sum(-1).unsqueeze(1)
    return inter / (area_p + area_g - inter).clamp_min(1.0)


def overlap_at(pos: np.ndarray, neg: np.ndarray, keep_small: float = 0.95) -> Tuple[float, float]:
    """Threshold that keeps `keep_small` of pos; fraction of neg that still pass."""
    if pos.size == 0 or neg.size == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(pos, 1.0 - keep_small))
    return thr, float((neg >= thr).mean())


@torch.no_grad()
def collect(model, loader, device, max_clips: int, ignore_bg: bool) -> Dict[str, np.ndarray]:
    rows = {k: [] for k in (
        "l1", "l2", "pi", "mass", "win_frac", "max_iou_fg", "gt_area", "matched"
    )}
    n_clips = 0
    for batch in loader:
        if "batch_padding_mask" in batch:
            batch = model._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                continue
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model.forward(batch, train=False, cycle=False)
            aux = model.aux_forward(batch, outputs)

        att = outputs["processor"]["state_attn_mask"].float()[0]  # (T,S,N)
        bind = outputs["encoder"].get("backbone_key")
        if bind is None:
            bind = outputs["encoder"]["backbone_features"]
        bind = bind.float()[0]
        lam1, lam2, pi = n8_spectrum(att, bind, chunk=8)
        t_len, n_slots, n_tokens = att.shape
        mass = att.sum(dim=-1) / float(n_tokens)
        winner = att.argmax(dim=1)
        win_counts = torch.zeros(t_len, n_slots, device=att.device)
        win_counts.scatter_add_(
            1, winner, torch.ones_like(winner, dtype=win_counts.dtype)
        )
        win_frac = win_counts / float(n_tokens)

        key = (
            "decoder_masks_vis_hard"
            if "decoder_masks_vis_hard" in aux
            else "decoder_masks_hard"
        )
        pred = aux[key].bool()[0]
        gt_oh = _one_hot_gt(batch["segmentations"])[0]
        gt_oh = F.interpolate(
            gt_oh.float(), size=pred.shape[-2:], mode="nearest"
        ).bool()
        if ignore_bg and gt_oh.shape[1] > 0:
            gt_oh = gt_oh.clone()
            gt_oh[:, 0] = False
        iou = _iou_slot_gt(pred, gt_oh)  # (T,S,C)
        if ignore_bg and iou.shape[-1] > 0:
            iou[:, :, 0] = 0.0
        max_iou, best_c = iou.max(dim=-1)
        hw = pred.shape[-2] * pred.shape[-1]
        gt_area = torch.zeros_like(max_iou)
        for c in range(gt_oh.shape[1]):
            area_c = gt_oh[:, c].flatten(1).float().sum(-1) / float(hw)  # (T,)
            gt_area = torch.where(best_c == c, area_c[:, None].expand_as(gt_area), gt_area)
        matched = max_iou >= IOU_MATCH

        def take(x):
            return x.reshape(-1).cpu().numpy()

        rows["l1"].append(take(lam1))
        rows["l2"].append(take(lam2))
        rows["pi"].append(take(pi))
        rows["mass"].append(take(mass))
        rows["win_frac"].append(take(win_frac))
        rows["max_iou_fg"].append(take(max_iou))
        rows["gt_area"].append(take(gt_area))
        rows["matched"].append(take(matched.float()))

        n_clips += 1
        if n_clips % 10 == 0:
            print(f"  clips={n_clips}", flush=True)
        if n_clips >= max_clips:
            break
    out = {k: np.concatenate(v) for k, v in rows.items()}
    out["n_clips"] = np.array([n_clips])
    return out


def report_buckets(name: str, score: np.ndarray, ghost, small, large):
    print(f"\n== {name} ==")
    print(f"  ghost  {summarize(score[ghost])}")
    print(f"  small  {summarize(score[small])}")
    print(f"  large  {summarize(score[large])}")
    print(
        f"  AUC  small>ghost={auc_rank(score[small], score[ghost]):.3f}  "
        f"large>ghost={auc_rank(score[large], score[ghost]):.3f}  "
        f"large>small={auc_rank(score[large], score[small]):.3f}"
    )
    thr, leak = overlap_at(score[small], score[ghost], 0.95)
    print(f"  keep 95% small: thr={thr:.4f}  ghost pass={leak:.1%}")
    thr2, leak2 = overlap_at(score[large], score[ghost], 0.95)
    print(f"  keep 95% large: thr={thr2:.4f}  ghost pass={leak2:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ytvis", "movi"), required=True)
    ap.add_argument("--max-clips", type=int, default=80)
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--step", type=int, default=100000)
    args = ap.parse_args()

    spec = RUNS[args.dataset]
    config = configuration.load_config(spec["config"])
    config.model.visualize = False
    config.dataset.val_batch_size = 1
    config.dataset.num_val_workers = 0
    model = models.build(config.model, config.optimizer)
    model.load_weights_from_checkpoint(spec["ckpt"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    model._trainer = _FakeTrainer(args.step)

    dataset = data.build(config.dataset, data_dir=args.data_dir)
    dataset.setup("validate")
    loader = dataset.val_dataloader()
    print(f"dataset={args.dataset} ckpt={spec['ckpt']} device={device}", flush=True)
    pack = collect(model, loader, device, args.max_clips, spec["ignore_bg"])
    n = int(pack["n_clips"][0])
    l1, pi, mass = pack["l1"], pack["pi"], pack["mass"]
    win = pack["win_frac"]
    ghost = win == 0
    small = (win > 0) & (win <= SMALL_FRAC)
    large = win > SMALL_FRAC
    print(f"\nclips={n}  slots={l1.size}  "
          f"ghost={ghost.sum()} small={small.sum()} large={large.sum()}  "
          f"(win-share buckets)")
    print(f"corr(λ1, mass)={np.corrcoef(l1, mass)[0,1]:.3f}  "
          f"corr(λ1, win)={np.corrcoef(l1, win)[0,1]:.3f}  "
          f"corr(λ1, π)={np.corrcoef(l1, pi)[0,1]:.3f}")

    report_buckets("λ1  (win-share)", l1, ghost, small, large)
    report_buckets("mass (win-share)", mass, ghost, small, large)
    report_buckets("π    (win-share)", pi, ghost, small, large)

    iou = pack["max_iou_fg"]
    area = pack["gt_area"]
    g_gt = iou < IOU_GHOST
    m_gt = pack["matched"] > 0.5
    small_gt = m_gt & (area <= SMALL_FRAC)
    large_gt = m_gt & (area > SMALL_FRAC)
    print(f"\nGT view: unmatched(iou<{IOU_GHOST})={g_gt.sum()}  "
          f"matched-small={small_gt.sum()} matched-large={large_gt.sum()}")
    report_buckets("λ1  (GT match)", l1, g_gt, small_gt, large_gt)
    report_buckets("mass (GT match)", mass, g_gt, small_gt, large_gt)
    report_buckets("π    (GT match)", pi, g_gt, small_gt, large_gt)

    if m_gt.any():
        print(f"\nmatched only: corr(λ1, GT area)={np.corrcoef(l1[m_gt], area[m_gt])[0,1]:.3f}  "
              f"corr(mass, GT area)={np.corrcoef(mass[m_gt], area[m_gt])[0,1]:.3f}  "
              f"corr(π, GT area)={np.corrcoef(pi[m_gt], area[m_gt])[0,1]:.3f}")


if __name__ == "__main__":
    main()
