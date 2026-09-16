"""GT object size vs v37 π and v37g gap on a trained v37 eval partition.

Per foreground instance: best slot by patch-grid IoU. Reports that slot's
  occupancy, v37 π=(λ1-λ2)/mass, v37g gap=λ1-λ2
against the instance's image-area fraction. C_s on raw backbone tokens
(eval already mix=1).

Usage:
  python event_analysis/v37g_gt_size_sep.py --dataset ytvis --max-clips 40
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from slotcurri import configuration, data, metrics, models
from slotcurri.modules.video import spectral_slot_purity

IOU_MATCH = 0.30
IOU_GHOST = 0.10
SMALL_FRAC = 0.05

RUNS = {
    "ytvis": {
        "config": "configs/slotcurri/ytvis2021_attnmass_v37.yaml",
        "ckpt": "logs/_ytvis_attnmass_v37/checkpoints/slotcurri_step=step=100000-v1.ckpt",
        "ignore_bg": True,
    },
    "movi": {
        "config": "configs/slotcurri/movi_c_attnmass_v37.yaml",
        "ckpt": "logs/_movi_c_attnmass_v37/checkpoints/slotcurri_step=step=100000-v1.ckpt",
        "ignore_bg": True,
    },
}


class _FakeTrainer:
    def __init__(self, step: int):
        self.global_step = step


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
    p = pred.flatten(-2).float()
    g = gt.flatten(-2).float()
    inter = torch.einsum("tsn,tcn->tsc", p, g)
    area_p = p.sum(-1).unsqueeze(-1)
    area_g = g.sum(-1).unsqueeze(1)
    return inter / (area_p + area_g - inter).clamp_min(1.0)


def summarize(xs: np.ndarray) -> str:
    if xs.size == 0:
        return "n=0"
    return (
        f"n={xs.size:5d}  mean={xs.mean():.4f}  med={np.median(xs):.4f}  "
        f"p10={np.percentile(xs, 10):.4f}  p90={np.percentile(xs, 90):.4f}"
    )


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort().astype(np.float64) + 1.0
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def overlap_at(pos: np.ndarray, neg: np.ndarray, keep_small: float = 0.95):
    if pos.size == 0 or neg.size == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(pos, 1.0 - keep_small))
    return thr, float((neg >= thr).mean())


@torch.no_grad()
def collect(model, loader, max_clips: int, ignore_bg: bool, proj: int):
    rec = {k: [] for k in ("area", "iou", "occ", "pi", "gap", "mass", "rho")}
    slot = {k: [] for k in (
        "occ", "pi", "gap", "mass", "win_frac", "max_iou", "gt_area", "r_gap", "r_pi",
    )}
    n_clips = 0
    for batch in loader:
        if "batch_padding_mask" in batch:
            batch = model._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                continue
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
        outputs = model.forward(batch, train=False, cycle=False)
        att = outputs["processor"]["state_attn_mask"].float()[0]  # (T,S,N)
        raw = outputs["encoder"]["backbone_features"].float()[0]
        t, s, n = att.shape
        grid = int(round(n ** 0.5))
        occ = (att * att).sum(-1) / att.sum(-1).clamp_min(1e-8)
        pi = spectral_slot_purity(att, raw, divide_by_mass=True, proj_dim=proj)
        gap = spectral_slot_purity(att, raw, divide_by_mass=False, proj_dim=proj)
        mass = att.sum(-1)
        denom = (occ * mass).clamp_min(1e-6)
        rho = (1.0 - gap / denom).clamp(0.0, 1.0)
        r_gap = gap / gap.max(dim=-1, keepdim=True).values.clamp_min(1e-8)
        r_pi = pi / pi.max(dim=-1, keepdim=True).values.clamp_min(1e-8)

        winner = att.argmax(dim=1)
        pred = torch.zeros(t, s, grid, grid, dtype=torch.bool, device=att.device)
        for si in range(s):
            pred[:, si] = (winner == si).reshape(t, grid, grid)
        win_frac = pred.flatten(-2).float().mean(-1)

        gt_oh = _one_hot_gt(batch["segmentations"])[0]
        gt_p = F.interpolate(gt_oh.float(), size=(grid, grid), mode="nearest").bool()
        c0 = 1 if ignore_bg else 0
        if gt_p.shape[1] <= c0:
            n_clips += 1
            if n_clips >= max_clips:
                break
            continue
        fg = gt_p[:, c0:]
        iou = _iou_slot_gt(pred, fg)  # (T,S,C)
        hw = float(grid * grid)
        max_iou, best_c = iou.max(dim=-1)
        gt_area = torch.zeros_like(max_iou)
        for ci in range(fg.shape[1]):
            area_c = fg[:, ci].flatten(1).float().sum(-1) / hw
            gt_area = torch.where(best_c == ci, area_c[:, None].expand_as(gt_area), gt_area)

        def take(x):
            return x.reshape(-1).detach().cpu().numpy()

        slot["occ"].append(take(occ))
        slot["pi"].append(take(pi))
        slot["gap"].append(take(gap))
        slot["mass"].append(take(mass))
        slot["win_frac"].append(take(win_frac))
        slot["max_iou"].append(take(max_iou))
        slot["gt_area"].append(take(gt_area))
        slot["r_gap"].append(take(r_gap))
        slot["r_pi"].append(take(r_pi))

        for ti in range(t):
            for ci in range(fg.shape[1]):
                gmask = fg[ti, ci]
                area = float(gmask.float().sum() / hw)
                if area <= 0:
                    continue
                best = int(iou[ti, :, ci].argmax())
                best_iou = float(iou[ti, best, ci])
                if best_iou < IOU_MATCH:
                    continue
                rec["area"].append(area)
                rec["iou"].append(best_iou)
                rec["occ"].append(float(occ[ti, best]))
                rec["pi"].append(float(pi[ti, best]))
                rec["gap"].append(float(gap[ti, best]))
                rec["mass"].append(float(mass[ti, best]))
                rec["rho"].append(float(rho[ti, best]))
        n_clips += 1
        if n_clips % 10 == 0:
            print(f"  clips={n_clips} objects={len(rec['area'])}", flush=True)
        if n_clips >= max_clips:
            break
    out = {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}
    out["n_clips"] = n_clips
    out["slot"] = {k: np.concatenate(v) for k, v in slot.items()}
    return out


def report(name, score, small, large):
    print(f"\n== {name} ==")
    print(f"  small (<=5% img)  {summarize(score[small])}")
    print(f"  large (>5% img)   {summarize(score[large])}")
    if small.any() and large.any():
        ratio = float(score[large].mean() / max(score[small].mean(), 1e-8))
        print(
            f"  large/small mean={ratio:.2f}  "
            f"AUC large>small={auc_rank(score[large], score[small]):.3f}"
        )


def report_buckets(name, score, ghost, small, large):
    print(f"\n== {name} ==")
    print(f"  ghost  {summarize(score[ghost])}")
    print(f"  small  {summarize(score[small])}")
    print(f"  large  {summarize(score[large])}")
    print(
        f"  AUC  small>ghost={auc_rank(score[small], score[ghost]):.3f}  "
        f"large>ghost={auc_rank(score[large], score[ghost]):.3f}  "
        f"large>small={auc_rank(score[large], score[small]):.3f}"
    )
    if small.any() and ghost.any():
        thr, leak = overlap_at(score[small], score[ghost], 0.95)
        print(f"  keep 95% small: thr={thr:.4f}  ghost pass={leak:.1%}")
    if large.any() and ghost.any():
        thr2, leak2 = overlap_at(score[large], score[ghost], 0.95)
        print(f"  keep 95% large: thr={thr2:.4f}  ghost pass={leak2:.1%}")
    if small.any() and ghost.any():
        print(
            f"  mean small/ghost={score[small].mean()/max(score[ghost].mean(), 1e-8):.2f}  "
            f"large/ghost={score[large].mean()/max(score[ghost].mean(), 1e-8):.2f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(RUNS), required=True)
    ap.add_argument("--max-clips", type=int, default=40)
    ap.add_argument("--data-dir", default="/workspace/dataset")
    args = ap.parse_args()
    spec = RUNS[args.dataset]
    print(f"loading {spec['ckpt']}", flush=True)
    config = configuration.load_config(spec["config"])
    config.model.visualize = False
    config.dataset.val_batch_size = 1
    config.dataset.num_val_workers = 0
    built = {n: metrics.build(c) for n, c in config.val_metrics.items()}
    model = models.build(config.model, config.optimizer, None, built)
    sd = torch.load(spec["ckpt"], map_location="cpu")
    model.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd, strict=False)
    model = model.cuda().eval()
    model._trainer = _FakeTrainer(100000)
    proj = int(getattr(model, "amc_spectral_proj_dim", 64) or 64)
    dm = data.build(config.dataset, data_dir=args.data_dir)
    dm.setup("validate")
    pack = collect(model, dm.val_dataloader(), args.max_clips, spec["ignore_bg"], proj)
    area, gap, pi, occ = pack["area"], pack["gap"], pack["pi"], pack["occ"]
    print(
        f"\n{args.dataset} clips={pack['n_clips']}  matched FG objects={area.size}  "
        f"proj={proj}  IoU>={IOU_MATCH}  C_s=raw X"
    )
    if area.size == 0:
        return
    small = area <= SMALL_FRAC
    large = area > SMALL_FRAC
    print(f"area  {summarize(area)}")
    print(
        f"corr(area, v37g gap)={np.corrcoef(area, gap)[0,1]:.3f}  "
        f"corr(area, v37 π)={np.corrcoef(area, pi)[0,1]:.3f}  "
        f"corr(area, occ)={np.corrcoef(area, occ)[0,1]:.3f}"
    )
    report("v37g gap  (λ1-λ2)", gap, small, large)
    report("v37 π     (gap/mass)", pi, small, large)
    report("occupancy", occ, small, large)
    q = np.quantile(area, [1 / 3, 2 / 3])
    print(f"\narea tertiles at {q[0]:.4f}, {q[1]:.4f}")
    bins = [
        ("T1 small", area <= q[0]),
        ("T2 mid", (area > q[0]) & (area <= q[1])),
        ("T3 large", area > q[1]),
    ]
    for title, m in bins:
        print(
            f"  {title:10s} n={m.sum():4d}  area={area[m].mean():.3f}  "
            f"gap={gap[m].mean():.1f}  π={pi[m].mean():.3f}  occ={occ[m].mean():.3f}"
        )
    if bins[0][1].any() and bins[2][1].any():
        print(
            f"  T3/T1  gap={gap[bins[2][1]].mean()/gap[bins[0][1]].mean():.2f}  "
            f"π={pi[bins[2][1]].mean()/max(pi[bins[0][1]].mean(),1e-8):.2f}"
        )

    sl = pack["slot"]
    win, iou_s, area_s = sl["win_frac"], sl["max_iou"], sl["gt_area"]
    g_win = win == 0
    sm_win = (win > 0) & (win <= SMALL_FRAC)
    lg_win = win > SMALL_FRAC
    print(
        f"\n--- slot view  n={win.size}  "
        f"win-ghost={g_win.sum()} small={sm_win.sum()} large={lg_win.sum()} ---"
    )
    report_buckets("v37g gap  (win-share)", sl["gap"], g_win, sm_win, lg_win)
    report_buckets("v37 π     (win-share)", sl["pi"], g_win, sm_win, lg_win)
    report_buckets("occupancy (win-share)", sl["occ"], g_win, sm_win, lg_win)
    report_buckets("decoder r_gap=gap/max", sl["r_gap"], g_win, sm_win, lg_win)
    report_buckets("decoder r_π=π/max", sl["r_pi"], g_win, sm_win, lg_win)

    g_gt = iou_s < IOU_GHOST
    m_gt = iou_s >= IOU_MATCH
    sm_gt = m_gt & (area_s <= SMALL_FRAC)
    lg_gt = m_gt & (area_s > SMALL_FRAC)
    print(
        f"\n--- GT slot view  unmatched(iou<{IOU_GHOST})={g_gt.sum()}  "
        f"matched-small={sm_gt.sum()} matched-large={lg_gt.sum()} ---"
    )
    report_buckets("v37g gap  (GT)", sl["gap"], g_gt, sm_gt, lg_gt)
    report_buckets("v37 π     (GT)", sl["pi"], g_gt, sm_gt, lg_gt)
    report_buckets("occupancy (GT)", sl["occ"], g_gt, sm_gt, lg_gt)
    report_buckets("decoder r_gap (GT)", sl["r_gap"], g_gt, sm_gt, lg_gt)
    report_buckets("decoder r_π (GT)", sl["r_pi"], g_gt, sm_gt, lg_gt)


if __name__ == "__main__":
    main()
