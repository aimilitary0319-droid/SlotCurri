"""Offline probe for the proposed intra-slot motion-contrast loss (merge splitter).

Question: does the attention-weighted variance of the patch temporal feature-change
magnitude separate MERGED slots (one slot covering >= 2 GT instances) from SINGLE-object
slots? If yes, the statistic can be charged as a loss (gradient into attention) to split
similar adjacent objects that featrec/purity/rent are all blind to.

Per frame pair (t, t+1), per slot s:
  m(f)      = || h_{t+1}(f) - h_t(f) ||                (frozen DINO features, detached)
  w_s(f)    = A~_{s,f} / sum_f A~_{s,f}                (gamma-sharpened attention, frame t)
  score_var = Var_{w_s}[m] / Var_scene[m]              (the proposed loss statistic)
  score_gain= 1 - min_split SSE_2(m; w_s)/SSE_1(m; w_s) (robust bimodality alternative;
              best 1-D two-cluster split with >= 5% weight on each side)

Slot categories from GT segmentations projected on the patch grid, using the slot's hard
decoder-mask territory at frame t:
  merged  = >= 2 GT instances, each holding >= max(8, 20%) of the territory
  single  = one GT instance holds >= 60% of the territory, runner-up < 8 patches
  (background/junk/ghost territories fall into 'other' and are excluded)

Usage (inside the slotcurri container):
  python event_analysis/motion_split_probe.py \
      --config configs/slotcurri/ytvis2021_attnmass_v26.yaml \
      --ckpt "logs/_ytvis_attnmass_v26/checkpoints/slotcurri_step=step=100000-v1.ckpt" \
      --data-dir /workspace/dataset --max-clips 200
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from slotcurri import configuration, data, models

EPS = 1e-8


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(pos > neg). 1.0 = perfectly separable."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort().astype(np.float64) + 1.0
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def seg_to_patch_ids(seg: torch.Tensor, grid: int) -> torch.Tensor:
    """GT segmentations -> integer ids on the (grid x grid) patch lattice. (B,T,P)"""
    if seg.ndim == 5:  # (B, T, C, H, W) one-hot / scores
        seg = seg.float().argmax(dim=2)
    seg = seg.float().unsqueeze(2)  # (B, T, 1, H, W)
    b, t = seg.shape[:2]
    seg = seg.reshape(b * t, 1, *seg.shape[-2:])
    ids = F.interpolate(seg, size=(grid, grid), mode="nearest")
    return ids.reshape(b, t, grid * grid).long()


def split_gain(mval: torch.Tensor, w: torch.Tensor, min_side: float = 0.05) -> torch.Tensor:
    """1 - best weighted 2-cluster SSE / 1-cluster SSE, per (N, S) row.

    mval: (N, F) patch motion magnitudes; w: (N, S, F) weights summing to 1 over F.
    Split candidates are thresholds in sorted-m order with >= min_side weight per side.
    """
    n, s, f = w.shape
    idx = mval.argsort(dim=-1)  # (N, F)
    m_sorted = mval.gather(-1, idx).unsqueeze(1)  # (N, 1, F)
    w_sorted = w.gather(-1, idx.unsqueeze(1).expand(-1, s, -1))  # (N, S, F)

    wm = w_sorted * m_sorted
    wm2 = w_sorted * m_sorted.pow(2)
    W = w_sorted.cumsum(-1)
    M1 = wm.cumsum(-1)
    M2 = wm2.cumsum(-1)
    Wt, M1t, M2t = W[..., -1:], M1[..., -1:], M2[..., -1:]

    sse1 = (M2t - M1t.pow(2) / Wt.clamp_min(EPS)).squeeze(-1)  # (N, S)

    sse_a = M2 - M1.pow(2) / W.clamp_min(EPS)
    Wb, M1b, M2b = Wt - W, M1t - M1, M2t - M2
    sse_b = M2b - M1b.pow(2) / Wb.clamp_min(EPS)
    sse2 = sse_a + sse_b  # (N, S, F)

    valid = (W >= min_side) & (Wb >= min_side)
    sse2 = torch.where(valid, sse2, torch.full_like(sse2, float("inf")))
    best = sse2.min(dim=-1).values  # (N, S)
    gain = 1.0 - best / sse1.clamp_min(EPS)
    return gain.clamp(0.0, 1.0).where(torch.isfinite(best), torch.zeros_like(best))


@torch.no_grad()
def collect(model, loader, max_clips, device, mass_gamma):
    rows = {k: [] for k in ("score_var", "score_gain", "category", "clip", "mass")}
    n_clips = 0
    skipped_static = 0
    for batch in loader:
        if "batch_padding_mask" in batch:
            batch = model._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                continue
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model.forward(batch, train=False, cycle=model.cyclic_inference)

        h = outputs["encoder"]["backbone_features"].float()  # (B, T, F, D)
        att = outputs["processor"]["state_attn_mask"].float()  # (B, T, S, F)
        dec_masks = outputs["decoder"]["masks"].float()  # (B, T, S, F)
        seg = batch["segmentations"]
        b, t, s, f = att.shape
        if t < 2:
            continue
        grid = int(round(f ** 0.5))

        # gamma-sharpened attention, normalized over patches -> per-slot weights
        att_sharp = att.pow(mass_gamma)
        att_sharp = att_sharp / att_sharp.sum(dim=2, keepdim=True).clamp_min(EPS)
        w = att_sharp / att_sharp.sum(dim=-1, keepdim=True).clamp_min(EPS)  # (B,T,S,F)
        mass = att_sharp.sum(dim=-1) / f  # (B, T, S)

        # patch motion magnitude per frame pair
        mval = (h[:, 1:] - h[:, :-1]).norm(dim=-1)  # (B, T-1, F)
        scene_var = mval.var(dim=-1, unbiased=False)  # (B, T-1)

        w_t = w[:, :-1].reshape(b * (t - 1), s, f)
        m_t = mval.reshape(b * (t - 1), f)
        mu = (w_t * m_t.unsqueeze(1)).sum(-1)  # (N, S)
        var = (w_t * (m_t.unsqueeze(1) - mu.unsqueeze(-1)).pow(2)).sum(-1)  # (N, S)
        sv = scene_var.reshape(-1, 1).clamp_min(EPS)
        score_var = var / sv
        score_gain = split_gain(m_t, w_t)

        # categories from GT on the slot's hard decoder territory (frame t of each pair)
        gt = seg_to_patch_ids(seg, grid)  # (B, T, F)
        winner = dec_masks.argmax(dim=2)  # (B, T, F)
        cat = torch.zeros(b, t - 1, s, dtype=torch.long)
        for bi in range(b):
            for ti in range(t - 1):
                ids_frame = gt[bi, ti]
                win_frame = winner[bi, ti]
                for si in range(s):
                    terr = win_frame == si
                    n_terr = int(terr.sum())
                    if n_terr < 12:
                        continue
                    ids, counts = ids_frame[terr].unique(return_counts=True)
                    fg = ids != 0
                    ids, counts = ids[fg], counts[fg]
                    if ids.numel() == 0:
                        continue
                    counts = counts.sort(descending=True).values
                    thr = max(8, int(0.2 * n_terr))
                    if ids.numel() >= 2 and int(counts[1]) >= thr:
                        cat[bi, ti, si] = 2  # merged
                    elif int(counts[0]) >= 0.6 * n_terr and (
                        ids.numel() == 1 or int(counts[1]) < 8
                    ):
                        cat[bi, ti, si] = 1  # single

        keep = (scene_var.reshape(-1) > 1e-6).cpu()
        skipped_static += int((~keep).sum())
        rows["score_var"].append(score_var.cpu()[keep])
        rows["score_gain"].append(score_gain.cpu()[keep])
        rows["category"].append(cat.reshape(-1, s)[keep])
        rows["mass"].append(mass[:, :-1].reshape(-1, s).cpu()[keep])
        rows["clip"].append(
            torch.full((int(keep.sum()), s), n_clips, dtype=torch.long)
        )

        n_clips += b
        if n_clips >= max_clips:
            break
    out = {k: torch.cat(v).numpy() for k, v in rows.items()}
    out["n_clips"] = n_clips
    out["skipped_static"] = skipped_static
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/slotcurri/ytvis2021_attnmass_v26.yaml")
    ap.add_argument(
        "--ckpt",
        default="logs/_ytvis_attnmass_v26/checkpoints/slotcurri_step=step=100000-v1.ckpt",
    )
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--max-clips", type=int, default=200)
    ap.add_argument("--mass-gamma", type=float, default=2.0)
    ap.add_argument("--cache", default="event_analysis/motion_split_stats_v26.npz")
    ap.add_argument("--out", default="event_analysis/motion_split_probe_v26.png")
    args = ap.parse_args()

    if args.cache and os.path.exists(args.cache):
        blob = dict(np.load(args.cache))
        print(f"loaded cached stats from {args.cache}")
    else:
        config = configuration.load_config(args.config)
        dataset = data.build(config.dataset, data_dir=args.data_dir)
        model = models.build(config.model, config.optimizer, None, None)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)
        dataset.setup("validate")
        blob = collect(model, dataset.val_dataloader(), args.max_clips, device,
                       args.mass_gamma)
        if args.cache:
            np.savez_compressed(args.cache, **blob)
            print(f"cached stats to {args.cache}")

    sv, sg = blob["score_var"], blob["score_gain"]
    cat, mass = blob["category"], blob["mass"]
    print(f"clips={int(blob['n_clips'])}  slot-frames={sv.size}  "
          f"static frame-pairs skipped={int(blob['skipped_static'])}")

    merged, single = cat == 2, cat == 1
    print(f"category counts: merged={int(merged.sum())}  single={int(single.sum())}  "
          f"other={int((cat == 0).sum())}")

    print("\n--- statistic distributions (median [q10, q90]) ---")
    print(f"{'statistic':>10s} | {'merged':>24s} | {'single':>24s} | {'AUC':>6s}")
    for name, x in (("var_norm", sv), ("split_gain", sg)):
        vm, vs = x[merged], x[single]
        auc = auc_rank(vm, vs)
        print(f"{name:>10s} | {np.median(vm):6.3f} [{np.quantile(vm, .1):6.3f},"
              f" {np.quantile(vm, .9):6.3f}] | {np.median(vs):6.3f}"
              f" [{np.quantile(vs, .1):6.3f}, {np.quantile(vs, .9):6.3f}] | {auc:.4f}")

    # articulated-object false-positive check: among singles, how many score above the
    # merged median?
    for name, x in (("var_norm", sv), ("split_gain", sg)):
        thr = np.median(x[merged])
        fp = float((x[single] > thr).mean()) if single.any() else float("nan")
        print(f"{name}: singles above merged-median threshold: {fp:.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (name, x) in zip(axes, (("var_norm (proposed loss)", sv),
                                    ("split_gain (robust alt)", sg))):
        for gname, mask, color in (("merged", merged, "tab:red"),
                                   ("single", single, "tab:green")):
            v = x[mask]
            if v.size:
                hi = np.quantile(x[merged | single], 0.99)
                ax.hist(np.clip(v, 0, hi), bins=80, alpha=0.55, density=True,
                        label=f"{gname} (n={v.size})", color=color)
        ax.set_xlabel(name)
        ax.legend()
    m_all, s_all = auc_rank(sv[merged], sv[single]), auc_rank(sg[merged], sg[single])
    fig.suptitle(f"intra-slot motion contrast on v26 @ 100k  "
                 f"(AUC var={m_all:.3f}, gain={s_all:.3f})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved plot: {args.out}")


if __name__ == "__main__":
    main()
