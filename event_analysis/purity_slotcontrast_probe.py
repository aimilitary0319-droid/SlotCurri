"""Purity probe on SlotContrast checkpoints: does purity separate large / small / ghost?

The v29 design question, re-asked on the UNGATED baseline (SlotContrast) so the answer
does not depend on our own curriculum having shaped the attention: per (frame, slot),
compute the confidence variants of the logratio gate

  c_ent     = 1 - H/log F   over the gamma-sharpened attention   (v26, entropy)
  pur_raw   = sum A^2 / sum A     over the RAW attention
  pur_sharp = sum A~^2 / sum A~   over the sharpened attention   (v29)

and test how well each separates slot groups. Two labeling modes:

  --label-mode wins   (YTVIS; GT too sparse/partial to trust): the established
      methodology of conf_vs_purity_probe.py -- slots grouped per frame by their
      argmax-win share on the raw attention:
        ghost = 0 wins | small = (0, small_frac] of patches | large > small_frac

  --label-mode gt     (MOVi-C; dense GT available, so use it): the slot's won patches
      are labeled with the majority GT segment (GT one-hot majority-pooled onto the
      patch grid), which fixes the two blind spots of the wins mode -- object size is
      the OBJECT's true size, not the slot's win count, and slots winning junk patches
      that belong to no object stop polluting the "small" group:
        ghost = 0 argmax wins (GT cannot rescue a slot that owns nothing)
        bg    = majority segment is background (class 0)
        mixed = wins > 0 but no GT segment reaches --align-thresh among won patches
        small = matched object whose per-frame patch-grid size <= small_frac
        large = matched object above small_frac

Inference is always a single forward sweep (SlotContrast protocol), regardless of the
config's cyclic_inference.

Usage (inside the slotcurri container):
  # MOVi-C, official SlotContrast checkpoint, GT labels
  python event_analysis/purity_slotcontrast_probe.py \
      --config configs/slotcurri/movi_c.yaml \
      --ckpt checkpoints/slotcontrast_official/movi_c.ckpt \
      --label-mode gt --max-clips 250 \
      --cache event_analysis/purity_slotcontrast_movi_c.npz \
      --out event_analysis/purity_slotcontrast_movi_c.png

  # YTVIS, locally trained SlotContrast baseline, wins labels (established method)
  python event_analysis/purity_slotcontrast_probe.py \
      --config configs/slotcurri/ytvis2021_slotcontrast.yaml \
      --ckpt "logs/_ytvis_slotcontrast/checkpoints/slotcurri_step=step=92000.ckpt" \
      --label-mode wins --max-clips 200 \
      --cache event_analysis/purity_slotcontrast_ytvis.npz \
      --out event_analysis/purity_slotcontrast_ytvis.png
"""
import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from slotcurri import configuration, data, models

EPS = 1e-6
GHOST, SMALL, LARGE, BG, MIXED = 0, 1, 2, 3, 4
LABEL_NAMES = {GHOST: "ghost", SMALL: "small", LARGE: "large", BG: "bg", MIXED: "mixed"}


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(pos > neg). 1.0 = perfectly separable, 0.5 = chance."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort().astype(np.float64) + 1.0
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def gt_patch_ids(seg: torch.Tensor, grid: int) -> torch.Tensor:
    """(B, T, C, H, W) one-hot (or (B, T, H, W) id map) -> (B*T, grid*grid) patch GT ids.

    One-hot channels are average-pooled onto the patch grid and argmaxed, i.e. each
    patch takes the segment that covers most of its pixels (majority vote).
    """
    if seg.ndim == 5:
        b, t, c, h, w = seg.shape
        one_hot = seg.reshape(b * t, c, h, w).float()
    else:
        b, t, h, w = seg.shape
        ids = seg.reshape(b * t, h, w).long()
        c = int(ids.max().item()) + 1
        one_hot = F.one_hot(ids, c).permute(0, 3, 1, 2).float()
    pooled = F.adaptive_avg_pool2d(one_hot, (grid, grid))  # (N, C, g, g)
    return pooled.argmax(dim=1).reshape(one_hot.shape[0], -1)  # (N, grid*grid)


def label_slots_gt(
    winner: torch.Tensor, wins: torch.Tensor, gt_ids: torch.Tensor,
    n_slots: int, small_frac: float, align_thresh: float,
):
    """GT labels per (frame, slot). Returns (labels, obj_frac, gt_align), each (N, S).

    counts[n, s, c] = #patches where slot s wins the argmax AND the patch's GT id is c.
    The slot is matched to its majority segment; alignment = counts_max / wins.
    """
    n, f = gt_ids.shape
    n_cls = int(gt_ids.max().item()) + 1
    win_oh = F.one_hot(winner, n_slots).float()  # (N, F, S)
    gt_oh = F.one_hot(gt_ids, n_cls).float()  # (N, F, C)
    counts = torch.bmm(win_oh.transpose(1, 2), gt_oh)  # (N, S, C)
    best_cnt, maj = counts.max(dim=-1)  # (N, S)
    align = best_cnt / wins.clamp_min(1.0)
    seg_size = gt_oh.sum(dim=1)  # (N, C) per-frame segment size in patches
    obj_frac = seg_size.gather(1, maj) / float(f)  # (N, S) majority segment's size

    labels = torch.full_like(wins, MIXED, dtype=torch.long)
    labels[maj == 0] = BG
    is_obj = (maj != 0) & (align >= align_thresh)
    labels[is_obj & (obj_frac <= small_frac)] = SMALL
    labels[is_obj & (obj_frac > small_frac)] = LARGE
    labels[wins == 0] = GHOST  # takes precedence: a slot that owns nothing is a ghost
    return labels, obj_frac, align


@torch.no_grad()
def collect(model, loader, max_clips, device, label_mode, small_frac, align_thresh):
    ms, ents, purs, pur_sharps, wins_all, labels_all = [], [], [], [], [], []
    n_clips = 0
    for batch in loader:
        if "batch_padding_mask" in batch:
            batch = model._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                continue
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            # single forward sweep, the SlotContrast inference protocol
            outputs = model.forward(batch, train=False, cycle=False)
        att = outputs["processor"]["state_attn_mask"].float()  # (B, T, S, F)
        b, t, s, f = att.shape
        att = att.reshape(b * t, s, f)

        att_sharp = att.pow(2.0)
        att_sharp = att_sharp / att_sharp.sum(dim=1, keepdim=True).clamp_min(1e-8)
        m = att_sharp.sum(dim=-1) / f  # (N, S)

        p_feat = att_sharp / att_sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ent = -(p_feat * p_feat.clamp_min(1e-8).log()).sum(dim=-1)
        c_ent = (1.0 - ent / math.log(f)).clamp(0.0, 1.0)

        pur_raw = (att * att).sum(dim=-1) / att.sum(dim=-1).clamp_min(1e-8)
        pur_sharp = (att_sharp * att_sharp).sum(dim=-1) / att_sharp.sum(dim=-1).clamp_min(1e-8)

        winner = att.argmax(dim=1)  # (N, F) raw-attention argmax, as in the v20 probe
        win_counts = torch.zeros(b * t, s, device=att.device)
        win_counts.scatter_add_(1, winner, torch.ones_like(winner, dtype=win_counts.dtype))

        if label_mode == "gt":
            grid = int(round(math.sqrt(f)))
            assert grid * grid == f, f"non-square patch grid: F={f}"
            gt_ids = gt_patch_ids(batch["segmentations"], grid).to(att.device)
            labels, _, _ = label_slots_gt(
                winner, win_counts, gt_ids, s, small_frac, align_thresh
            )
        else:
            small_hi = small_frac * f
            labels = torch.full_like(win_counts, LARGE, dtype=torch.long)
            labels[win_counts <= small_hi] = SMALL
            labels[win_counts == 0] = GHOST

        ms.append(m.cpu())
        ents.append(c_ent.cpu())
        purs.append(pur_raw.cpu())
        pur_sharps.append(pur_sharp.cpu())
        wins_all.append(win_counts.cpu())
        labels_all.append(labels.cpu())

        n_clips += b
        if n_clips >= max_clips:
            break
    return (
        torch.cat(ms).numpy(),
        torch.cat(ents).numpy(),
        torch.cat(purs).numpy(),
        torch.cat(pur_sharps).numpy(),
        torch.cat(wins_all).numpy(),
        torch.cat(labels_all).numpy(),
        n_clips,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label-mode", choices=["wins", "gt"], required=True)
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--max-clips", type=int, default=200)
    ap.add_argument("--small-frac", type=float, default=0.05,
                    help="object size threshold as a fraction of the patch grid")
    ap.add_argument("--align-thresh", type=float, default=0.5,
                    help="gt mode: min fraction of won patches on the majority segment")
    ap.add_argument("--beta", type=float, default=0.7, help="v29 beta_final working point")
    ap.add_argument("--p-end-mult", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.5, help="gate_tau_log working point")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.cache and os.path.exists(args.cache):
        blob = np.load(args.cache)
        m, c_ent, pur_raw, pur_sharp, wins, labels = (
            blob["m"], blob["c_ent"], blob["pur_raw"], blob["pur_sharp"],
            blob["wins"], blob["labels"],
        )
        n_clips = int(blob["n_clips"])
        print(f"loaded cached stats from {args.cache}")
    else:
        config = configuration.load_config(args.config)
        dataset = data.build(config.dataset, data_dir=args.data_dir)
        model = models.build(config.model, config.optimizer, None, None)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(
            ckpt["state_dict"] if "state_dict" in ckpt else ckpt, strict=False
        )
        assert not missing and not unexpected, (missing, unexpected)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)

        dataset.setup("validate")
        loader = dataset.val_dataloader()
        m, c_ent, pur_raw, pur_sharp, wins, labels, n_clips = collect(
            model, loader, args.max_clips, device,
            args.label_mode, args.small_frac, args.align_thresh,
        )
        if args.cache:
            np.savez_compressed(
                args.cache, m=m, c_ent=c_ent, pur_raw=pur_raw, pur_sharp=pur_sharp,
                wins=wins, labels=labels, n_clips=n_clips,
            )
            print(f"cached stats to {args.cache}")

    n_slots = m.shape[1]
    f_total = int(wins.sum(axis=1).max())
    p = args.p_end_mult / n_slots
    variants = {"entropy": c_ent, "pur_raw": pur_raw, "pur_sharp": pur_sharp}

    group_ids = [GHOST, SMALL, LARGE] + ([BG, MIXED] if args.label_mode == "gt" else [])
    masks = {LABEL_NAMES[g]: labels == g for g in group_ids}

    print(f"\nconfig={args.config}")
    print(f"ckpt={args.ckpt}")
    print(f"label_mode={args.label_mode}  clips={n_clips}  frames*slots={m.size}  "
          f"slots={n_slots}  patches/frame={f_total}")
    print("group sizes: " + "  ".join(f"{k}={v.sum()}" for k, v in masks.items()))
    print(f"working point: beta={args.beta}  p={p:.5f} "
          f"(p_end_mult={args.p_end_mult}/S={n_slots})  tau_g={args.tau}")

    print("\n--- 1. confidence by slot group (median [q10, q90]) ---")
    header = f"{'variant':>9s} |" + "".join(f" {k:>22s} |" for k in masks)
    print(header[:-2])
    for name, c in variants.items():
        row = f"{name:>9s} |"
        for mask in masks.values():
            v = c[mask]
            if v.size:
                row += (f" {np.median(v):5.3f} [{np.quantile(v, .1):5.3f},"
                        f" {np.quantile(v, .9):5.3f}] |")
            else:
                row += f" {'---':>22s} |"
        print(row[:-2])

    print("\n--- 2. conf-only separability (AUC, 1.0 = perfectly separable) ---")
    pairs = [("small", "ghost"), ("large", "ghost"), ("small", "large")]
    if args.label_mode == "gt":
        pairs += [("bg", "ghost"), ("small", "mixed")]
    hdr = f"{'variant':>9s} |" + "".join(f" {a + ' vs ' + b:>16s} |" for a, b in pairs)
    print(hdr[:-2])
    row = f"{'mass':>9s} |"
    for a, b in pairs:
        row += f" {auc_rank(m[masks[a]], m[masks[b]]):16.4f} |"
    print(row[:-2] + "   (coverage baseline)")
    for name, c in variants.items():
        row = f"{name:>9s} |"
        for a, b in pairs:
            row += f" {auc_rank(c[masks[a]], c[masks[b]]):16.4f} |"
        print(row[:-2])
    print("(small vs large ~ 0.5 means size-invariant confidence)")

    print(f"\n--- 3. evidence log-ratio (beta={args.beta}) and gate at the working point ---")
    hdr = (f"{'variant':>9s} | {'AUC s|g':>8s} |"
           + "".join(f" {'r>p ' + k:>12s} |" for k in masks)
           + f" {'g_small med':>11s} | {'g_ghost med':>11s}")
    print(hdr)
    print("-" * len(hdr))
    for name, c in variants.items():
        log_r = args.beta * np.log(m + EPS) + (1.0 - args.beta) * np.log(c + EPS)
        lr = log_r - math.log(p + EPS)
        g = 1.0 / (1.0 + np.exp(-lr / args.tau))
        row = f"{name:>9s} | {auc_rank(lr[masks['small']], lr[masks['ghost']]):8.4f} |"
        for mask in masks.values():
            row += (f" {float((lr[mask] > 0).mean()):12.1%} |" if mask.sum()
                    else f" {'---':>12s} |")
        row += (f" {np.median(g[masks['small']]) if masks['small'].sum() else float('nan'):11.3f} |"
                f" {np.median(g[masks['ghost']]) if masks['ghost'].sum() else float('nan'):11.3f}")
        print(row)

    if args.label_mode == "gt":
        print("\n--- 4. GT label vs wins-based group (what the wins method would have said) ---")
        small_hi = args.small_frac * f_total
        wins_group = np.full(wins.shape, LARGE)
        wins_group[wins <= small_hi] = SMALL
        wins_group[wins == 0] = GHOST
        corner = "GT \\ wins"
        hdr = f"{corner:>10s} |" + "".join(
            f" {LABEL_NAMES[g]:>8s} |" for g in (GHOST, SMALL, LARGE))
        print(hdr[:-2])
        for g_gt in group_ids:
            row = f"{LABEL_NAMES[g_gt]:>10s} |"
            for g_w in (GHOST, SMALL, LARGE):
                row += f" {int(((labels == g_gt) & (wins_group == g_w)).sum()):8d} |"
            print(row[:-2])
        print("(wins-'small' rows under bg/mixed = junk the wins method mislabels as objects)")

    # ---- plots ----
    if args.out:
        colors = {"ghost": "tab:red", "small": "tab:green", "large": "tab:blue",
                  "bg": "tab:gray", "mixed": "tab:orange"}
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        for ax, (name, c) in zip(axes.flat, variants.items()):
            for gname, mask in masks.items():
                v = c[mask]
                if v.size:
                    ax.hist(v, bins=80, alpha=0.5, density=True, label=gname,
                            color=colors[gname])
            ax.set_xlabel(name)
            auc = auc_rank(c[masks["small"]], c[masks["ghost"]])
            ax.set_title(f"{name} by slot group (small|ghost AUC {auc:.3f})")
            ax.legend()
        ax = axes[1, 1]
        for name, c, ls in (("entropy", c_ent, "-"), ("pur_sharp", pur_sharp, "--")):
            log_r = args.beta * np.log(m + EPS) + (1.0 - args.beta) * np.log(c + EPS)
            lr = log_r - math.log(p + EPS)
            for gname, mask in masks.items():
                v = lr[mask]
                if v.size:
                    ax.hist(v, bins=100, alpha=0.35, density=True, histtype="step",
                            ls=ls, label=f"{gname} ({name})", color=colors[gname])
        ax.axvline(0.0, color="k", ls="--", lw=1)
        ax.set_xlabel(f"log(r/p)  (beta={args.beta}, p={p:.4f}; solid=entropy, dashed=pur_sharp)")
        ax.set_title("evidence log-ratio by group")
        ax.legend(fontsize=7)
        fig.suptitle(
            f"purity probe on {os.path.basename(args.ckpt)} "
            f"({args.label_mode} labels, {n_clips} val clips)"
        )
        fig.tight_layout()
        fig.savefig(args.out, dpi=130)
        print(f"\nsaved plot: {args.out}")


if __name__ == "__main__":
    main()
