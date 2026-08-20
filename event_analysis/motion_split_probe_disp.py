"""Displacement-field escalation of motion_split_probe.py (the magnitude probe failed:
var AUC 0.554, split-gain AUC 0.459, 42.6% articulated false positives).

Instead of the feature-change magnitude ||dh||, estimate an actual 2-D patch displacement
field from frozen-feature correspondence and test whether its WITHIN-SLOT spread separates
merged slots from single-object slots:

  A(f, f')  = cos(h_t(f), h_{t+1}(f')) / tau        restricted to |pos(f') - pos(f)| <= R
              (locality kills the identical-twin ambiguity: an instance patch would
               otherwise match its lookalike far away and corrupt the displacement)
  d(f)      = sum_f' softmax_f'(A) pos(f') - pos(f)   then centered by the frame median
  score_dvar  = E_{w_s}||d - mu_s||^2 / E_scene||d - d_med||^2
  score_dgain = split gain of the per-slot principal-axis projection of d (>= 5% weight
                per side), the robust bimodality variant

Categories identical to motion_split_probe.py (GT on patch grid vs hard decoder masks).

Usage (inside the slotcurri container):
  python event_analysis/motion_split_probe_disp.py --data-dir /workspace/dataset
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from slotcurri import configuration, data, models

from motion_split_probe import auc_rank, seg_to_patch_ids, split_gain

EPS = 1e-8


def displacement_field(h_t, h_t1, grid, radius, tau):
    """Local feature-correspondence displacement. h_*: (N, F, D) -> d: (N, F, 2)."""
    n, f, _ = h_t.shape
    ht = torch.nn.functional.normalize(h_t.float(), dim=-1)
    ht1 = torch.nn.functional.normalize(h_t1.float(), dim=-1)
    sim = torch.bmm(ht, ht1.transpose(1, 2)) / tau  # (N, F, F)

    ys, xs = torch.meshgrid(
        torch.arange(grid, device=h_t.device),
        torch.arange(grid, device=h_t.device),
        indexing="ij",
    )
    pos = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1).float()  # (F, 2)
    dist2 = (pos.unsqueeze(1) - pos.unsqueeze(0)).pow(2).sum(-1)  # (F, F)
    local = dist2 <= radius * radius
    sim = sim.masked_fill(~local.unsqueeze(0), float("-inf"))
    T = sim.softmax(dim=-1)  # (N, F, F)

    exp_pos = torch.bmm(T, pos.unsqueeze(0).expand(n, -1, -1))  # (N, F, 2)
    d = exp_pos - pos.unsqueeze(0)
    d = d - d.median(dim=1, keepdim=True).values  # camera-motion compensation
    return d


def principal_projection(d, w):
    """Project d (N, F, 2) onto each slot's weighted principal axis. w: (N, S, F).

    Returns scalars (N, S, F) suitable for the 1-D split-gain statistic.
    """
    mu = torch.einsum("nsf,nfc->nsc", w, d)  # (N, S, 2)
    dc = d.unsqueeze(1) - mu.unsqueeze(2)  # (N, S, F, 2)
    cov = torch.einsum("nsf,nsfc,nsfe->nsce", w, dc, dc)  # (N, S, 2, 2)
    a, b, c = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
    # analytic principal eigenvector of a symmetric 2x2 matrix
    lam = 0.5 * (a + c + ((a - c).pow(2) + 4 * b.pow(2)).clamp_min(0).sqrt())
    vx, vy = lam - c, b
    norm = (vx.pow(2) + vy.pow(2)).sqrt().clamp_min(EPS)
    fallback = (b.abs() < 1e-12) & (a >= c)
    vx = torch.where(fallback, torch.ones_like(vx), vx / norm)
    vy = torch.where(fallback, torch.zeros_like(vy), vy / norm)
    proj = dc[..., 0] * vx.unsqueeze(-1) + dc[..., 1] * vy.unsqueeze(-1)  # (N, S, F)
    return proj


@torch.no_grad()
def collect(model, loader, max_clips, device, mass_gamma, radius, tau):
    rows = {k: [] for k in ("score_dvar", "score_dgain", "category", "clip")}
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

        att_sharp = att.pow(mass_gamma)
        att_sharp = att_sharp / att_sharp.sum(dim=2, keepdim=True).clamp_min(EPS)
        w = att_sharp / att_sharp.sum(dim=-1, keepdim=True).clamp_min(EPS)
        w_t = w[:, :-1].reshape(b * (t - 1), s, f)

        # displacement per frame pair, chunked to bound the (N, F, F) memory
        h_t = h[:, :-1].reshape(b * (t - 1), f, -1)
        h_t1 = h[:, 1:].reshape(b * (t - 1), f, -1)
        d_chunks = []
        for i in range(0, h_t.shape[0], 8):
            d_chunks.append(
                displacement_field(h_t[i : i + 8], h_t1[i : i + 8], grid, radius, tau)
            )
        d = torch.cat(d_chunks)  # (N, F, 2)

        scene_var = d.pow(2).sum(-1).mean(-1)  # (N,)  (d is median-centered)
        mu = torch.einsum("nsf,nfc->nsc", w_t, d)
        dev2 = (d.unsqueeze(1) - mu.unsqueeze(2)).pow(2).sum(-1)  # (N, S, F)
        var = (w_t * dev2).sum(-1)  # (N, S)
        score_dvar = var / scene_var.unsqueeze(-1).clamp_min(EPS)

        proj = principal_projection(d, w_t)  # (N, S, F)
        ns = proj.shape[0] * proj.shape[1]
        score_dgain = split_gain(
            proj.reshape(ns, f), w_t.reshape(ns, 1, f)
        ).reshape(proj.shape[0], proj.shape[1])

        gt = seg_to_patch_ids(seg, grid)
        winner = dec_masks.argmax(dim=2)
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
                        cat[bi, ti, si] = 2
                    elif int(counts[0]) >= 0.6 * n_terr and (
                        ids.numel() == 1 or int(counts[1]) < 8
                    ):
                        cat[bi, ti, si] = 1

        keep = (scene_var > 1e-4).cpu()
        skipped_static += int((~keep).sum())
        rows["score_dvar"].append(score_dvar.cpu()[keep])
        rows["score_dgain"].append(score_dgain.cpu()[keep])
        rows["category"].append(cat.reshape(-1, s)[keep])
        rows["clip"].append(torch.full((int(keep.sum()), s), n_clips, dtype=torch.long))

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
    ap.add_argument("--radius", type=float, default=6.0)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--cache", default="event_analysis/motion_split_disp_v26.npz")
    ap.add_argument("--out", default="event_analysis/motion_split_disp_v26.png")
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
                       args.mass_gamma, args.radius, args.tau)
        if args.cache:
            np.savez_compressed(args.cache, **blob)
            print(f"cached stats to {args.cache}")

    sv, sg = blob["score_dvar"], blob["score_dgain"]
    cat = blob["category"]
    print(f"clips={int(blob['n_clips'])}  slot-frames={sv.size}  "
          f"low-motion frame-pairs skipped={int(blob['skipped_static'])}")

    merged, single = cat == 2, cat == 1
    print(f"category counts: merged={int(merged.sum())}  single={int(single.sum())}  "
          f"other={int((cat == 0).sum())}")

    print("\n--- statistic distributions (median [q10, q90]) ---")
    print(f"{'statistic':>10s} | {'merged':>24s} | {'single':>24s} | {'AUC':>6s}")
    for name, x in (("disp_var", sv), ("disp_gain", sg)):
        vm, vs = x[merged], x[single]
        auc = auc_rank(vm, vs)
        print(f"{name:>10s} | {np.median(vm):6.3f} [{np.quantile(vm, .1):6.3f},"
              f" {np.quantile(vm, .9):6.3f}] | {np.median(vs):6.3f}"
              f" [{np.quantile(vs, .1):6.3f}, {np.quantile(vs, .9):6.3f}] | {auc:.4f}")

    for name, x in (("disp_var", sv), ("disp_gain", sg)):
        thr = np.median(x[merged])
        fp = float((x[single] > thr).mean()) if single.any() else float("nan")
        print(f"{name}: singles above merged-median threshold: {fp:.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (name, x) in zip(axes, (("disp_var (within-slot / scene)", sv),
                                    ("disp_gain (principal-axis bimodality)", sg))):
        for gname, mask, color in (("merged", merged, "tab:red"),
                                   ("single", single, "tab:green")):
            v = x[mask]
            if v.size:
                hi = np.quantile(x[merged | single], 0.99)
                ax.hist(np.clip(v, 0, hi), bins=80, alpha=0.55, density=True,
                        label=f"{gname} (n={v.size})", color=color)
        ax.set_xlabel(name)
        ax.legend()
    fig.suptitle(f"displacement-field motion contrast on v26 @ 100k  "
                 f"(AUC var={auc_rank(sv[merged], sv[single]):.3f}, "
                 f"gain={auc_rank(sg[merged], sg[single]):.3f})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved plot: {args.out}")


if __name__ == "__main__":
    main()
