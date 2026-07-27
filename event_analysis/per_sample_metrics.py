#!/usr/bin/env python3
"""Per-sample / per-frame ARI & MBO for viz comparison clips."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from slotcurri import configuration, data, models
from slotcurri.metrics import adjusted_rand_index


def load_model(settings, ckpt, device):
    cfg = configuration.load_config(settings)
    cfg.model.visualize = False
    m = models.build(cfg.model, cfg.optimizer)
    m.load_weights_from_checkpoint(ckpt)
    m.to(device).eval()
    return m


def hard_onehot(masks: torch.Tensor) -> torch.Tensor:
    if masks.dtype == torch.bool:
        return masks.float()
    idx = masks.argmax(dim=2)
    s = masks.shape[2]
    return F.one_hot(idx, num_classes=s).permute(0, 1, 4, 2, 3).float()


@torch.no_grad()
def predict(model, batch, device):
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model.forward(batch, train=False, cycle=True)
    aux = model.aux_forward(batch, out)
    key = "decoder_masks_vis_hard" if "decoder_masks_vis_hard" in aux else "decoder_masks_hard"
    return aux[key]


def resize_pred(pred, spatial):
    if pred.shape[-2:] == spatial:
        return pred
    b, t, s, h, w = pred.shape
    p = pred.float().reshape(b * t, s, h, w)
    p = F.interpolate(p, size=spatial, mode="nearest")
    return p.reshape(b, t, s, *spatial)


def ari_per_frame(true_btnchw, pred_btnchw):
    pred_btnchw = resize_pred(pred_btnchw, true_btnchw.shape[-2:])
    pred = hard_onehot(pred_btnchw)
    true = true_btnchw.float()
    b, t, c, h, w = true.shape
    s = pred.shape[2]
    true_f = true.reshape(b * t, c, h * w).permute(0, 2, 1)
    pred_f = pred.reshape(b * t, s, h * w).permute(0, 2, 1)
    vals = adjusted_rand_index(true_f, pred_f)
    nonempty = true_f.sum(dim=(1, 2)) > 0
    return vals, nonempty


def mbo_per_frame(true_btnchw, pred_btnchw):
    pred_btnchw = resize_pred(pred_btnchw, true_btnchw.shape[-2:])
    pred = hard_onehot(pred_btnchw) > 0.5
    true = true_btnchw > 0.5
    _, t, c, _, _ = true.shape
    s = pred.shape[2]
    out = []
    for ti in range(t):
        ious = []
        for ci in range(c):
            gt = true[0, ti, ci]
            if gt.sum() == 0:
                continue
            best = 0.0
            for si in range(s):
                pr = pred[0, ti, si]
                inter = (gt & pr).sum().float()
                union = (gt | pr).sum().float().clamp(min=1)
                best = max(best, (inter / union).item())
            ious.append(best)
        out.append(sum(ious) / len(ious) if ious else float("nan"))
    return out


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    runs = [
        (
            "baseline",
            "logs/_ytvis/settings/slotcurri/settings.yaml",
            "logs/_ytvis/checkpoints/slotcurri_step=step=100000-v1.ckpt",
        ),
        (
            "v7",
            "logs/_ytvis_attnmass_v7/settings/slotcurri/settings.yaml",
            "logs/_ytvis_attnmass_v7/checkpoints/slotcurri_step=step=81000.ckpt",
        ),
        (
            "v8",
            "logs/_ytvis_attnmass_v8/settings/slotcurri/settings.yaml",
            "logs/_ytvis_attnmass_v8/checkpoints/slotcurri_step=step=81000.ckpt",
        ),
    ]
    print("loading models...")
    models_l = [(n, load_model(s, c, device)) for n, s, c in runs]

    cfg = configuration.load_config(runs[0][1])
    cfg.dataset.num_val_workers = 0
    cfg.dataset.val_batch_size = 1
    dm = data.build(cfg.dataset, data_dir="/workspace/dataset")
    dm.setup("fit")
    loader = dm.val_dataloader()

    results = []
    for si, batch in enumerate(loader):
        if si >= 6:
            break
        seg = batch["segmentations"]
        tlen = seg.shape[1]
        mid = tlen // 2
        labeled = (seg[0].sum(1) > 0).float()
        lab_frac = labeled.mean(dim=(1, 2))
        n_inst_mid = int(seg[0, mid].flatten(1).any(1).sum().item())

        preds = {name: predict(m, batch, device).cpu() for name, m in models_l}

        print(
            f"\n===== sample{si:02d}  T={tlen}  "
            f"lab_frac_mid={lab_frac[mid]:.3f} mean={lab_frac.mean():.3f}  "
            f"n_inst_mid={n_inst_mid} ====="
        )
        row = {"sample": si}
        for name in ["baseline", "v7", "v8"]:
            ari_f, nonempty = ari_per_frame(seg, preds[name])
            mbo_f = mbo_per_frame(seg, preds[name])
            ari_vid = ari_f[nonempty].mean().item() if nonempty.any() else float("nan")
            valid_m = [x for x in mbo_f if not math.isnan(x)]
            mbo_vid = sum(valid_m) / len(valid_m) if valid_m else float("nan")
            ari_mid = ari_f[mid].item() if nonempty[mid] else float("nan")
            mbo_mid = mbo_f[mid]
            print(
                f"  {name:8s}  mid: ARI={ari_mid:.3f} MBO={mbo_mid:.3f} | "
                f"video-mean: ARI={ari_vid:.3f} MBO={mbo_vid:.3f} | "
                f"labeled_frames={int(nonempty.sum())}/{tlen}"
            )
            for ti in (0, mid, tlen - 1):
                a = ari_f[ti].item() if nonempty[ti] else float("nan")
                print(f"           t{ti:02d}: ARI={a:.3f} MBO={mbo_f[ti]:.3f}")
            if nonempty.any():
                vals = ari_f.clone()
                vals[~nonempty] = -1
                bi = int(vals.argmax())
                wi_vals = ari_f.clone()
                wi_vals[~nonempty] = 2.0
                wi = int(wi_vals.argmin())
                print(f"           best_t{bi:02d} ARI={ari_f[bi]:.3f} | worst_t{wi:02d} ARI={ari_f[wi]:.3f}")
            row[f"{name}_ari_mid"] = ari_mid
            row[f"{name}_mbo_mid"] = mbo_mid
            row[f"{name}_ari_vid"] = ari_vid
            row[f"{name}_mbo_vid"] = mbo_vid
        results.append(row)

    print("\n===== MID FRAME (matches *_mid.png) =====")
    print(f"{'sample':>7} {'bARI':>7} {'v7ARI':>7} {'v8ARI':>7} {'bMBO':>7} {'v7MBO':>7} {'v8MBO':>7}")
    for r in results:
        print(
            f"{r['sample']:7d} {r['baseline_ari_mid']:7.3f} {r['v7_ari_mid']:7.3f} {r['v8_ari_mid']:7.3f} "
            f"{r['baseline_mbo_mid']:7.3f} {r['v7_mbo_mid']:7.3f} {r['v8_mbo_mid']:7.3f}"
        )

    print("\n===== VIDEO MEAN (labeled frames only) =====")
    print(f"{'sample':>7} {'bARI':>7} {'v7ARI':>7} {'v8ARI':>7} {'bMBO':>7} {'v7MBO':>7} {'v8MBO':>7}")
    for r in results:
        print(
            f"{r['sample']:7d} {r['baseline_ari_vid']:7.3f} {r['v7_ari_vid']:7.3f} {r['v8_ari_vid']:7.3f} "
            f"{r['baseline_mbo_vid']:7.3f} {r['v7_mbo_vid']:7.3f} {r['v8_mbo_vid']:7.3f}"
        )


if __name__ == "__main__":
    main()
