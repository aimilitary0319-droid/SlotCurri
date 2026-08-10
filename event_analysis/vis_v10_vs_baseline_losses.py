#!/usr/bin/env python3
"""Scan full YTVIS val: find clips where v10 loses badly to baseline, visualize them."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from slotcurri import configuration, data, metrics as metric_lib, models
from slotcurri.data.transforms import Denormalize
from slotcurri.visualizations import mix_videos_with_masks


def load_model(settings_yaml: str, ckpt: str, device: torch.device):
    config = configuration.load_config(settings_yaml)
    config.model.visualize = False
    model = models.build(config.model, config.optimizer)
    model.load_weights_from_checkpoint(ckpt)
    model.to(device).eval()
    return model


def build_val_metrics():
    return {
        "ari": metric_lib.VideoARI(
            ignore_background=False, pred_key="decoder_masks_hard", true_key="segmentations"
        ),
        "image_ari": metric_lib.ImageARI(
            video_input=True,
            ignore_background=False,
            pred_key="decoder_masks_hard",
            true_key="segmentations",
        ),
        "mbo": metric_lib.VideoIoU(
            matching="overlap",
            ignore_background=False,
            pred_key="decoder_masks_hard",
            true_key="segmentations",
        ),
        "image_mbo": metric_lib.ImageIoU(
            matching="overlap",
            video_input=True,
            ignore_background=False,
            pred_key="decoder_masks_hard",
            true_key="segmentations",
        ),
    }


@torch.no_grad()
def score_and_masks(model, batch, device):
    batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    outputs = model.forward(batch_dev, train=False, cycle=True)
    aux = model.aux_forward(batch_dev, outputs)
    scores = {}
    for name, metric in build_val_metrics().items():
        metric = metric.to(device)
        metric.reset()
        metric.update(**batch_dev, **outputs, **aux)
        val = metric.compute()
        scores[name] = float(val.detach().cpu()) if torch.is_tensor(val) else float(val)
        metric.reset()
    key = "decoder_masks_vis_hard" if "decoder_masks_vis_hard" in aux else "decoder_masks_hard"
    masks = aux[key].cpu()
    return scores, masks


def one_hot_segmentations(seg: torch.Tensor, max_classes: int = 25) -> torch.Tensor:
    b, t, h, w = seg.shape
    oh = torch.zeros(b, t, max_classes, h, w, dtype=torch.bool, device=seg.device)
    for c in range(max_classes):
        oh[:, :, c] = seg == c
    return oh


def to_uint8_video(video: torch.Tensor) -> torch.Tensor:
    denorm = Denormalize(input_type="video")
    vid = denorm(video[0].cpu()).clamp(0, 1)
    return vid.unsqueeze(0)


def overlay(video_btchw: torch.Tensor, masks_btnchw: torch.Tensor) -> np.ndarray:
    mixed = mix_videos_with_masks(video_btchw, masks_btnchw.float(), alpha=0.45)
    frames = mixed[0].permute(0, 2, 3, 1).cpu().numpy()
    return frames.astype(np.uint8)


def prep_masks(masks: torch.Tensor, spatial) -> torch.Tensor:
    if masks.shape[-2:] != spatial:
        b, t, s, h, w = masks.shape
        m = masks.float().reshape(b * t, s, h, w)
        m = torch.nn.functional.interpolate(m, size=spatial, mode="nearest")
        masks = m.reshape(b, t, s, *spatial)
    return masks.bool()


def hstack_labeled(frame_lists, labels) -> list[np.ndarray]:
    out = []
    n = min(len(f) for f in frame_lists)
    for i in range(n):
        parts = []
        for p, lab in zip([f[i] for f in frame_lists], labels):
            img = Image.fromarray(p)
            bar = Image.new("RGB", (img.width, 36), (20, 20, 20))
            draw = ImageDraw.Draw(bar)
            draw.text((8, 10), lab, fill=(240, 240, 240))
            canvas = Image.new("RGB", (img.width, img.height + 36))
            canvas.paste(bar, (0, 0))
            canvas.paste(img, (0, 36))
            parts.append(np.asarray(canvas))
        out.append(np.concatenate(parts, axis=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--out-dir", default="logs/vis_v10_loses_to_baseline")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = full val")
    ap.add_argument("--top-k", type=int, default=20, help="visualize worst K by loss score")
    ap.add_argument(
        "--min-dari",
        type=float,
        default=0.05,
        help="also keep if baseline_ari - v10_ari >= this",
    )
    ap.add_argument(
        "--min-dmbo",
        type=float,
        default=0.05,
        help="also keep if baseline_mbo - v10_mbo >= this",
    )
    ap.add_argument("--frame-stride", type=int, default=2)
    args = ap.parse_args()

    root = Path("/workspace/SlotCurri")
    if not (root / "slotcurri").exists():
        root = Path("/mnt/ssd2/hmlee/SlotCurri")

    base_settings = root / "logs/_ytvis/settings/slotcurri/settings.yaml"
    base_ckpt = root / "logs/_ytvis/checkpoints/slotcurri_step=step=100000-v1.ckpt"
    v10_settings = root / "logs/_ytvis_attnmass_v10/settings/slotcurri/settings.yaml"
    v10_ckpt = root / "logs/_ytvis_attnmass_v10/checkpoints/slotcurri_step=step=100000-v1.ckpt"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = configuration.load_config(str(base_settings))
    cfg.dataset.num_val_workers = 0
    cfg.dataset.val_batch_size = 1
    dm = data.build(cfg.dataset, data_dir=args.data_dir)
    dm.setup("fit")
    loader = dm.val_dataloader()

    print("Loading models...")
    base = load_model(str(base_settings), str(base_ckpt), device)
    v10 = load_model(str(v10_settings), str(v10_ckpt), device)

    rows = []
    # keep CPU tensors for later viz of losers only (masks can be large — recompute for top-k)
    print("Scanning val...")
    for si, batch in enumerate(loader):
        if args.max_samples and si >= args.max_samples:
            break
        sb, _ = score_and_masks(base, batch, device)
        sv, _ = score_and_masks(v10, batch, device)
        dari = sb["ari"] - sv["ari"]
        dmbo = sb["mbo"] - sv["mbo"]
        # positive = baseline better / v10 loses
        loss_score = max(0.0, dari) + max(0.0, dmbo)
        row = {
            "sample": si,
            "baseline_ari": sb["ari"],
            "baseline_mbo": sb["mbo"],
            "baseline_image_ari": sb["image_ari"],
            "baseline_image_mbo": sb["image_mbo"],
            "v10_ari": sv["ari"],
            "v10_mbo": sv["mbo"],
            "v10_image_ari": sv["image_ari"],
            "v10_image_mbo": sv["image_mbo"],
            "d_ari": dari,
            "d_mbo": dmbo,
            "loss_score": loss_score,
            "v10_loses_ari": dari >= args.min_dari,
            "v10_loses_mbo": dmbo >= args.min_dmbo,
        }
        rows.append(row)
        if (si + 1) % 10 == 0:
            print(f"  scanned {si+1}  last loss_score={loss_score:.3f} dARI={dari:+.3f} dMBO={dmbo:+.3f}")

    csv_path = out_dir / "all_val_metrics.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", csv_path)

    # select losers: threshold OR top-k by loss_score among those with any positive loss
    thresh = [
        r
        for r in rows
        if r["v10_loses_ari"] or r["v10_loses_mbo"]
    ]
    thresh_sorted = sorted(thresh, key=lambda r: -r["loss_score"])
    # also take global top-k by loss_score (even if below threshold) to fill
    top_global = sorted(rows, key=lambda r: -r["loss_score"])
    chosen = []
    seen = set()
    for r in thresh_sorted + top_global:
        if r["loss_score"] <= 0:
            continue
        if r["sample"] in seen:
            continue
        chosen.append(r)
        seen.add(r["sample"])
        if len(chosen) >= args.top_k:
            break

    summary = {
        "n_val": len(rows),
        "n_lose_ari_ge": sum(1 for r in rows if r["v10_loses_ari"]),
        "n_lose_mbo_ge": sum(1 for r in rows if r["v10_loses_mbo"]),
        "n_lose_either": len(thresh),
        "mean_d_ari": float(np.mean([r["d_ari"] for r in rows])),
        "mean_d_mbo": float(np.mean([r["d_mbo"] for r in rows])),
        "chosen_samples": [r["sample"] for r in chosen],
        "thresholds": {"min_dari": args.min_dari, "min_dmbo": args.min_dmbo, "top_k": args.top_k},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    want = {r["sample"] for r in chosen}
    print(f"Visualizing {len(want)} loser clips:", sorted(want))

    # second pass for viz
    for si, batch in enumerate(loader):
        if args.max_samples and si >= args.max_samples:
            break
        if si not in want:
            continue
        meta = next(r for r in rows if r["sample"] == si)
        print(
            f"viz sample {si}: dARI={meta['d_ari']:+.3f} dMBO={meta['d_mbo']:+.3f} "
            f"loss={meta['loss_score']:.3f}"
        )
        _, masks_b = score_and_masks(base, batch, device)
        _, masks_v = score_and_masks(v10, batch, device)
        video = to_uint8_video(batch["video"])
        spatial = video.shape[-2:]

        overlays = {}
        if "segmentations" in batch:
            seg = batch["segmentations"].cpu()
            if seg.ndim == 4:
                ncls = int(seg.max().item()) + 1
                gt = one_hot_segmentations(seg, max_classes=max(ncls, 2))
            else:
                gt = seg.bool()
            gt = prep_masks(gt.float(), spatial)
            overlays["gt"] = overlay(video, gt)

        overlays["baseline"] = overlay(video, prep_masks(masks_b, spatial))
        overlays["v10"] = overlay(video, prep_masks(masks_v, spatial))

        labels = [
            "gt",
            f"baseline ARI={meta['baseline_ari']:.3f} mBO={meta['baseline_mbo']:.3f}",
            f"v10 ARI={meta['v10_ari']:.3f} mBO={meta['v10_mbo']:.3f}  "
            f"dARI={meta['d_ari']:+.3f} dMBO={meta['d_mbo']:+.3f}",
        ]
        order = ["gt", "baseline", "v10"]
        frames = hstack_labeled([overlays[k] for k in order], labels)
        frames = frames[:: max(args.frame_stride, 1)]
        mid = frames[len(frames) // 2]
        rank = sorted(want).index(si) if False else None
        # rank by loss among chosen
        rank_i = next(i for i, r in enumerate(chosen) if r["sample"] == si)
        stem = f"rank{rank_i:02d}_sample{si:03d}_dARI{meta['d_ari']:+.3f}_dMBO{meta['d_mbo']:+.3f}"
        Image.fromarray(mid).save(out_dir / f"{stem}_mid.png")
        try:
            import imageio

            imageio.mimsave(out_dir / f"{stem}.gif", frames, fps=4)
        except Exception as e:
            print("gif skip:", e)

    # compact table of chosen
    table_path = out_dir / "losers_table.csv"
    with open(table_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(chosen[0].keys()) if chosen else [])
        if chosen:
            w.writeheader()
            w.writerows(chosen)
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
