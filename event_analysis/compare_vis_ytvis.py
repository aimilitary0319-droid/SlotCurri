#!/usr/bin/env python3
"""Side-by-side mask visualizations: baseline vs attn-mass checkpoints on a few val clips."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from slotcurri import configuration, data, metrics as metric_lib, models
from slotcurri.data.transforms import Denormalize
from slotcurri.visualizations import mix_videos_with_masks


def build_val_metrics():
    """Same metric setup as ytvis2021_* configs (ignore_background=false)."""
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
def score_sample(model, batch, device) -> dict[str, float]:
    batch_dev = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
    }
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
    return scores


def load_model(settings_yaml: str, ckpt: str, device: torch.device):
    config = configuration.load_config(settings_yaml)
    # eval-style: no training viz spam
    config.model.visualize = False
    model = models.build(config.model, config.optimizer)
    model.load_weights_from_checkpoint(ckpt)
    model.to(device)
    model.eval()
    return model, config


def one_hot_segmentations(seg: torch.Tensor, max_classes: int = 25) -> torch.Tensor:
    """(B,T,H,W) int -> (B,T,C,H,W) bool one-hot for overlay."""
    b, t, h, w = seg.shape
    # ignore background 0 for cleaner overlay? keep all including bg as class 0
    oh = torch.zeros(b, t, max_classes, h, w, dtype=torch.bool, device=seg.device)
    for c in range(max_classes):
        oh[:, :, c] = seg == c
    # drop empty classes later in mix by still drawing — ok
    return oh


@torch.no_grad()
def predict_masks(model, batch, device):
    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    outputs = model.forward(batch, train=False, cycle=True)
    aux = model.aux_forward(batch, outputs)
    # hard decoder masks at video resolution if available
    key = "decoder_masks_vis_hard" if "decoder_masks_vis_hard" in aux else "decoder_masks_hard"
    masks = aux[key]
    return masks.cpu()


def to_uint8_video(video: torch.Tensor) -> torch.Tensor:
    denorm = Denormalize(input_type="video")
    vid = denorm(video[0].cpu()).clamp(0, 1)  # T,C,H,W
    return vid.unsqueeze(0)


def overlay(video_btchw: torch.Tensor, masks_btnchw: torch.Tensor) -> np.ndarray:
    """Return THWC uint8 frames."""
    mixed = mix_videos_with_masks(video_btchw, masks_btnchw.float(), alpha=0.45)
    # mixed: B,T,C,H,W uint8-ish
    frames = mixed[0].permute(0, 2, 3, 1).cpu().numpy()
    return frames.astype(np.uint8)


def hstack_frames(*frame_lists, labels=None) -> list[np.ndarray]:
    out = []
    n = min(len(f) for f in frame_lists)
    for i in range(n):
        parts = [f[i] for f in frame_lists]
        if labels:
            labeled = []
            for p, lab in zip(parts, labels):
                img = Image.fromarray(p)
                # simple top bar
                bar = Image.new("RGB", (img.width, 28), (20, 20, 20))
                from PIL import ImageDraw, ImageFont

                draw = ImageDraw.Draw(bar)
                draw.text((8, 6), lab, fill=(240, 240, 240))
                canvas = Image.new("RGB", (img.width, img.height + 28))
                canvas.paste(bar, (0, 0))
                canvas.paste(img, (0, 28))
                labeled.append(np.asarray(canvas))
            parts = labeled
        out.append(np.concatenate(parts, axis=1))
    return out


def _known_runs(root: Path) -> dict[str, tuple[str, str]]:
    """name -> (settings.yaml, ckpt)."""
    return {
        "baseline": (
            str(root / "logs/_ytvis/settings/slotcurri/settings.yaml"),
            str(root / "logs/_ytvis/checkpoints/slotcurri_step=step=100000-v1.ckpt"),
        ),
        "v7": (
            str(root / "logs/_ytvis_attnmass_v7/settings/slotcurri/settings.yaml"),
            str(root / "logs/_ytvis_attnmass_v7/checkpoints/slotcurri_step=step=100000-v1.ckpt"),
        ),
        "v8": (
            str(root / "logs/_ytvis_attnmass_v8/settings/slotcurri/settings.yaml"),
            str(root / "logs/_ytvis_attnmass_v8/checkpoints/slotcurri_step=step=100000-v1.ckpt"),
        ),
        "v10": (
            str(root / "logs/_ytvis_attnmass_v10/settings/slotcurri/settings.yaml"),
            str(root / "logs/_ytvis_attnmass_v10/checkpoints/slotcurri_step=step=100000-v1.ckpt"),
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--out-dir", default="logs/vis_compare_baseline_v7_v8")
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--frame-stride", type=int, default=2, help="keep every Nth frame in GIF/PNG strip")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-v8", action="store_true")
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to compare, e.g. baseline v10. Default: baseline v7 [v8].",
    )
    args = ap.parse_args()

    root = Path("/workspace/SlotCurri")
    if not (root / "slotcurri").exists():
        root = Path("/mnt/ssd2/hmlee/SlotCurri")

    catalog = _known_runs(root)
    if args.models:
        names = list(args.models)
    else:
        names = ["baseline", "v7"]
        if not args.skip_v8:
            names.append("v8")
    runs = []
    for name in names:
        if name not in catalog:
            raise SystemExit(f"unknown model {name!r}; known: {sorted(catalog)}")
        settings, ckpt = catalog[name]
        runs.append((name, settings, ckpt))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # datamodule from baseline settings (same val shards)
    base_cfg = configuration.load_config(runs[0][1])
    # lighter loader
    base_cfg.dataset.num_val_workers = 0
    base_cfg.dataset.val_batch_size = 1
    dm = data.build(base_cfg.dataset, data_dir=args.data_dir)
    dm.setup("fit")
    loader = dm.val_dataloader()

    print("Loading models...")
    loaded = []
    for name, settings, ckpt in runs:
        assert Path(ckpt).exists(), ckpt
        print(f"  {name}: {ckpt}")
        m, _ = load_model(settings, ckpt, device)
        loaded.append((name, m))

    rows = []
    for si, batch in enumerate(loader):
        if si >= args.num_samples:
            break
        print(f"sample {si}...")
        video = to_uint8_video(batch["video"])  # 1,T,C,H,W float 0-1 after denorm path inside overlay

        overlays = {}
        # GT: already one-hot bool (B,T,C,H,W) on this datamodule
        if "segmentations" in batch:
            seg = batch["segmentations"].cpu()
            if seg.ndim == 4:  # (B,T,H,W) class ids
                ncls = int(seg.max().item()) + 1
                gt = one_hot_segmentations(seg, max_classes=max(ncls, 2))
            else:
                gt = seg.bool()
            if gt.shape[-2:] != video.shape[-2:]:
                bt, tt, cc, _, _ = gt.shape
                gt_f = gt.float().reshape(bt * tt, cc, gt.shape[-2], gt.shape[-1])
                gt_f = torch.nn.functional.interpolate(gt_f, size=video.shape[-2:], mode="nearest")
                gt = gt_f.reshape(bt, tt, cc, video.shape[-2], video.shape[-1]).bool()
            overlays["gt"] = overlay(video, gt)

        sample_scores = {"sample": si}
        for name, model in loaded:
            masks = predict_masks(model, batch, device)
            if masks.ndim == 4:
                # B,T,H,W class ids? unlikely
                pass
            # expect B,T,S,H,W
            if masks.shape[-2:] != video.shape[-2:]:
                b, t, s, h, w = masks.shape
                m = masks.float().reshape(b * t, s, h, w)
                m = torch.nn.functional.interpolate(m, size=video.shape[-2:], mode="nearest")
                masks = m.reshape(b, t, s, *video.shape[-2:]).bool()
            else:
                masks = masks.bool()
            overlays[name] = overlay(video, masks)

            scores = score_sample(model, batch, device)
            for mk, mv in scores.items():
                sample_scores[f"{name}_{mk}"] = mv
            print(
                f"  {name}: ARI={scores['ari']:.4f} mBO={scores['mbo']:.4f} "
                f"iARI={scores['image_ari']:.4f} iMBO={scores['image_mbo']:.4f}"
            )

        rows.append(sample_scores)

        order = ["gt"] + [n for n, _ in loaded] if "gt" in overlays else [n for n, _ in loaded]
        # Put per-model ARI/mBO into panel labels
        labels = []
        for k in order:
            if k == "gt":
                labels.append("gt")
            else:
                a = sample_scores.get(f"{k}_ari")
                m = sample_scores.get(f"{k}_mbo")
                labels.append(f"{k}  ARI={a:.3f} mBO={m:.3f}" if a is not None else k)
        frames = hstack_frames(*[overlays[k] for k in order], labels=labels)
        frames = frames[:: max(args.frame_stride, 1)]

        # save middle frame PNG + GIF
        mid = frames[len(frames) // 2]
        Image.fromarray(mid).save(out_dir / f"sample{si:02d}_mid.png")
        # strip of a few frames vertically optional: save gif
        try:
            import imageio

            imageio.mimsave(out_dir / f"sample{si:02d}.gif", frames, fps=4)
        except Exception as e:
            print("gif skip:", e)
            for fi, fr in enumerate(frames[:4]):
                Image.fromarray(fr).save(out_dir / f"sample{si:02d}_f{fi}.png")

        print(f"  wrote {out_dir / f'sample{si:02d}_mid.png'}")

    # Persist per-sample metrics (same protocol as val/*)
    if rows:
        import csv

        keys = list(rows[0].keys())
        csv_path = out_dir / "per_sample_metrics.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("metrics ->", csv_path)
        # mean over visualized samples
        model_names = [n for n, _ in loaded]
        for name in model_names:
            for mk in ("ari", "mbo", "image_ari", "image_mbo"):
                vals = [r[f"{name}_{mk}"] for r in rows]
                print(f"mean {name}/{mk}: {sum(vals)/len(vals):.4f}")

    print("done ->", out_dir)


if __name__ == "__main__":
    main()
