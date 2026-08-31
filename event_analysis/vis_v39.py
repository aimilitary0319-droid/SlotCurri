#!/usr/bin/env python3
"""Visualize v39 checkpoints at eval (raw Keys; curriculum off).

Per-sample panels: RGB | GT | decoder overlay, plus a per-slot grid labeled
with π and π/Σπ. Reuses helpers from vis_movi_c_v29.py.

Usage (slotcurri image):
  python event_analysis/vis_v39.py --run ytvis
  python event_analysis/vis_v39.py --run movi
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slotcurri import configuration, data, metrics as metric_lib
from vis_movi_c_v29 import (
    add_bar,
    heatmap_png,
    hstack,
    load_model,
    overlay,
    resize_masks,
    rgb_frames,
    run_sample,
    save_gif,
    slot_gt_stats,
    slot_grid,
    to_uint8_video,
)

RUNS = {
    "ytvis": {
        "settings": "logs/_ytvis_attnmass_v39/settings/slotcurri/settings.yaml",
        "ckpt_dir": "logs/_ytvis_attnmass_v39/checkpoints",
        "out_dir": "logs/vis_v39_ytvis",
        "ignore_background": False,
        "label": "YT-VIS v39",
    },
    "movi": {
        "settings": "logs/_movi_c_attnmass_v39/settings/slotcurri/settings.yaml",
        "ckpt_dir": "logs/_movi_c_attnmass_v39/checkpoints",
        "out_dir": "logs/vis_v39_movi",
        "ignore_background": True,
        "label": "MOVi-C v39",
    },
}


def latest_ckpt(ckpt_dir: Path) -> Path:
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints in {ckpt_dir}")
    return ckpts[-1]


def build_val_metrics(ignore_background: bool):
    kw = dict(pred_key="decoder_masks_hard", true_key="segmentations")
    return {
        "ari": metric_lib.VideoARI(ignore_background=ignore_background, **kw),
        "image_ari": metric_lib.ImageARI(
            video_input=True, ignore_background=ignore_background, **kw
        ),
        "mbo": metric_lib.VideoIoU(
            matching="overlap", ignore_background=ignore_background, **kw
        ),
        "image_mbo": metric_lib.ImageIoU(
            matching="overlap",
            ignore_background=ignore_background,
            video_input=True,
            **kw,
        ),
    }


def annotate_pi_share(grid: np.ndarray, gate_ts, mid: int) -> np.ndarray:
    """Overwrite is already in slot_grid labels; add a π-share bar under the grid."""
    if gate_ts is None:
        return grid
    pi = np.asarray(gate_ts[mid], dtype=np.float64)
    s = float(pi.sum()) + 1e-8
    share = pi / s
    order = np.argsort(-share)
    bar_h = 22
    canvas = Image.new("RGB", (grid.shape[1], grid.shape[0] + bar_h), (18, 18, 18))
    canvas.paste(Image.fromarray(grid), (0, 0))
    from PIL import ImageDraw, ImageFont

    dr = ImageDraw.Draw(canvas)
    parts = [f"s{int(i)}={share[i]*100:.0f}%" for i in order[:6]]
    text = "π share  " + "  ".join(parts) + f"   Σπ={s:.3f}"
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13
        )
    except Exception:
        font = ImageFont.load_default()
    dr.text((8, grid.shape[0] + 4), text, fill=(240, 240, 240), font=font)
    return np.asarray(canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=("ytvis", "movi"), required=True)
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    spec = RUNS[args.run]
    root = Path("/workspace/SlotCurri")
    if not (root / "slotcurri").exists():
        root = Path("/mnt/ssd2/hmlee/SlotCurri")

    settings = root / spec["settings"]
    ckpt = Path(args.ckpt) if args.ckpt else latest_ckpt(root / spec["ckpt_dir"])
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    out_dir = root / (args.out_dir or spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.30, device.index or 0)

    print(f"{spec['label']}  device={device}  ckpt={ckpt.name}")
    print("eval mode: raw Keys (feature curriculum off)")

    model, _config = load_model(str(settings), str(ckpt), device)
    cycle = bool(getattr(model, "cyclic_inference", False))

    # Swap metrics to the dataset's ignore_background (YT false, MOVi true).
    import vis_movi_c_v29 as v29

    v29.build_val_metrics = lambda: build_val_metrics(spec["ignore_background"])

    cfg = configuration.load_config(str(settings))
    cfg.dataset.num_val_workers = 0
    cfg.dataset.val_batch_size = 1
    dm = data.build(cfg.dataset, data_dir=args.data_dir)
    dm.setup("fit")
    loader = dm.val_dataloader()

    rows = []
    for si, batch in enumerate(loader):
        if si >= args.num_samples:
            break
        print(f"\n=== sample {si} ===")
        scores, masks, gate, conf, mass = run_sample(model, batch, device, cycle)
        print(
            f"  ARI={scores['ari']:.4f} mBO={scores['mbo']:.4f} "
            f"iARI={scores['image_ari']:.4f} iMBO={scores['image_mbo']:.4f}"
        )
        if gate is not None:
            pi = gate[0].mean(axis=0)
            sm = float(pi.sum()) + 1e-8
            order = np.argsort(-pi)
            shares = ", ".join(f"s{int(i)}={pi[i]/sm*100:.0f}%" for i in order[:5])
            print(f"  mean-π share: {shares}  Σπ={sm:.3f}")

        video = to_uint8_video(batch["video"])
        hw = video.shape[-2:]
        masks = resize_masks(masks, hw).bool()
        pred = masks[0].numpy()

        gt = batch["segmentations"].cpu()
        if gt.ndim == 4:
            ncls = int(gt.max().item()) + 1
            b, t, h, w = gt.shape
            oh = torch.zeros(b, t, ncls, h, w, dtype=torch.bool)
            for c in range(ncls):
                oh[:, :, c] = gt == c
            gt = oh
        gt = resize_masks(gt, hw).bool()
        gt_np = gt[0].numpy()
        stats = slot_gt_stats(pred, gt_np)

        rgb = rgb_frames(video)
        ov_gt = overlay(video, gt)
        ov_pr = overlay(video, masks)
        frames = hstack(
            [rgb, ov_gt, ov_pr],
            [
                "rgb",
                "gt",
                f"v39 pred  ARI={scores['ari']:.3f} mBO={scores['mbo']:.3f}",
            ],
        )
        frames = frames[:: max(args.frame_stride, 1)]
        Image.fromarray(frames[len(frames) // 2]).save(out_dir / f"sample{si:02d}_mid.png")
        save_gif(out_dir / f"sample{si:02d}.gif", frames)

        g_ts = gate[0] if gate is not None else None
        c_ts = conf[0] if conf is not None else None
        m_ts = mass[0] if mass is not None else None
        grid = slot_grid(rgb, pred, g_ts, c_ts, m_ts, stats["slots"])
        mid = rgb.shape[0] // 2
        grid = annotate_pi_share(grid, g_ts, mid)
        Image.fromarray(grid).save(out_dir / f"sample{si:02d}_slots.png")

        s = pred.shape[1]
        c = gt_np.shape[1]
        ov = np.zeros((s, c), dtype=np.float32)
        for slot_i in range(s):
            pix = pred[:, slot_i].sum()
            if pix < 1:
                continue
            for ci in range(c):
                ov[slot_i, ci] = (pred[:, slot_i] & gt_np[:, ci]).sum() / pix
        heatmap_png(
            ov,
            [f"s{i}" for i in range(s)],
            ["bg" if i == 0 else f"o{i}" for i in range(c)],
            f"sample{si} slot pixel share of GT class",
            out_dir / f"sample{si:02d}_heatmap.png",
        )
        (out_dir / f"sample{si:02d}_stats.json").write_text(json.dumps(stats, indent=2))
        rows.append({"sample": si, **scores})

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rows:
        with open(out_dir / "per_sample_metrics.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        mean = {k: sum(r[k] for r in rows) / len(rows) for k in ("ari", "mbo")}
        print("\nmean ARI={:.4f} mBO={:.4f}".format(mean["ari"], mean["mbo"]))
        (out_dir / "summary.json").write_text(
            json.dumps({"ckpt": str(ckpt), "n": len(rows), "mean": mean}, indent=2)
        )
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
