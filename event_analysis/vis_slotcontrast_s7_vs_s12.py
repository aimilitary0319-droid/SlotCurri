#!/usr/bin/env python3
"""Side-by-side SlotContrast YTVIS: 7 slots vs 12 slots.

Scan val, then write overlay clips (GT | s7 | s12) plus mid-frame per-slot
grids so extra-slot fragmentation is visible.

  python event_analysis/vis_slotcontrast_s7_vs_s12.py

Outputs under logs/vis_slotcontrast_s7_vs_s12/:
  all_val_metrics.csv, summary.json, scatter_ari_mbo.png
  typical/   first N val clips (unfiltered look)
  s7_wins/   clips where 7-slot is better
  s12_wins/  clips where 12-slot is better
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vis_v10_vs_baseline_losses import (
    hstack_labeled,
    load_model,
    one_hot_segmentations,
    overlay,
    prep_masks,
    project_root,
    save_clip,
    score_and_masks,
    to_uint8_video,
)
from slotcurri import configuration, data

TAB = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (174, 199, 232),
    (255, 187, 120),
    (152, 223, 138),
]


def _font(size: int = 13):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def occupied_slots(masks: torch.Tensor, min_frac: float = 0.005) -> int:
    """Count slots that own at least min_frac of pixels on some frame."""
    # masks: B,T,S,H,W
    pix = masks.float().flatten(3).sum(-1)  # B,T,S
    hw = float(masks.shape[-1] * masks.shape[-2])
    frac = pix / max(hw, 1.0)
    occ = (frac.max(dim=1).values > min_frac).sum(dim=-1)
    return int(occ[0].item())


def rgb_frames(video_01: torch.Tensor) -> np.ndarray:
    return (video_01[0].permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def tint_slot(rgb: np.ndarray, mask_hw: np.ndarray, color, alpha=0.62) -> np.ndarray:
    out = rgb.astype(np.float32)
    c = np.array(color, dtype=np.float32)
    m = mask_hw.astype(np.float32)[..., None]
    return (out * (1 - alpha * m) + c * (alpha * m)).astype(np.uint8)


def add_bar(frame: np.ndarray, text: str, color=(20, 20, 20)) -> np.ndarray:
    img = Image.fromarray(frame)
    bar_h = 24
    canvas = Image.new("RGB", (img.width, img.height + bar_h), color)
    canvas.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), text, fill=(240, 240, 240), font=_font(12))
    return np.asarray(canvas)


def slot_grid_mid(rgb: np.ndarray, pred_tshw: np.ndarray, title: str, vis_hw=(160, 160)) -> np.ndarray:
    """Mid-frame RGB + one tinted cell per slot."""
    t, s, h, w = pred_tshw.shape
    mid = t // 2
    th, tw = vis_hw
    frame = np.array(Image.fromarray(rgb[mid]).resize((tw, th), Image.BILINEAR))
    cells = [add_bar(frame, f"{title} rgb")]
    for si in range(s):
        m = pred_tshw[mid, si]
        m_img = np.array(Image.fromarray((m.astype(np.uint8) * 255)).resize((tw, th), Image.NEAREST))
        tinted = tint_slot(frame, m_img > 127, TAB[si % len(TAB)])
        pix = float(m_img.mean() / 255.0)
        cells.append(add_bar(tinted, f"s{si}  {pix * 100:.1f}%"))
    cols = 4
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(row[0]))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def write_scatter(rows, path: Path) -> None:
    s7_ari = np.array([r["s7_ari"] for r in rows])
    s12_ari = np.array([r["s12_ari"] for r in rows])
    s7_mbo = np.array([r["s7_mbo"] for r in rows])
    s12_mbo = np.array([r["s12_mbo"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, x, y, name in (
        (axes[0], s7_ari, s12_ari, "ARI"),
        (axes[1], s7_mbo, s12_mbo, "mBO"),
    ):
        ax.scatter(x, y, s=18, alpha=0.7, c="#1f77b4", edgecolors="none")
        lo = min(float(x.min()), float(y.min()))
        hi = max(float(x.max()), float(y.max()))
        pad = 0.03 * (hi - lo + 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8)
        ax.set_xlabel(f"s7 (7 slots) {name}")
        ax.set_ylabel(f"s12 (12 slots) {name}")
        ax.set_title(f"YTVIS val  {name}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def pick_wins(rows, key: str, top_k: int):
    """key is 's7_win_score' or 's12_win_score'."""
    ranked = sorted(rows, key=lambda r: -r[key])
    return [r for r in ranked if r[key] > 0][:top_k]


def load_metric_rows(path: Path) -> list[dict]:
    int_keys = {"sample", "s7_occ", "s12_occ"}
    rows = []
    with open(path) as fh:
        for raw in csv.DictReader(fh):
            row = {}
            for k, v in raw.items():
                row[k] = int(float(v)) if k in int_keys else float(v)
            rows.append(row)
    return rows


def viz_clip(
    batch,
    s7,
    s12,
    device,
    meta: dict,
    out_stem: Path,
    frame_stride: int,
) -> None:
    _, masks7 = score_and_masks(s7, batch, device, False)
    _, masks12 = score_and_masks(s12, batch, device, False)
    video = to_uint8_video(batch["video"])
    spatial = video.shape[-2:]
    rgb = rgb_frames(video)

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

    m7 = prep_masks(masks7, spatial)
    m12 = prep_masks(masks12, spatial)
    overlays["s7"] = overlay(video, m7)
    overlays["s12"] = overlay(video, m12)

    labels = [
        "gt",
        f"s7  ARI={meta['s7_ari']:.3f} mBO={meta['s7_mbo']:.3f}  occ={meta['s7_occ']}",
        f"s12 ARI={meta['s12_ari']:.3f} mBO={meta['s12_mbo']:.3f}  occ={meta['s12_occ']}  "
        f"dARI={meta['d_ari']:+.3f} dMBO={meta['d_mbo']:+.3f}",
    ]
    order = ["gt", "s7", "s12"]
    frames = hstack_labeled([overlays[k] for k in order], labels)
    frames = frames[:: max(frame_stride, 1)]
    save_clip(frames, out_stem)

    pred7 = m7[0].cpu().numpy().astype(bool)
    pred12 = m12[0].cpu().numpy().astype(bool)
    g7 = slot_grid_mid(rgb, pred7, "s7")
    g12 = slot_grid_mid(rgb, pred12, "s12")
    # match widths
    w = max(g7.shape[1], g12.shape[1])

    def pad_w(im, w):
        if im.shape[1] == w:
            return im
        canvas = np.zeros((im.shape[0], w, 3), dtype=np.uint8)
        canvas[:, : im.shape[1]] = im
        return canvas

    stacked = np.concatenate([pad_w(g7, w), pad_w(g12, w)], axis=0)
    Image.fromarray(stacked).save(out_stem.parent / f"{out_stem.name}_slots.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--out-dir", default="logs/vis_slotcontrast_s7_vs_s12")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = full val")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--typical-n", type=int, default=8, help="first N clips as typical look")
    ap.add_argument(
        "--metrics-csv",
        default=None,
        help="skip val scan and typical vis; only write summary + win clips from this CSV",
    )
    ap.add_argument("--frame-stride", type=int, default=2)
    args = ap.parse_args()

    root = project_root()
    out_dir = root / args.out_dir
    typical_dir = out_dir / "typical"
    s7_dir = out_dir / "s7_wins"
    s12_dir = out_dir / "s12_wins"
    for d in (out_dir, typical_dir, s7_dir, s12_dir):
        d.mkdir(parents=True, exist_ok=True)

    s7_settings = root / "logs/_ytvis_slotcontrast/settings/slotcurri/settings.yaml"
    s7_ckpt = root / "logs/_ytvis_slotcontrast/checkpoints/slotcurri_step=step=92000.ckpt"
    s12_settings = root / "logs/_ytvis_slotcontrast_s12/settings/slotcurri/settings.yaml"
    s12_ckpt = root / "logs/_ytvis_slotcontrast_s12/checkpoints/slotcurri_step=step=100000-v1.ckpt"
    data_settings = s7_settings

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = configuration.load_config(str(data_settings))
    cfg.dataset.num_val_workers = 0
    cfg.dataset.val_batch_size = 1
    dm = data.build(cfg.dataset, data_dir=args.data_dir)
    dm.setup("fit")
    loader = dm.val_dataloader()

    print("Loading models...")
    print(f"  s7  {s7_ckpt}")
    print(f"  s12 {s12_ckpt}")
    s7 = load_model(str(s7_settings), str(s7_ckpt), device)
    s12 = load_model(str(s12_settings), str(s12_ckpt), device)
    print(
        f"  s7 cycle={getattr(s7, 'cyclic_inference', True)}  "
        f"s12 cycle={getattr(s12, 'cyclic_inference', True)}"
    )

    if args.metrics_csv:
        csv_path = Path(args.metrics_csv)
        if not csv_path.is_absolute():
            csv_path = root / csv_path
        rows = load_metric_rows(csv_path)
        print(f"Loaded {len(rows)} rows from {csv_path}")
        write_scatter(rows, out_dir / "scatter_ari_mbo.png")
    else:
        rows = []
        print("Scanning val...")
        for si, batch in enumerate(loader):
            if args.max_samples and si >= args.max_samples:
                break
            sc7, m7 = score_and_masks(s7, batch, device, False)
            sc12, m12 = score_and_masks(s12, batch, device, False)
            dari = sc7["ari"] - sc12["ari"]
            dmbo = sc7["mbo"] - sc12["mbo"]
            occ7 = occupied_slots(m7)
            occ12 = occupied_slots(m12)
            row = {
                "sample": si,
                "s7_ari": sc7["ari"],
                "s7_mbo": sc7["mbo"],
                "s7_image_ari": sc7["image_ari"],
                "s7_image_mbo": sc7["image_mbo"],
                "s7_occ": occ7,
                "s12_ari": sc12["ari"],
                "s12_mbo": sc12["mbo"],
                "s12_image_ari": sc12["image_ari"],
                "s12_image_mbo": sc12["image_mbo"],
                "s12_occ": occ12,
                "d_ari": dari,
                "d_mbo": dmbo,
                "s7_win_score": max(0.0, dari) + max(0.0, dmbo),
                "s12_win_score": max(0.0, -dari) + max(0.0, -dmbo),
            }
            rows.append(row)
            if si < args.typical_n:
                stem = typical_dir / f"sample{si:03d}_dARI{dari:+.3f}_dMBO{dmbo:+.3f}"
                print(f"  typical sample {si}: dARI={dari:+.3f} dMBO={dmbo:+.3f}")
                viz_clip(batch, s7, s12, device, row, stem, args.frame_stride)
            if (si + 1) % 10 == 0:
                print(
                    f"  scanned {si + 1}  dARI={dari:+.3f} dMBO={dmbo:+.3f}  "
                    f"occ {occ7}/{occ12}"
                )

        csv_path = out_dir / "all_val_metrics.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", csv_path)
        write_scatter(rows, out_dir / "scatter_ari_mbo.png")

    s7_wins = pick_wins(rows, "s7_win_score", args.top_k)
    s12_wins = pick_wins(rows, "s12_win_score", args.top_k)

    summary = {
        "n_val": len(rows),
        "s7_ckpt": str(s7_ckpt),
        "s12_ckpt": str(s12_ckpt),
        "mean_s7_ari": float(np.mean([r["s7_ari"] for r in rows])),
        "mean_s12_ari": float(np.mean([r["s12_ari"] for r in rows])),
        "mean_s7_mbo": float(np.mean([r["s7_mbo"] for r in rows])),
        "mean_s12_mbo": float(np.mean([r["s12_mbo"] for r in rows])),
        "mean_s7_occ": float(np.mean([r["s7_occ"] for r in rows])),
        "mean_s12_occ": float(np.mean([r["s12_occ"] for r in rows])),
        "mean_d_ari": float(np.mean([r["d_ari"] for r in rows])),
        "mean_d_mbo": float(np.mean([r["d_mbo"] for r in rows])),
        "n_s7_better_ari": int(sum(1 for r in rows if r["d_ari"] > 0)),
        "n_s12_better_ari": int(sum(1 for r in rows if r["d_ari"] < 0)),
        "n_s7_better_mbo": int(sum(1 for r in rows if r["d_mbo"] > 0)),
        "n_s12_better_mbo": int(sum(1 for r in rows if r["d_mbo"] < 0)),
        "s7_win_samples": [r["sample"] for r in s7_wins],
        "s12_win_samples": [r["sample"] for r in s12_wins],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    for name, chosen, dest in (
        ("s7", s7_wins, s7_dir),
        ("s12", s12_wins, s12_dir),
    ):
        with open(dest / "winners_table.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(chosen[0].keys()) if chosen else [])
            if chosen:
                w.writeheader()
                w.writerows(chosen)

    want = {}
    for i, r in enumerate(s7_wins):
        want[r["sample"]] = ("s7", i, r)
    for i, r in enumerate(s12_wins):
        # a clip can theoretically appear in both lists; keep both by viz twice
        prev = want.get(r["sample"])
        if prev is None:
            want[r["sample"]] = ("s12", i, r)
        else:
            want[r["sample"]] = ("both", i, r)

    print(f"Visualizing {len(want)} win clips:", sorted(want))
    for si, batch in enumerate(loader):
        if args.max_samples and si >= args.max_samples:
            break
        if si not in want:
            continue
        side, rank_i, meta = want[si]
        if side in ("s7", "both"):
            r = next(x for x in s7_wins if x["sample"] == si)
            rank = next(i for i, x in enumerate(s7_wins) if x["sample"] == si)
            stem = s7_dir / (
                f"rank{rank:02d}_sample{si:03d}_"
                f"dARI{r['d_ari']:+.3f}_dMBO{r['d_mbo']:+.3f}"
            )
            print(f"viz s7-win sample {si}: dARI={r['d_ari']:+.3f} dMBO={r['d_mbo']:+.3f}")
            viz_clip(batch, s7, s12, device, r, stem, args.frame_stride)
        if side in ("s12", "both"):
            r = next(x for x in s12_wins if x["sample"] == si)
            rank = next(i for i, x in enumerate(s12_wins) if x["sample"] == si)
            stem = s12_dir / (
                f"rank{rank:02d}_sample{si:03d}_"
                f"dARI{r['d_ari']:+.3f}_dMBO{r['d_mbo']:+.3f}"
            )
            print(f"viz s12-win sample {si}: dARI={r['d_ari']:+.3f} dMBO={r['d_mbo']:+.3f}")
            viz_clip(batch, s7, s12, device, r, stem, args.frame_stride)

    print("done ->", out_dir)


if __name__ == "__main__":
    main()
