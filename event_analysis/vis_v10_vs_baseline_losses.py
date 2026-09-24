#!/usr/bin/env python3
"""Scan full val: clips where a method loses badly to SlotCurri baseline.

Default --method v10 --dataset ytvis reproduces logs/vis_v10_loses_to_baseline.
  python event_analysis/vis_v10_vs_baseline_losses.py --method v26
writes logs/vis_v26_loses_to_baseline (v26 uses its own cyclic_inference=False).

  python event_analysis/vis_v10_vs_baseline_losses.py --method v39 --dataset movi_c
compares logs/_movi_c_attnmass_v39 to the official SlotCurri MOVi-C checkpoint
(checkpoints/movi_c.ckpt) with ignore_background=True, and writes
logs/vis_v39_movi_c_loses_to_baseline.
"""

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


def project_root() -> Path:
    root = Path("/workspace/SlotCurri")
    if (root / "slotcurri").exists():
        return root
    return Path("/mnt/ssd2/hmlee/SlotCurri")


def method_ckpt(root: Path, method: str, dataset: str = "ytvis") -> Path:
    prefix = "movi_c" if dataset == "movi_c" else "ytvis"
    ckpt_dir = root / f"logs/_{prefix}_attnmass_{method}" / "checkpoints"
    named = ckpt_dir / "slotcurri_step=step=100000-v1.ckpt"
    if named.is_file():
        return named
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if ckpts:
        return ckpts[-1]
    # Eval-only overlays (v39lam1gu_perron): no own run dir; reuse parent 100k ckpt.
    if method.endswith("_perron"):
        parent = method[: -len("_perron")]
        if parent:
            return method_ckpt(root, parent, dataset)
    raise FileNotFoundError(f"no checkpoint in {ckpt_dir}")


def method_settings_path(root: Path, method: str, dataset: str) -> Path:
    """Trained-run dump if present, else the source yaml (eval-only overlays)."""
    prefix = "movi_c" if dataset == "movi_c" else "ytvis"
    logged = root / f"logs/_{prefix}_attnmass_{method}" / "settings/slotcurri/settings.yaml"
    if logged.is_file():
        return logged
    cfg_name = (
        f"movi_c_attnmass_{method}.yaml"
        if dataset == "movi_c"
        else f"ytvis2021_attnmass_{method}.yaml"
    )
    cfg = root / "configs/slotcurri" / cfg_name
    if cfg.is_file():
        return cfg
    raise FileNotFoundError(
        f"no settings for method={method!r} dataset={dataset!r}: tried {logged} and {cfg}"
    )


def resolve_run(root: Path, method: str, dataset: str) -> dict:
    """SlotCurri is the baseline on both datasets (SOTA paper checkpoint / local run)."""
    method_settings = method_settings_path(root, method, dataset)
    if dataset == "movi_c":
        data_settings = method_settings
        logged = (
            root / f"logs/_movi_c_attnmass_{method}" / "settings/slotcurri/settings.yaml"
        )
        if not logged.is_file() and method.endswith("_perron"):
            parent_logged = (
                root
                / f"logs/_movi_c_attnmass_{method[: -len('_perron')]}"
                / "settings/slotcurri/settings.yaml"
            )
            if parent_logged.is_file():
                data_settings = parent_logged
        return {
            "dataset": dataset,
            "baseline_name": "slotcurri",
            "baseline_settings": root / "configs/slotcurri/movi_c.yaml",
            "baseline_ckpt": root / "checkpoints/movi_c.ckpt",
            "method_settings": method_settings,
            "method_ckpt": method_ckpt(root, method, dataset),
            "data_settings": data_settings,
            "ignore_background": True,
            "out_stem": f"{method}_movi_c",
        }
    if dataset != "ytvis":
        raise ValueError(f"unknown dataset {dataset!r}")
    return {
        "dataset": dataset,
        "baseline_name": "slotcurri",
        "baseline_settings": root / "logs/_ytvis/settings/slotcurri/settings.yaml",
        "baseline_ckpt": root / "logs/_ytvis/checkpoints/slotcurri_step=step=100000-v1.ckpt",
        "method_settings": method_settings,
        "method_ckpt": method_ckpt(root, method, dataset),
        "data_settings": root / "logs/_ytvis/settings/slotcurri/settings.yaml",
        "ignore_background": False,
        "out_stem": method,
    }


def save_clip(frames, stem: Path) -> None:
    Image.fromarray(frames[len(frames) // 2]).save(stem.parent / f"{stem.name}_mid.png")
    try:
        import imageio

        imageio.mimsave(stem.parent / f"{stem.name}.gif", frames, fps=4)
    except Exception as e:
        print("gif skip:", e)
    try:
        import imageio

        imageio.mimsave(stem.parent / f"{stem.name}.mp4", frames, fps=4)
    except Exception as e:
        print("mp4 skip:", e)


def load_model(settings_yaml: str, ckpt: str, device: torch.device):
    config = configuration.load_config(settings_yaml)
    config.model.visualize = False
    model = models.build(config.model, config.optimizer)
    model.load_weights_from_checkpoint(ckpt)
    model.to(device).eval()
    return model


def build_val_metrics(ignore_background: bool = False):
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
            video_input=True,
            ignore_background=ignore_background,
            **kw,
        ),
    }


@torch.no_grad()
def score_and_masks(model, batch, device, ignore_background: bool = False):
    batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    cycle = getattr(model, "cyclic_inference", True)
    outputs = model.forward(batch_dev, train=False, cycle=cycle)
    aux = model.aux_forward(batch_dev, outputs)
    scores = {}
    for name, metric in build_val_metrics(ignore_background).items():
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
    ap.add_argument("--method", default="v10", help="attn-mass run tag, e.g. v10 or v26")
    ap.add_argument(
        "--dataset",
        default="ytvis",
        choices=("ytvis", "movi_c"),
        help="ytvis uses logs/_ytvis; movi_c uses official SlotCurri checkpoints/movi_c.ckpt",
    )
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = full val")
    ap.add_argument("--top-k", type=int, default=20, help="visualize worst K by loss score")
    ap.add_argument(
        "--min-dari",
        type=float,
        default=0.05,
        help="also keep if baseline_ari - method_ari >= this",
    )
    ap.add_argument(
        "--min-dmbo",
        type=float,
        default=0.05,
        help="also keep if baseline_mbo - method_mbo >= this",
    )
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument(
        "--skip-viz",
        action="store_true",
        help="write all_val_metrics.csv / summary only; skip loser clip rendering",
    )
    args = ap.parse_args()

    method = args.method
    root = project_root()
    spec = resolve_run(root, method, args.dataset)
    ign_bg = spec["ignore_background"]
    out_dir = root / (args.out_dir or f"logs/vis_{spec['out_stem']}_loses_to_baseline")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = configuration.load_config(str(spec["data_settings"]))
    cfg.dataset.num_val_workers = 0
    cfg.dataset.val_batch_size = 1
    dm = data.build(cfg.dataset, data_dir=args.data_dir)
    dm.setup("fit")
    loader = dm.val_dataloader()

    print("Loading models...")
    print(f"  dataset={spec['dataset']} baseline={spec['baseline_name']} {spec['baseline_ckpt']}")
    print(f"  method={method} ckpt={spec['method_ckpt']}")
    print(f"  ignore_background={ign_bg}")
    base = load_model(str(spec["baseline_settings"]), str(spec["baseline_ckpt"]), device)
    meth = load_model(str(spec["method_settings"]), str(spec["method_ckpt"]), device)
    print(
        f"  baseline cycle={getattr(base, 'cyclic_inference', True)}  "
        f"{method} cycle={getattr(meth, 'cyclic_inference', True)}"
    )
    print(
        f"  {method} settings={spec['method_settings']}  "
        f"src_gate={getattr(meth, 'amc_predictor_src_gate', None)}  "
        f"eval_src_gate={getattr(meth, 'amc_eval_predictor_src_gate', None)}  "
        f"eval_perron={getattr(meth, 'amc_eval_perron_readout', None)}  "
        f"eval_pair_iso={getattr(meth, 'amc_eval_predictor_pair_isolate', None)}"
    )

    rows = []
    # keep CPU tensors for later viz of losers only (masks can be large — recompute for top-k)
    print("Scanning val...")
    for si, batch in enumerate(loader):
        if args.max_samples and si >= args.max_samples:
            break
        sb, _ = score_and_masks(base, batch, device, ign_bg)
        sv, _ = score_and_masks(meth, batch, device, ign_bg)
        dari = sb["ari"] - sv["ari"]
        dmbo = sb["mbo"] - sv["mbo"]
        # positive = baseline better / method loses
        loss_score = max(0.0, dari) + max(0.0, dmbo)
        row = {
            "sample": si,
            "baseline_ari": sb["ari"],
            "baseline_mbo": sb["mbo"],
            "baseline_image_ari": sb["image_ari"],
            "baseline_image_mbo": sb["image_mbo"],
            f"{method}_ari": sv["ari"],
            f"{method}_mbo": sv["mbo"],
            f"{method}_image_ari": sv["image_ari"],
            f"{method}_image_mbo": sv["image_mbo"],
            "d_ari": dari,
            "d_mbo": dmbo,
            "loss_score": loss_score,
            f"{method}_loses_ari": dari >= args.min_dari,
            f"{method}_loses_mbo": dmbo >= args.min_dmbo,
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
        if r[f"{method}_loses_ari"] or r[f"{method}_loses_mbo"]
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
        "method": method,
        "dataset": spec["dataset"],
        "baseline": spec["baseline_name"],
        "baseline_ckpt": str(spec["baseline_ckpt"]),
        "ignore_background": ign_bg,
        "n_lose_ari_ge": sum(1 for r in rows if r[f"{method}_loses_ari"]),
        "n_lose_mbo_ge": sum(1 for r in rows if r[f"{method}_loses_mbo"]),
        "n_lose_either": len(thresh),
        "mean_d_ari": float(np.nanmean([r["d_ari"] for r in rows])),
        "mean_d_mbo": float(np.nanmean([r["d_mbo"] for r in rows])),
        "chosen_samples": [r["sample"] for r in chosen],
        "thresholds": {"min_dari": args.min_dari, "min_dmbo": args.min_dmbo, "top_k": args.top_k},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    want = {r["sample"] for r in chosen}
    if args.skip_viz:
        print(f"skip viz; would have rendered {len(want)} loser clips:", sorted(want))
    else:
        print(f"Visualizing {len(want)} loser clips:", sorted(want))
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
            _, masks_b = score_and_masks(base, batch, device, ign_bg)
            _, masks_v = score_and_masks(meth, batch, device, ign_bg)
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
            overlays[method] = overlay(video, prep_masks(masks_v, spatial))

            labels = [
                "gt",
                f"{spec['baseline_name']} ARI={meta['baseline_ari']:.3f} mBO={meta['baseline_mbo']:.3f}",
                f"{method} ARI={meta[f'{method}_ari']:.3f} mBO={meta[f'{method}_mbo']:.3f}  "
                f"dARI={meta['d_ari']:+.3f} dMBO={meta['d_mbo']:+.3f}",
            ]
            order = ["gt", "baseline", method]
            frames = hstack_labeled([overlays[k] for k in order], labels)
            frames = frames[:: max(args.frame_stride, 1)]
            rank_i = next(i for i, r in enumerate(chosen) if r["sample"] == si)
            stem = f"rank{rank_i:02d}_sample{si:03d}_dARI{meta['d_ari']:+.3f}_dMBO{meta['d_mbo']:+.3f}"
            save_clip(frames, out_dir / stem)

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
