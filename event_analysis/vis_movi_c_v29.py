#!/usr/bin/env python3
"""Load MOVi-C v29 checkpoint and visualize slot masks vs GT.

Matches val: train=False (p_end / beta_final), cyclic_inference from config,
ignore_background=True FG-ARI/mBO.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from slotcurri import configuration, data, metrics as metric_lib, models
from slotcurri.data.transforms import Denormalize
from slotcurri.visualizations import mix_videos_with_masks


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
]


def _font(size: int = 14):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def build_val_metrics():
    kw = dict(pred_key="decoder_masks_hard", true_key="segmentations")
    return {
        "ari": metric_lib.VideoARI(ignore_background=True, **kw),
        "image_ari": metric_lib.ImageARI(
            video_input=True, ignore_background=True, **kw
        ),
        "mbo": metric_lib.VideoIoU(matching="overlap", ignore_background=True, **kw),
        "image_mbo": metric_lib.ImageIoU(
            matching="overlap", ignore_background=True, video_input=True, **kw
        ),
    }


def load_model(settings_yaml: str, ckpt: str, device: torch.device):
    config = configuration.load_config(settings_yaml)
    config.model.visualize = False
    model = models.build(config.model, config.optimizer)
    model.load_weights_from_checkpoint(ckpt)
    model.to(device)
    model.eval()
    return model, config


def to_uint8_video(video: torch.Tensor) -> torch.Tensor:
    denorm = Denormalize(input_type="video")
    vid = denorm(video[0].cpu()).clamp(0, 1)
    return vid.unsqueeze(0)


def overlay(video_btchw: torch.Tensor, masks_btnchw: torch.Tensor, alpha=0.5) -> np.ndarray:
    mixed = mix_videos_with_masks(video_btchw, masks_btnchw.float(), alpha=alpha)
    return mixed[0].permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)


def add_bar(frame: np.ndarray, text: str, color=(20, 20, 20)) -> np.ndarray:
    img = Image.fromarray(frame)
    bar_h = 26
    canvas = Image.new("RGB", (img.width, img.height + bar_h), color)
    canvas.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), text, fill=(240, 240, 240), font=_font(14))
    return np.asarray(canvas)


def hstack(frames_list, labels):
    out = []
    n = min(len(f) for f in frames_list)
    for i in range(n):
        parts = [add_bar(f[i], lab) for f, lab in zip(frames_list, labels)]
        out.append(np.concatenate(parts, axis=1))
    return out


def save_gif(path: Path, frames, fps=4):
    try:
        import imageio

        imageio.mimsave(path, frames, fps=fps)
    except Exception:
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


def resize_masks(masks: torch.Tensor, hw):
    if masks.shape[-2:] == hw:
        return masks
    b, t, s, _, _ = masks.shape
    m = masks.float().reshape(b * t, s, masks.shape[-2], masks.shape[-1])
    m = torch.nn.functional.interpolate(m, size=hw, mode="nearest")
    return m.reshape(b, t, s, *hw)


def rgb_frames(video_01: torch.Tensor) -> np.ndarray:
    return (video_01[0].permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def tint_slot(rgb: np.ndarray, mask_thw: np.ndarray, color, alpha=0.55) -> np.ndarray:
    out = rgb.copy()
    c = np.array(color, dtype=np.float32)
    m = mask_thw[..., None].astype(np.float32)
    out = (out.astype(np.float32) * (1 - alpha * m) + c * (alpha * m)).astype(np.uint8)
    return out


def slot_gt_stats(pred_ts_hw: np.ndarray, gt_tc_hw: np.ndarray) -> dict:
    """pred: (T,S,H,W) bool; gt: (T,C,H,W) bool. class 0 = background."""
    t, s, h, w = pred_ts_hw.shape
    c = gt_tc_hw.shape[1]
    pred = pred_ts_hw.reshape(t, s, -1)
    gt = gt_tc_hw.reshape(t, c, -1)
    # overlap[t,s,c]
    overlap = np.einsum("tsn,tcn->tsc", pred.astype(np.float32), gt.astype(np.float32))
    slot_pix = pred.sum(axis=-1).astype(np.float32)  # (T,S)
    gt_pix = gt.sum(axis=-1).astype(np.float32)  # (T,C)

    # video-level (sum over time)
    ov_s = overlap.sum(axis=0)  # (S,C)
    slot_s = slot_pix.sum(axis=0)
    gt_s = gt_pix.sum(axis=0)

    slots = []
    for si in range(s):
        pix = float(slot_s[si])
        if pix < 1:
            slots.append(
                {
                    "slot": si,
                    "kind": "empty",
                    "pix_frac": 0.0,
                    "dom_gt": None,
                    "dom_frac": 0.0,
                    "n_gt_over_10": 0,
                }
            )
            continue
        fracs = ov_s[si] / max(pix, 1.0)
        dom = int(np.argmax(fracs))
        n_over = int((fracs >= 0.10).sum())
        if dom == 0 and fracs[0] >= 0.70:
            kind = "bg_chip"
        elif dom == 0:
            kind = "mixed_bg"
        elif n_over >= 2 and fracs[dom] < 0.70:
            kind = "merge"
        else:
            kind = "object"
        slots.append(
            {
                "slot": si,
                "kind": kind,
                "pix_frac": pix / (t * h * w),
                "dom_gt": dom,
                "dom_frac": float(fracs[dom]),
                "n_gt_over_10": n_over,
                "gt_fracs": [float(x) for x in fracs],
            }
        )

    # how many slots claim each FG object
    obj_split = {}
    for ci in range(1, c):
        if gt_s[ci] < 1:
            continue
        claimants = [
            st["slot"]
            for st in slots
            if st["kind"] in ("object", "merge", "mixed_bg")
            and st["dom_gt"] == ci
        ]
        # also slots with >=10% of their pixels on this object
        any_claim = [
            st["slot"]
            for st in slots
            if st.get("gt_fracs") and st["gt_fracs"][ci] >= 0.10
        ]
        obj_split[int(ci)] = {
            "gt_pix_frac": float(gt_s[ci] / (t * h * w)),
            "dominant_slots": claimants,
            "slots_with_10pct": any_claim,
        }

    kinds = {k: 0 for k in ("empty", "bg_chip", "mixed_bg", "object", "merge")}
    for st in slots:
        kinds[st["kind"]] += 1
    return {"slots": slots, "objects": obj_split, "kind_counts": kinds}


@torch.no_grad()
def run_sample(model, batch, device, cycle: bool):
    batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    outputs = model.forward(batch_dev, train=False, cycle=cycle)
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
    proc = outputs["processor"]
    gate = proc.get("active_mask")
    conf = proc.get("gate_conf")
    gate_np = gate.float().cpu().numpy() if gate is not None else None
    conf_np = conf.float().cpu().numpy() if conf is not None else None

    grouping = aux.get("grouping_masks")
    mass = None
    if grouping is not None:
        g = grouping.float()  # (B,T,S,H,W) or similar
        if g.ndim == 5:
            mass = (g / g.sum(dim=2, keepdim=True).clamp_min(1e-8)).sum(dim=(-1, -2))
            mass = (mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)).cpu().numpy()
    return scores, masks, gate_np, conf_np, mass


def slot_grid(rgb, pred_ts_hw, gate_ts, conf_ts, mass_ts, kinds, vis_hw=(168, 168)):
    """One mid-frame grid: RGB + 11 slot tints."""
    t = rgb.shape[0]
    mid = t // 2
    frame = rgb[mid]
    s = pred_ts_hw.shape[1]
    th, tw = vis_hw
    frame_s = np.array(Image.fromarray(frame).resize((tw, th), Image.BILINEAR))

    cells = [add_bar(frame_s, "rgb")]
    for si in range(s):
        m = pred_ts_hw[mid, si]
        m_img = np.array(Image.fromarray((m.astype(np.uint8) * 255)).resize((tw, th), Image.NEAREST))
        m_bool = m_img > 127
        tinted = tint_slot(frame_s, m_bool, TAB[si % len(TAB)], alpha=0.65)
        g = float(gate_ts[mid, si]) if gate_ts is not None else float("nan")
        c = float(conf_ts[mid, si]) if conf_ts is not None else float("nan")
        ms = float(mass_ts[mid, si]) if mass_ts is not None else float("nan")
        kind = kinds[si]["kind"]
        dom = kinds[si]["dom_gt"]
        lab = f"s{si} {kind} g={g:.2f} c={c:.2f} m={ms:.3f} gt={dom}"
        cells.append(add_bar(tinted, lab, color=(30, 10, 10) if kind == "bg_chip" else (20, 20, 20)))

    # 4 columns
    cols = 4
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(row[0]))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def heatmap_png(mat: np.ndarray, row_labels, col_labels, title: str, path: Path):
    rh, cw = 22, 36
    h = 28 + rh * (len(row_labels) + 1)
    w = 90 + cw * len(col_labels)
    img = Image.new("RGB", (w, h), (18, 18, 18))
    dr = ImageDraw.Draw(img)
    dr.text((8, 4), title, fill=(230, 230, 230), font=_font(13))
    mx = max(float(mat.max()), 1e-8)
    for j, lab in enumerate(col_labels):
        dr.text((90 + j * cw + 4, 24), lab, fill=(200, 200, 200), font=_font(11))
    for i, lab in enumerate(row_labels):
        dr.text((4, 46 + i * rh), lab, fill=(200, 200, 200), font=_font(11))
        for j in range(mat.shape[1]):
            v = float(mat[i, j])
            t = v / mx
            col = (int(20 + 200 * t), int(20 + 40 * (1 - t)), int(40 + 20 * (1 - t)))
            x0, y0 = 90 + j * cw, 44 + i * rh
            dr.rectangle([x0, y0, x0 + cw - 2, y0 + rh - 2], fill=col)
            dr.text((x0 + 3, y0 + 3), f"{v:.2f}", fill=(255, 255, 255), font=_font(10))
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument(
        "--settings",
        default="logs/_movi_c_attnmass_v29/settings/slotcurri/settings.yaml",
    )
    ap.add_argument(
        "--ckpt",
        default="logs/_movi_c_attnmass_v29/checkpoints/slotcurri_step=step=55000.ckpt",
    )
    ap.add_argument("--out-dir", default="logs/vis_movi_c_v29_55k")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    root = Path("/workspace/SlotCurri")
    if not (root / "slotcurri").exists():
        root = Path("/mnt/ssd2/hmlee/SlotCurri")

    settings = root / args.settings if not Path(args.settings).is_absolute() else Path(args.settings)
    ckpt = root / args.ckpt if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device} ckpt={ckpt}")

    model, config = load_model(str(settings), str(ckpt), device)
    cycle = bool(getattr(model, "cyclic_inference", False))
    print(
        f"cycle={cycle} p_end_mult={model.amc_p_end_mult} "
        f"eval_p={model._gate_threshold(False):.5f} "
        f"eval_beta={model._gate_beta(False)}"
    )

    cfg = configuration.load_config(str(settings))
    cfg.dataset.num_val_workers = 0
    cfg.dataset.val_batch_size = 1
    dm = data.build(cfg.dataset, data_dir=args.data_dir)
    dm.setup("fit")
    loader = dm.val_dataloader()

    rows = []
    kind_tot = {k: 0 for k in ("empty", "bg_chip", "mixed_bg", "object", "merge")}
    split_n = 0
    obj_n = 0

    for si, batch in enumerate(loader):
        if si >= args.num_samples:
            break
        print(f"\n=== sample {si} ===")
        scores, masks, gate, conf, mass = run_sample(model, batch, device, cycle)
        print(
            f"  FG-ARI={scores['ari']:.4f} mBO={scores['mbo']:.4f} "
            f"iARI={scores['image_ari']:.4f} iMBO={scores['image_mbo']:.4f}"
        )

        video = to_uint8_video(batch["video"])
        hw = video.shape[-2:]
        masks = resize_masks(masks, hw).bool()
        pred = masks[0].numpy()  # T,S,H,W

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
        for k, v in stats["kind_counts"].items():
            kind_tot[k] += v
        for od in stats["objects"].values():
            obj_n += 1
            if len(od["slots_with_10pct"]) >= 2:
                split_n += 1

        print("  slots:", " ".join(f"s{st['slot']}={st['kind']}/{st['dom_gt']}" for st in stats["slots"]))
        print("  objects split (>=2 slots@10%):",
              [(k, v["slots_with_10pct"]) for k, v in stats["objects"].items()
               if len(v["slots_with_10pct"]) >= 2])

        rgb = rgb_frames(video)
        ov_gt = overlay(video, gt)
        ov_pr = overlay(video, masks)
        labels = [
            "rgb",
            "gt (incl bg)",
            f"pred  FG-ARI={scores['ari']:.3f} mBO={scores['mbo']:.3f}",
        ]
        frames = hstack([rgb, ov_gt, ov_pr], labels)
        frames = frames[:: max(args.frame_stride, 1)]
        Image.fromarray(frames[len(frames) // 2]).save(out_dir / f"sample{si:02d}_mid.png")
        save_gif(out_dir / f"sample{si:02d}.gif", frames)

        # per-slot mid-frame grid
        g_ts = gate[0] if gate is not None else None
        c_ts = conf[0] if conf is not None else None
        m_ts = mass[0] if mass is not None else None
        grid = slot_grid(rgb, pred, g_ts, c_ts, m_ts, stats["slots"])
        Image.fromarray(grid).save(out_dir / f"sample{si:02d}_slots.png")

        # slot x GT heatmap (video-sum pixel fraction of slot)
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
            [f"bg" if i == 0 else f"o{i}" for i in range(c)],
            f"sample{si} slot pixel share of GT class",
            out_dir / f"sample{si:02d}_heatmap.png",
        )

        row = {"sample": si, **scores, **{f"n_{k}": v for k, v in stats["kind_counts"].items()}}
        if gate is not None:
            row["gate_mean"] = float(gate.mean())
            row["n_gate_gt_05"] = int((gate[0].mean(axis=0) > 0.5).sum())
        if conf is not None:
            row["conf_mean"] = float(conf.mean())
        rows.append(row)
        (out_dir / f"sample{si:02d}_stats.json").write_text(json.dumps(stats, indent=2))

    if rows:
        keys = list(rows[0].keys())
        with open(out_dir / "per_sample_metrics.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("\n=== means ===")
        for mk in ("ari", "mbo", "image_ari", "image_mbo"):
            vals = [r[mk] for r in rows]
            print(f"  {mk}: {sum(vals)/len(vals):.4f}")
        print("kind totals (over samples x slots):", kind_tot)
        print(f"FG objects split across >=2 slots: {split_n}/{obj_n}")
        summary = {
            "ckpt": str(ckpt),
            "n_samples": len(rows),
            "mean": {mk: sum(r[mk] for r in rows) / len(rows) for mk in ("ari", "mbo", "image_ari", "image_mbo")},
            "kind_totals": kind_tot,
            "fg_objects_split": {"split": split_n, "total": obj_n},
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
