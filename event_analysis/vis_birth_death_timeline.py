#!/usr/bin/env python3
"""Build a paper-style birth/death timeline figure from real YTVIS frames.

Scans the val set for clips where GT objects appear mid-video, measures which
slots become spatially diffuse over time, and renders:

    row0: RGB frames at (early, late-before-birth, birth)
    row1: spare-slot attention heatmap (the slot that diffused)
    row2: all-slot mask overlay

Default checkpoint is the SlotCurri YTVIS run (always-active 7-slot pool at
eval). Pass --settings/--ckpt to point at a SlotContrast checkpoint instead.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from slotcurri import configuration, data, models
from slotcurri.data.transforms import Denormalize
from slotcurri.visualizations import mix_videos_with_masks


def load_model(settings_yaml: str, ckpt: str, device: torch.device):
    config = configuration.load_config(settings_yaml)
    config.model.visualize = False
    model = models.build(config.model, config.optimizer)
    model.load_weights_from_checkpoint(ckpt)
    model.to(device).eval()
    return model, config


def to_uint8_video(video: torch.Tensor) -> torch.Tensor:
    denorm = Denormalize(input_type="video")
    vid = denorm(video[0].cpu()).clamp(0, 1)
    return vid.unsqueeze(0)


def spatial_side(n_patches: int) -> int:
    side = int(round(math.sqrt(n_patches)))
    if side * side != n_patches:
        raise ValueError(f"n_patches={n_patches} is not a square")
    return side


def slot_spatial_entropy(att_sf: torch.Tensor) -> torch.Tensor:
    """Entropy of each slot's attention map over patches. att: (T, S, F)."""
    # Renormalize over patches so each slot is a spatial distribution.
    p = att_sf / att_sf.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    ent = -(p * p.clamp_min(1e-8).log()).sum(dim=-1)  # (T, S)
    return ent


def find_birth_frames(seg: torch.Tensor) -> list[tuple[int, int]]:
    """Return list of (t, new_object_id) where a GT id first appears (id!=0)."""
    # seg: (T, H, W)
    births = []
    seen = set()
    for t in range(seg.shape[0]):
        ids = {int(i) for i in torch.unique(seg[t]) if int(i) != 0}
        new = ids - seen
        for oid in sorted(new):
            if seen:  # skip objects present from frame 0; keep true mid-clip births
                births.append((t, oid))
            seen.add(oid)
        # also record first-frame objects into seen without counting as birth
        if t == 0:
            seen |= ids
    return births


@torch.no_grad()
def analyze_clip(model, batch, device, cycle: bool):
    batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if "batch_padding_mask" in batch_dev:
        batch_dev = model._remove_padding(batch_dev, batch_dev["batch_padding_mask"])
        if batch_dev is None:
            return None
    outputs = model.forward(batch_dev, train=False, cycle=cycle)
    aux = model.aux_forward(batch_dev, outputs)
    att = outputs["processor"]["state_attn_mask"][0].float().cpu()  # (T, S, F)
    masks = aux.get("decoder_masks_vis_hard", aux["decoder_masks_hard"])[0].cpu()
    video = to_uint8_video(batch_dev["video"])
    seg = batch_dev["segmentations"][0].cpu()
    return {
        "att": att,
        "masks": masks,
        "video": video,  # (1, T, C, H, W) float 0-1 after denorm... actually uint path
        "seg": seg,
        "video_f": video,
    }


def heatmap_to_rgb(hm: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """hm: (h, w) in [0,1] -> RGB uint8 jet-like via matplotlib-free ramp."""
    hm = np.clip(hm, 0, 1)
    # simple blue->cyan->yellow->red
    r = np.clip(1.5 * hm - 0.2, 0, 1)
    g = np.clip(1.5 - abs(2 * hm - 1) * 1.2, 0, 1)
    b = np.clip(1.2 - 1.5 * hm, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    img = Image.fromarray((rgb * 255).astype(np.uint8)).resize(
        (size_hw[1], size_hw[0]), Image.BILINEAR
    )
    return np.asarray(img)


def overlay_attn_on_frame(frame_chw: torch.Tensor, attn_hw: np.ndarray, alpha=0.55):
    """frame: (C,H,W) float 0-1; attn_hw low-res."""
    H, W = frame_chw.shape[-2:]
    heat = heatmap_to_rgb(attn_hw / (attn_hw.max() + 1e-8), (H, W)).astype(np.float32) / 255
    base = frame_chw.permute(1, 2, 0).numpy()
    mix = (1 - alpha) * base + alpha * heat
    return (np.clip(mix, 0, 1) * 255).astype(np.uint8)


def label_bar(img: Image.Image, text: str, height=40) -> Image.Image:
    bar = Image.new("RGB", (img.width, height), (245, 245, 245))
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 10), text, fill=(20, 20, 20), font=font)
    canvas = Image.new("RGB", (img.width, img.height + height), (255, 255, 255))
    canvas.paste(bar, (0, 0))
    canvas.paste(img, (0, height))
    return canvas


def row_label(img: Image.Image, text: str, width=150) -> Image.Image:
    side = Image.new("RGB", (width, img.height), (255, 255, 255))
    draw = ImageDraw.Draw(side)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    # vertical-ish: just top-left text
    draw.text((8, img.height // 2 - 8), text, fill=(20, 20, 20), font=font)
    canvas = Image.new("RGB", (width + img.width, img.height), (255, 255, 255))
    canvas.paste(side, (0, 0))
    canvas.paste(img, (width, 0))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--settings",
        default="logs/_ytvis/settings/slotcurri/settings.yaml",
    )
    ap.add_argument(
        "--ckpt",
        default="logs/_ytvis/checkpoints/slotcurri_step=step=100000-v1.ckpt",
    )
    ap.add_argument("--out", default="assets/slot_birth_death_timeline_real.png")
    ap.add_argument("--scan", type=int, default=80, help="max val clips to scan")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cycle", action="store_true", help="use cyclic inference")
    ap.add_argument("--sample", type=int, default=None, help="force sample index")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.settings, args.ckpt, device)
    dataset = data.build(config.dataset)
    dataset.setup("validate")
    loader = dataset.val_dataloader()

    candidates = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if batch is None:
                continue
            if idx >= args.scan:
                break
            if args.sample is not None and idx != args.sample:
                continue

            result = analyze_clip(model, batch, device, cycle=args.cycle)
            if result is None:
                continue
            att = result["att"]  # (T,S,F)
            seg = result["seg"]
            T, S, F = att.shape
            side = spatial_side(F)
            ent = slot_spatial_entropy(att)  # (T,S)
            mass = att.sum(-1) / F  # (T,S)

            births = find_birth_frames(seg)

            # Always consider diffusion-only candidates (YTVIS rarely has mid-clip births
            # while other objects are already present).
            early = ent[: max(1, T // 5)].mean(0)
            late = ent[-max(1, T // 5) :].mean(0)
            d_ent = (late - early).numpy()
            early_mass = mass[: max(1, T // 5)].mean(0).numpy()
            # Prefer low-mass early slots that become spatially diffuse later.
            spare_score = d_ent - 0.35 * early_mass
            spare = int(np.argmax(spare_score))
            # Require that early frames actually contain GT objects (readable story).
            early_has_obj = bool(((seg[: max(1, T // 5)] > 0).flatten().sum() > 0).item())
            if (args.sample is not None or (d_ent[spare] >= 0.15 and early_has_obj)):
                t_end = T - 1
                candidates.append(
                    {
                        "idx": idx,
                        "score": float(spare_score[spare]),
                        "spare": spare,
                        "t_birth": t_end,
                        "t_early": 0,
                        "t_late": max(1, (T * 2) // 3),
                        "result": result,
                        "ent": ent,
                        "mass": mass,
                        "side": side,
                        "kind": "diffuse_only",
                    }
                )

            # Bonus candidates when a new GT id appears after frame 0.
            for t_birth, oid in births:
                if t_birth < 3:
                    continue
                t_early = 0
                t_late = max(0, t_birth - 1)
                early_e = ent[t_early : t_early + max(1, t_birth // 4)].mean(0)
                late_e = ent[max(0, t_late - max(1, t_birth // 6)) : t_late + 1].mean(0)
                d = (late_e - early_e).numpy()
                em = mass[t_early : t_early + max(1, t_birth // 4)].mean(0).numpy()
                ss = d - 0.5 * em
                spare_b = int(np.argmax(ss))
                candidates.append(
                    {
                        "idx": idx,
                        "score": float(ss[spare_b]) + 1.0,  # prefer true birth
                        "spare": spare_b,
                        "t_birth": t_birth,
                        "t_early": t_early,
                        "t_late": t_late,
                        "oid": oid,
                        "result": result,
                        "ent": ent,
                        "mass": mass,
                        "side": side,
                        "kind": "birth",
                    }
                )

            if args.sample is not None:
                break

    if not candidates:
        raise SystemExit("No suitable clips found. Try --sample N or increase --scan.")

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]
    print(
        f"selected sample={best['idx']} kind={best['kind']} spare_slot={best['spare']} "
        f"t_early={best['t_early']} t_late={best['t_late']} t_birth={best['t_birth']} "
        f"score={best['score']:.3f}"
    )
    for c in candidates[:8]:
        print(
            f"  cand sample={c['idx']} kind={c['kind']} spare={c['spare']} "
            f"birth={c['t_birth']} score={c['score']:.3f}"
        )

    res = best["result"]
    video = res["video_f"][0]  # (T,C,H,W) float 0-1
    masks = res["masks"]  # (T,S,H,W) or similar
    if masks.ndim == 4 and masks.shape[-2:] != video.shape[-2:]:
        m = F.interpolate(
            masks.float(), size=video.shape[-2:], mode="nearest"
        )
        masks = m
    att = res["att"]
    side = best["side"]
    spare = best["spare"]
    if best["kind"] == "birth":
        times = [
            ("t = early", best["t_early"]),
            ("t = late (before birth)", best["t_late"]),
            ("t = new object appears", best["t_birth"]),
        ]
    else:
        times = [
            ("t = early (spare still quiet)", best["t_early"]),
            ("t = mid (attention spreads)", best["t_late"]),
            ("t = late (spare is diffuse)", best["t_birth"]),
        ]

    # Build columns
    cols = []
    H, W = video.shape[-2:]
    for title, t in times:
        frame = video[t]
        # spare attention map
        a = att[t, spare].numpy().reshape(side, side)
        heat_overlay = overlay_attn_on_frame(frame, a, alpha=0.6)
        # all-slot overlay
        mask_t = masks[t : t + 1].unsqueeze(0)  # (1,1,S,H,W) wait
        # masks is (T,S,H,W)
        ov = mix_videos_with_masks(
            frame.unsqueeze(0).unsqueeze(0),
            masks[t : t + 1].unsqueeze(0).float(),
            alpha=0.45,
        )[0, 0].permute(1, 2, 0).numpy().astype(np.uint8)
        rgb = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # mark new object bbox-ish on birth panel
        panels = [
            label_bar(Image.fromarray(rgb), title),
            label_bar(Image.fromarray(heat_overlay), f"spare slot #{spare} attention"),
            label_bar(Image.fromarray(ov), "all slots overlay"),
        ]
        # stack vertically for this time
        w = panels[0].width
        h = sum(p.height for p in panels)
        col = Image.new("RGB", (w, h), (255, 255, 255))
        y = 0
        for p in panels:
            col.paste(p, (0, y))
            y += p.height
        cols.append(col)

    # row labels on first column only via a left strip for the whole figure
    gap = 12
    total_w = sum(c.width for c in cols) + gap * (len(cols) - 1) + 170
    total_h = cols[0].height + 70
    fig = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(fig)
    try:
        font_t = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font_t = font_s = ImageFont.load_default()
    draw.text(
        (12, 16),
        "Slot Birth / Death Failure (real YTVIS frames)",
        fill=(15, 15, 15),
        font=font_t,
    )
    draw.text(
        (12, 44),
        "Unused always-active slots diffuse over time and lose anchor capacity for newly appearing objects.",
        fill=(60, 60, 60),
        font=font_s,
    )

    # left row labels
    row_names = ["Video", "Spare attn", "All slots"]
    row_h = cols[0].height // 3
    x0 = 10
    y0 = 70
    for i, name in enumerate(row_names):
        draw.text((x0, y0 + i * row_h + row_h // 2 - 8), name, fill=(30, 30, 30), font=font_s)

    x = 160
    for c in cols:
        fig.paste(c, (x, 70))
        x += c.width + gap

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.save(out)
    print(f"wrote {out} ({fig.size[0]}x{fig.size[1]})")

    # also dump the three key frames separately for PPT flexibility
    stem = out.with_suffix("")
    for title, t in times:
        safe = title.replace(" ", "_").replace("(", "").replace(")", "")
        rgb = (video[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        a = att[t, spare].numpy().reshape(side, side)
        heat = overlay_attn_on_frame(video[t], a, alpha=0.6)
        ov = mix_videos_with_masks(
            video[t].unsqueeze(0).unsqueeze(0),
            masks[t : t + 1].unsqueeze(0).float(),
            alpha=0.45,
        )[0, 0].permute(1, 2, 0).numpy().astype(np.uint8)
        Image.fromarray(rgb).save(f"{stem}_{safe}_rgb.png")
        Image.fromarray(heat).save(f"{stem}_{safe}_spare.png")
        Image.fromarray(ov).save(f"{stem}_{safe}_overlay.png")
    meta = out.with_suffix(".txt")
    meta.write_text(
        f"sample={best['idx']}\nkind={best['kind']}\nspare_slot={spare}\n"
        f"t_early={best['t_early']}\nt_late={best['t_late']}\nt_birth={best['t_birth']}\n"
        f"settings={args.settings}\nckpt={args.ckpt}\n"
    )
    print(f"wrote side panels under {stem}_*.png and {meta}")


if __name__ == "__main__":
    main()
