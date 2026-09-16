#!/usr/bin/env python3
"""Does v39 π = λ1-λ2+ actually mark two-object merges?

Two views on a trained v39 ckpt:
  oracle  GT exclusive mask vs GT A∪B (touching vs separated)
  slot    decoder-hard IoU: exclusive (top2 small) vs merge (two GT objects)

Usage:
  python event_analysis/v39_merge_spectrum.py --dataset movi --max-clips 80
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slotcurri import configuration, data, models
from v39_occlusion_spectrum import (
    RUNS,
    _FakeTrainer,
    _area_to_patches,
    _iou_slot_gt,
    _n_cc8,
    _one_hot_gt,
    _resize_bool,
    _summarize,
    _to_device,
    n8_spectrum,
)


def _touch8(a: np.ndarray, b: np.ndarray) -> bool:
    g = a.shape[0]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            aa = a[max(dy, 0) : g + min(dy, 0), max(dx, 0) : g + min(dx, 0)]
            bb = b[max(-dy, 0) : g + min(-dy, 0), max(-dx, 0) : g + min(-dx, 0)]
            if np.any(aa & bb):
                return True
    return False


@torch.no_grad()
def collect(model, loader, device, max_clips, ignore_bg, iou_ex, iou_m2):
    oracle = {k: [] for k in ("excl", "union_touch", "union_apart")}
    slots = {k: [] for k in ("exclusive", "merge_touch", "merge_apart", "ghost")}
    n_clips = 0
    for batch in loader:
        if "batch_padding_mask" in batch:
            batch = model._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                continue
        batch = _to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model.forward(batch, train=False, cycle=False)
            aux = model.aux_forward(batch, outputs)

        att = outputs["processor"]["state_attn_mask"].float()[0]
        bind = outputs["encoder"].get("backbone_key")
        if bind is None:
            bind = outputs["encoder"]["backbone_features"]
        bind = bind.float()[0]
        lam1, lam2, pi = n8_spectrum(att, bind, chunk=8)
        mass = att.sum(dim=-1) / att.shape[-1]

        key = (
            "decoder_masks_vis_hard"
            if "decoder_masks_vis_hard" in aux
            else "decoder_masks_hard"
        )
        pred = aux[key].bool()[0]
        gt_oh = _one_hot_gt(batch["segmentations"])[0]
        gt_oh = _resize_bool(gt_oh.unsqueeze(0), pred.shape[-2:])[0]
        if ignore_bg and gt_oh.shape[1] > 0:
            gt_oh = gt_oh.clone()
            gt_oh[:, 0] = False
        iou = _iou_slot_gt(pred, gt_oh)
        if ignore_bg and iou.shape[-1] > 0:
            iou[:, :, 0] = 0.0

        seg = batch["segmentations"][0]
        if seg.ndim == 4:
            seg = seg.float().argmax(dim=1)
        seg = F.interpolate(
            seg.float().unsqueeze(1), size=pred.shape[-2:], mode="nearest"
        ).squeeze(1).long()

        t_len, n_slots, n_tokens = att.shape
        grid = int(round(math.sqrt(n_tokens)))
        obj_ids = [int(i) for i in seg.unique().tolist() if int(i) != 0]
        occ = {oid: _area_to_patches(seg == oid, grid) for oid in obj_ids}

        # oracle unions of large enough objects, a few pairs per frame
        for t in range(t_len):
            sizes = sorted(
                ((oid, float(occ[oid][t].sum())) for oid in obj_ids),
                key=lambda x: -x[1],
            )
            sizes = [(o, n) for o, n in sizes if n >= 4]
            if not sizes:
                continue
            a_stack = []
            tags = []
            oa = sizes[0][0]
            a_stack.append(occ[oa][t])
            tags.append("excl")
            for ob, _ in sizes[1:4]:
                union = torch.maximum(occ[oa][t], occ[ob][t])
                pa = (occ[oa][t].reshape(grid, grid) > 0.25).cpu().numpy()
                pb = (occ[ob][t].reshape(grid, grid) > 0.25).cpu().numpy()
                tag = "union_touch" if _touch8(pa, pb) else "union_apart"
                a_stack.append(union)
                tags.append(tag)
            a = torch.stack(a_stack, dim=0).unsqueeze(0)  # (1,P,N)
            l1, l2, p = n8_spectrum(a, bind[t : t + 1], chunk=8)
            for i, tag in enumerate(tags):
                oracle[tag].append(
                    {
                        "l1": float(l1[0, i].cpu()),
                        "l2": float(l2[0, i].cpu()),
                        "pi": float(p[0, i].cpu()),
                        "r": float(
                            max(float(l2[0, i].cpu()), 0.0)
                            / max(float(l1[0, i].cpu()), 1e-6)
                        ),
                    }
                )

        for t in range(t_len):
            for s in range(n_slots):
                scores = iou[t, s].cpu()
                if ignore_bg and scores.numel() > 0:
                    scores = scores.clone()
                    scores[0] = 0.0
                topv, topi = scores.topk(min(2, scores.numel()))
                v1 = float(topv[0])
                v2 = float(topv[1]) if topv.numel() > 1 else 0.0
                rec = {
                    "l1": float(lam1[t, s].cpu()),
                    "l2": float(lam2[t, s].cpu()),
                    "pi": float(pi[t, s].cpu()),
                    "r": float(
                        max(float(lam2[t, s].cpu()), 0.0)
                        / max(float(lam1[t, s].cpu()), 1e-6)
                    ),
                    "mass": float(mass[t, s].cpu()),
                    "iou1": v1,
                    "iou2": v2,
                }
                if v1 < 0.1:
                    slots["ghost"].append(rec)
                    continue
                if v1 >= iou_ex and v2 < 0.12:
                    slots["exclusive"].append(rec)
                    continue
                if v1 >= 0.2 and v2 >= iou_m2:
                    i1, i2 = int(topi[0]), int(topi[1])
                    if i1 in occ and i2 in occ:
                        pa = (occ[i1][t].reshape(grid, grid) > 0.25).cpu().numpy()
                        pb = (occ[i2][t].reshape(grid, grid) > 0.25).cpu().numpy()
                        tag = "merge_touch" if _touch8(pa, pb) else "merge_apart"
                    else:
                        tag = "merge_apart"
                    slots[tag].append(rec)

        n_clips += 1
        if n_clips % 10 == 0:
            print(f"  clips {n_clips}/{max_clips}", flush=True)
        if n_clips >= max_clips:
            break
    return oracle, slots, n_clips


def _hist(path, groups, field, title):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    colors = {
        "exclusive": "#2ca02c",
        "excl": "#2ca02c",
        "merge_apart": "#d62728",
        "union_apart": "#d62728",
        "merge_touch": "#ff7f0e",
        "union_touch": "#ff7f0e",
        "ghost": "#7f7f7f",
    }
    for name, rows in groups.items():
        xs = np.array([r[field] for r in rows], float)
        if xs.size == 0:
            continue
        ax.hist(
            xs,
            bins=36,
            density=True,
            histtype="step",
            lw=1.7,
            color=colors.get(name, "k"),
            label=f"{name} n={xs.size} μ={xs.mean():.3f}",
        )
    ax.set_xlabel(field)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("movi", "ytvis"), default="movi")
    ap.add_argument("--max-clips", type=int, default=80)
    ap.add_argument("--iou-ex", type=float, default=0.3)
    ap.add_argument("--iou-m2", type=float, default=0.15)
    ap.add_argument("--out", default=None)
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--data-dir", default="/workspace/dataset")
    args = ap.parse_args()

    spec = RUNS[args.dataset]
    out = Path(args.out or f"logs/v39_merge_spectrum/{args.dataset}")
    out.mkdir(parents=True, exist_ok=True)
    config = configuration.load_config(spec["config"])
    config.model.visualize = False
    config.dataset.val_batch_size = 1
    config.dataset.num_val_workers = 0
    model = models.build(config.model, config.optimizer)
    model.load_weights_from_checkpoint(spec["ckpt"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    model._trainer = _FakeTrainer(args.step)
    dataset = data.build(config.dataset, data_dir=args.data_dir)
    dataset.setup("validate")
    print(f"dataset={args.dataset} device={device} max_clips={args.max_clips}", flush=True)
    oracle, slots, n_clips = collect(
        model,
        dataset.val_dataloader(),
        device,
        args.max_clips,
        spec["ignore_bg"],
        args.iou_ex,
        args.iou_m2,
    )

    def pack(store):
        return {
            k: {f: _summarize(np.array([r[f] for r in rows], float)) for f in ("pi", "l1", "l2", "r")}
            for k, rows in store.items()
        }

    report = {"n_clips": n_clips, "oracle": pack(oracle), "slot": pack(slots)}
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    _hist(out / "oracle_pi.png", oracle, "pi", r"oracle $a$: $\pi=\lambda_1-\lambda_2^+$")
    _hist(out / "oracle_r.png", oracle, "r", r"oracle $a$: $\lambda_2^+/\lambda_1$")
    _hist(out / "slot_pi.png", slots, "pi", r"slot attention: $\pi$")
    _hist(out / "slot_r.png", slots, "r", r"slot attention: $\lambda_2^+/\lambda_1$")
    _hist(out / "slot_l2.png", slots, "l2", r"slot attention: $\lambda_2$")

    def line(group, kind, field):
        d = report[group][kind][field]
        if d.get("n", 0) == 0:
            return f"  {kind:14s} {field:4s} n=0"
        return (
            f"  {kind:14s} {field:4s} n={d['n']:6d}  "
            f"mean={d['mean']:.3f}  med={d['median']:.3f}  "
            f"p10={d['p10']:.3f}  p90={d['p90']:.3f}"
        )

    print(f"\nclips={n_clips}", flush=True)
    print("\n=== ORACLE GT masks ===", flush=True)
    for field in ("pi", "l1", "l2", "r"):
        print(f"-- {field} --", flush=True)
        for k in ("excl", "union_touch", "union_apart"):
            print(line("oracle", k, field), flush=True)
    print("\n=== SLOT (decoder IoU) ===", flush=True)
    for field in ("pi", "l1", "l2", "r"):
        print(f"-- {field} --", flush=True)
        for k in ("exclusive", "merge_touch", "merge_apart", "ghost"):
            print(line("slot", k, field), flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
