#!/usr/bin/env python3
"""Eval-only: gamma-sharpen attention, then G = a S a vs sqrt(a) S sqrt(a).

On a trained v39 checkpoint (forward unchanged). For each classified
slot-frame, recompute n8 π = λ1 - max(λ2, 0) under:

  gamma in {1, 2, 4}  x  {asa, sqrt}

Slot labels from decoder-hard IoU (same cuts as v39_merge_spectrum).
Occlude labels from GT cover/split (same as v39_occlusion_spectrum),
matched to the t-1 slot.

Usage (slotcurri image):
  python event_analysis/v39_sqrt_gamma_probe.py --dataset ytvis --max-clips 80
  python event_analysis/v39_sqrt_gamma_probe.py --dataset movi --max-clips 80
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slotcurri import configuration, data, models
from v39_merge_spectrum import _touch8
from v39_occlusion_spectrum import (
    RUNS,
    _FakeTrainer,
    _area_to_patches,
    _classify_transition,
    _iou_slot_gt,
    _n_cc8,
    _one_hot_gt,
    _resize_bool,
    _summarize,
    _to_device,
    n8_spectrum,
)

VARIANTS = (
    ("g1_asa", 1.0, "asa"),
    ("g1_sqrt", 1.0, "sqrt"),
    ("g2_asa", 2.0, "asa"),
    ("g2_sqrt", 2.0, "sqrt"),
    ("g4_asa", 4.0, "asa"),
    ("g4_sqrt", 4.0, "sqrt"),
)
SLOT_KINDS = ("exclusive", "merge_touch", "merge_apart", "ghost")
OCC_KINDS = ("cover", "split")


def _sharpen(att: torch.Tensor, gamma: float) -> torch.Tensor:
    if float(gamma) == 1.0:
        return att
    sharp = att.pow(float(gamma))
    return sharp / sharp.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _weights(att: torch.Tensor, gamma: float, mode: str) -> torch.Tensor:
    a = _sharpen(att, gamma)
    if mode == "sqrt":
        return a.clamp_min(0.0).sqrt()
    if mode == "asa":
        return a
    raise ValueError(f"mode must be 'asa' or 'sqrt', got {mode!r}")


def _auroc(pos: np.ndarray, neg: np.ndarray):
    if pos.size == 0 or neg.size == 0:
        return None
    # subsample if the pairwise broadcast would be huge
    rng = np.random.RandomState(0)
    if pos.size > 8000:
        pos = rng.choice(pos, 8000, replace=False)
    if neg.size > 8000:
        neg = rng.choice(neg, 8000, replace=False)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def _ratio(a, b):
    if a is None or b is None or abs(b) < 1e-12:
        return None
    return float(a / b)


@torch.no_grad()
def collect(model, loader, device, max_clips, ignore_bg, iou_ex, iou_m2, iou_match):
    slot_rows = {name: {k: [] for k in SLOT_KINDS} for name, _, _ in VARIANTS}
    occ_rows = {name: {k: [] for k in OCC_KINDS} for name, _, _ in VARIANTS}
    n_clips = 0
    n_occ_unmatched = 0

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

        specs = {}
        for name, gamma, mode in VARIANTS:
            w = _weights(att, gamma, mode)
            lam1, lam2, pi = n8_spectrum(w, bind, chunk=8)
            specs[name] = {
                "l1": lam1,
                "l2": lam2,
                "pi": pi,
                "gmax": pi / pi.amax(dim=-1, keepdim=True).clamp_min(1e-8),
            }

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

        labels = [[None] * n_slots for _ in range(t_len)]
        for t in range(t_len):
            for s in range(n_slots):
                scores = iou[t, s].cpu()
                if ignore_bg and scores.numel() > 0:
                    scores = scores.clone()
                    scores[0] = 0.0
                topv, topi = scores.topk(min(2, scores.numel()))
                v1 = float(topv[0])
                v2 = float(topv[1]) if topv.numel() > 1 else 0.0
                kind = None
                if v1 < 0.1:
                    kind = "ghost"
                elif v1 >= iou_ex and v2 < 0.12:
                    kind = "exclusive"
                elif v1 >= 0.2 and v2 >= iou_m2:
                    i1, i2 = int(topi[0]), int(topi[1])
                    if i1 in occ and i2 in occ:
                        pa = (occ[i1][t].reshape(grid, grid) > 0.25).cpu().numpy()
                        pb = (occ[i2][t].reshape(grid, grid) > 0.25).cpu().numpy()
                        kind = "merge_touch" if _touch8(pa, pb) else "merge_apart"
                    else:
                        kind = "merge_apart"
                labels[t][s] = kind
                if kind is None:
                    continue
                for name, _, _ in VARIANTS:
                    sp = specs[name]
                    slot_rows[name][kind].append(
                        {
                            "pi": float(sp["pi"][t, s].cpu()),
                            "l1": float(sp["l1"][t, s].cpu()),
                            "l2": float(sp["l2"][t, s].cpu()),
                            "gmax": float(sp["gmax"][t, s].cpu()),
                        }
                    )

        # GT occlude (cover/split), matched slot at t-1
        for oid in obj_ids:
            masks = seg == oid
            occ_pat = occ[oid]
            prev_run = None
            for t in range(1, t_len):
                kind = _classify_transition(
                    masks[t - 1],
                    masks[t],
                    seg[t],
                    oid,
                    occ_frac=0.25,
                    drop_ratio=0.75,
                    stay_ratio=0.85,
                    border=0.08,
                )
                if kind != "occlude":
                    prev_run = None
                    continue
                rem = occ_pat[t].reshape(grid, grid).cpu().numpy() > 0.25
                n_cc = _n_cc8(rem)
                kind = "split" if n_cc >= 2 else "cover"
                if prev_run == "occlude":
                    continue
                prev_run = "occlude"
                scores = iou[t - 1, :, oid]
                s = int(scores.argmax().item())
                if float(scores[s].item()) < iou_match:
                    n_occ_unmatched += 1
                    continue
                for name, _, _ in VARIANTS:
                    sp = specs[name]
                    pi_prev = float(sp["pi"][t - 1, s].cpu())
                    pi_now = float(sp["pi"][t, s].cpu())
                    occ_rows[name][kind].append(
                        {
                            "pi_prev": pi_prev,
                            "pi": pi_now,
                            "d_pi": pi_now - pi_prev,
                            "gmax_prev": float(sp["gmax"][t - 1, s].cpu()),
                            "gmax": float(sp["gmax"][t, s].cpu()),
                            "l1_prev": float(sp["l1"][t - 1, s].cpu()),
                            "l1": float(sp["l1"][t, s].cpu()),
                            "l2_prev": float(sp["l2"][t - 1, s].cpu()),
                            "l2": float(sp["l2"][t, s].cpu()),
                        }
                    )

        n_clips += 1
        if n_clips % 10 == 0:
            n_ex = len(slot_rows["g1_asa"]["exclusive"])
            n_gh = len(slot_rows["g1_asa"]["ghost"])
            print(
                f"  clips {n_clips}/{max_clips}  exclusive={n_ex} ghost={n_gh}",
                flush=True,
            )
        if n_clips >= max_clips:
            break
    return slot_rows, occ_rows, n_clips, n_occ_unmatched


def _pack_slots(rows):
    out = {}
    for name in rows:
        out[name] = {}
        for kind in SLOT_KINDS:
            arrs = {
                f: np.array([r[f] for r in rows[name][kind]], float)
                for f in ("pi", "l1", "l2", "gmax")
            }
            out[name][kind] = {f: _summarize(xs) for f, xs in arrs.items()}
        pos = np.array([r["pi"] for r in rows[name]["exclusive"]], float)
        ghost = np.array([r["pi"] for r in rows[name]["ghost"]], float)
        merge = np.array(
            [r["pi"] for k in ("merge_touch", "merge_apart") for r in rows[name][k]],
            float,
        )
        gpos = np.array([r["gmax"] for r in rows[name]["exclusive"]], float)
        gghost = np.array([r["gmax"] for r in rows[name]["ghost"]], float)
        out[name]["sep"] = {
            "auroc_ex_ghost": _auroc(pos, ghost),
            "auroc_ex_merge": _auroc(pos, merge),
            "mean_ex_over_ghost": _ratio(
                float(pos.mean()) if pos.size else None,
                float(ghost.mean()) if ghost.size else None,
            ),
            "median_ex_over_ghost": _ratio(
                float(np.median(pos)) if pos.size else None,
                float(np.median(ghost)) if ghost.size else None,
            ),
            "mean_gmax_ex": float(gpos.mean()) if gpos.size else None,
            "mean_gmax_ghost": float(gghost.mean()) if gghost.size else None,
        }
    return out


def _pack_occ(rows):
    out = {}
    for name in rows:
        out[name] = {}
        for kind in OCC_KINDS:
            recs = rows[name][kind]
            arrs = {
                f: np.array([r[f] for r in recs], float)
                for f in ("d_pi", "pi", "pi_prev", "gmax", "gmax_prev", "l1", "l2")
            }
            out[name][kind] = {f: _summarize(xs) for f, xs in arrs.items()}
        both = rows[name]["cover"] + rows[name]["split"]
        if both:
            d = np.array([r["d_pi"] for r in both], float)
            out[name]["occlude_all"] = {
                "d_pi": _summarize(d),
                "frac_drop": float((d < 0).mean()),
                "frac_drop_gt_0p02": float((d < -0.02).mean()),
            }
        else:
            out[name]["occlude_all"] = {"d_pi": {"n": 0}}
    return out


def _line(stat, kind, field):
    d = stat.get(kind, {}).get(field, {})
    if d.get("n", 0) == 0:
        return f"  {kind:14s} {field:8s} n=0"
    return (
        f"  {kind:14s} {field:8s} n={d['n']:6d}  "
        f"mean={d['mean']:.4f}  med={d['median']:.4f}  "
        f"p10={d['p10']:.4f}  p90={d['p90']:.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("movi", "ytvis"), default="ytvis")
    ap.add_argument("--max-clips", type=int, default=80)
    ap.add_argument("--iou-ex", type=float, default=0.3)
    ap.add_argument("--iou-m2", type=float, default=0.15)
    ap.add_argument("--iou-match", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--data-dir", default="/workspace/dataset")
    args = ap.parse_args()

    spec = RUNS[args.dataset]
    out = Path(args.out or f"logs/v39_sqrt_gamma_probe/{args.dataset}")
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
    print(
        f"dataset={args.dataset} device={device} max_clips={args.max_clips}",
        flush=True,
    )
    slot_rows, occ_rows, n_clips, n_occ_unmatched = collect(
        model,
        dataset.val_dataloader(),
        device,
        args.max_clips,
        spec["ignore_bg"],
        args.iou_ex,
        args.iou_m2,
        args.iou_match,
    )
    report = {
        "n_clips": n_clips,
        "n_occ_unmatched": n_occ_unmatched,
        "ckpt": spec["ckpt"],
        "slot": _pack_slots(slot_rows),
        "occlude": _pack_occ(occ_rows),
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2))

    print(f"\nclips={n_clips}  occ_unmatched={n_occ_unmatched}", flush=True)
    for name, gamma, mode in VARIANTS:
        print(f"\n======== {name}  gamma={gamma:g}  {mode} ========", flush=True)
        print("-- slot pi --", flush=True)
        for k in SLOT_KINDS:
            print(_line(report["slot"][name], k, "pi"), flush=True)
        print("-- slot gmax=pi/max(pi) --", flush=True)
        for k in SLOT_KINDS:
            print(_line(report["slot"][name], k, "gmax"), flush=True)
        sep = report["slot"][name]["sep"]
        print(
            f"  sep  auroc_ex_ghost={sep['auroc_ex_ghost']}  "
            f"auroc_ex_merge={sep['auroc_ex_merge']}  "
            f"mean_ex/ghost={sep['mean_ex_over_ghost']}  "
            f"med_ex/ghost={sep['median_ex_over_ghost']}",
            flush=True,
        )
        print("-- occlude d_pi --", flush=True)
        for k in OCC_KINDS:
            print(_line(report["occlude"][name], k, "d_pi"), flush=True)
        oa = report["occlude"][name]["occlude_all"]
        if oa.get("d_pi", {}).get("n", 0):
            print(
                f"  occlude_all    d_pi     n={oa['d_pi']['n']:6d}  "
                f"mean={oa['d_pi']['mean']:.4f}  "
                f"frac_drop={oa['frac_drop']:.3f}  "
                f"frac_drop>0.02={oa['frac_drop_gt_0p02']:.3f}",
                flush=True,
            )
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
