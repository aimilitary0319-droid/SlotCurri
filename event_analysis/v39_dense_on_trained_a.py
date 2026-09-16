#!/usr/bin/env python3
"""On trained v39 attention, does dense (v38) π separate exclusive vs merge?

Same slot labels as v39_merge_spectrum.py. Adds dense G_s = diag(a) S diag(a)
with S = full ReLU-cosine (not n8). Also reports λ1(1-λ2) vs λ1-λ2.

Usage:
  python event_analysis/v39_dense_on_trained_a.py --dataset movi --max-clips 80
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
from slotcurri.modules.video import (  # noqa: E402
    _relation_graph_S,
    _top2_algebraic_diag_s_diag,
)
from v39_merge_spectrum import (  # noqa: E402
    _touch8,
    _area_to_patches,
    _iou_slot_gt,
    _one_hot_gt,
    _resize_bool,
    _summarize,
    _to_device,
    n8_spectrum,
)
from v39_occlusion_spectrum import RUNS, _FakeTrainer  # noqa: E402
from slotcurri import configuration, data, models  # noqa: E402

EPS = 1e-6


def dense_spectrum(att: torch.Tensor, feat: torch.Tensor, chunk: int = 4):
    """att (T,S,N), feat (T,N,D) -> λ1, λ2, π of dense v38 G_s."""
    z = F.normalize(feat.float(), dim=-1)
    a = att.float()
    t, s, _ = a.shape
    lam1 = a.new_empty(t, s)
    lam2 = a.new_empty(t, s)
    step = max(int(chunk), 1)
    for i in range(0, t, step):
        s_mat = _relation_graph_S(z[i : i + step], EPS)
        l1, l2 = _top2_algebraic_diag_s_diag(s_mat, a[i : i + step], 16, EPS)
        lam1[i : i + step] = l1
        lam2[i : i + step] = l2
    pi = (lam1 - lam2.clamp_min(0.0)).clamp_min(0.0)
    return lam1, lam2, pi


def _pack(l1, l2, pi):
    l2p = max(float(l2), 0.0)
    l1f = float(l1)
    pif = float(pi)
    r = l2p / max(l1f, EPS)
    return {
        "l1": l1f,
        "l2": float(l2),
        "pi": pif,
        "r": r,
        "prod": l1f * max(1.0 - l2p, 0.0),  # λ1(1-λ2) raw
        "relprod": pif,  # λ1(1-λ2/λ1) = π
    }


def _auroc(pure: np.ndarray, impure: np.ndarray) -> float:
    """Higher score = more pure. AUROC of exclusive vs merge."""
    if pure.size == 0 or impure.size == 0:
        return float("nan")
    wins = 0.0
    for p in pure:
        wins += float((p > impure).sum()) + 0.5 * float((p == impure).sum())
    return wins / (pure.size * impure.size)


def _cohend(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    v = 0.5 * (a.var() + b.var())
    if v <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / math.sqrt(v))


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
        amp = torch.cuda.amp.autocast(enabled=device.type == "cuda")
        with amp:
            outputs = model.forward(batch, train=False, cycle=False)
            aux = model.aux_forward(batch, outputs)

        att = outputs["processor"]["state_attn_mask"].float()[0]
        bind = outputs["encoder"].get("backbone_key")
        if bind is None:
            bind = outputs["encoder"]["backbone_features"]
        bind = bind.float()[0]
        n8_l1, n8_l2, n8_pi = n8_spectrum(att, bind, chunk=8)
        d_l1, d_l2, d_pi = dense_spectrum(att, bind, chunk=4)

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
            a = torch.stack(a_stack, dim=0).unsqueeze(0)
            nl1, nl2, np_ = n8_spectrum(a, bind[t : t + 1], chunk=8)
            dl1, dl2, dp = dense_spectrum(a, bind[t : t + 1], chunk=4)
            for i, tag in enumerate(tags):
                oracle[tag].append(
                    {
                        "n8": _pack(nl1[0, i], nl2[0, i], np_[0, i]),
                        "dense": _pack(dl1[0, i], dl2[0, i], dp[0, i]),
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
                    "n8": _pack(n8_l1[t, s], n8_l2[t, s], n8_pi[t, s]),
                    "dense": _pack(d_l1[t, s], d_l2[t, s], d_pi[t, s]),
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
        if n_clips % 5 == 0:
            print(f"  clips {n_clips}/{max_clips}", flush=True)
        if n_clips >= max_clips:
            break
    return oracle, slots, n_clips


def _field(rows, graph, key):
    return np.array([r[graph][key] for r in rows], float)


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
    out = Path(args.out or f"logs/v39_dense_on_trained_a/{args.dataset}")
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
    oracle, slots, n_clips = collect(
        model,
        dataset.val_dataloader(),
        device,
        args.max_clips,
        spec["ignore_bg"],
        args.iou_ex,
        args.iou_m2,
    )

    def summarize_group(store):
        report = {}
        for kind, rows in store.items():
            report[kind] = {"n": len(rows)}
            for graph in ("n8", "dense"):
                keys = ("pi", "l1", "l2", "r", "prod")
                if not rows:
                    report[kind][graph] = {k: {"n": 0} for k in keys}
                else:
                    report[kind][graph] = {
                        k: _summarize(_field(rows, graph, k)) for k in keys
                    }
        return report

    report = {
        "n_clips": n_clips,
        "oracle": summarize_group(oracle),
        "slot": summarize_group(slots),
        "auroc_exclusive_vs_merge_touch": {},
    }
    ex = slots["exclusive"]
    mt = slots["merge_touch"] + slots["merge_apart"]
    for graph in ("n8", "dense"):
        for key in ("pi", "l1", "r", "prod"):
            pure = _field(ex, graph, key)
            impure = _field(mt, graph, key)
            report["auroc_exclusive_vs_merge_touch"][f"{graph}_{key}"] = {
                "auroc": _auroc(pure, impure),
                "cohend_ex_minus_merge": _cohend(pure, impure),
                "n_ex": int(pure.size),
                "n_merge": int(impure.size),
            }

    (out / "summary.json").write_text(json.dumps(report, indent=2))

    def line(group, kind, graph, key):
        d = report[group][kind][graph][key]
        if d.get("n", 0) == 0:
            return f"  {kind:14s} {graph:5s} {key:4s} n=0"
        return (
            f"  {kind:14s} {graph:5s} {key:4s} n={d['n']:5d}  "
            f"mean={d['mean']:.4f}  med={d['median']:.4f}  "
            f"p10={d['p10']:.4f}  p90={d['p90']:.4f}"
        )

    print(f"\nclips={n_clips}", flush=True)
    print("\n=== ORACLE ===", flush=True)
    for graph in ("n8", "dense"):
        for key in ("pi", "l1", "l2", "r", "prod"):
            print(f"-- {graph} {key} --", flush=True)
            for k in ("excl", "union_touch", "union_apart"):
                print(line("oracle", k, graph, key), flush=True)
    print("\n=== SLOT ===", flush=True)
    for graph in ("n8", "dense"):
        for key in ("pi", "l1", "l2", "r", "prod"):
            print(f"-- {graph} {key} --", flush=True)
            for k in ("exclusive", "merge_touch", "merge_apart", "ghost"):
                print(line("slot", k, graph, key), flush=True)
    print("\n=== AUROC exclusive vs merge (higher π should mark exclusive) ===", flush=True)
    for k, v in report["auroc_exclusive_vs_merge_touch"].items():
        print(
            f"  {k:16s}  AUROC={v['auroc']:.3f}  d={v['cohend_ex_minus_merge']:+.3f}  "
            f"n_ex={v['n_ex']} n_merge={v['n_merge']}",
            flush=True,
        )
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
