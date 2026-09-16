#!/usr/bin/env python3
"""How do v39 n8 π, λ1, λ2 move when a GT object is actually occluded?

Eval-only on a trained v39 checkpoint. Events are GT-centric (not slot
occupied→empty):

  occlude  ≥occ_frac of the object's previous pixels become another FG id,
           and its area drops
  split    occlude and the remaining visible patches have ≥2 8-connected
           components (pole / line cut)
  cover    occlude and remaining support is 0 or 1 component
  exit     area drop, most previous pixels go to background, mask was
           near the image border
  track    stable visibility, almost no pixels stolen by another instance

For each event, record the n8 spectrum of (1) the slot matched to the
object at t-1 and (2) the GT object mask itself as a (oracle support).

Usage (slotcurri image):
  python event_analysis/v39_occlusion_spectrum.py --dataset movi --max-clips 250
  python event_analysis/v39_occlusion_spectrum.py --dataset ytvis --max-clips 80
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from slotcurri import configuration, data, models
from slotcurri.modules.video import _top2_algebraic_n8


class _FakeTrainer:
    def __init__(self, step: int):
        self.global_step = step


RUNS = {
    "movi": {
        "config": "configs/slotcurri/movi_c_attnmass_v39.yaml",
        "ckpt": "logs/_movi_c_attnmass_v39/checkpoints/slotcurri_step=step=100000-v1.ckpt",
        "ignore_bg": True,
    },
    "ytvis": {
        "config": "configs/slotcurri/ytvis2021_attnmass_v39.yaml",
        "ckpt": "logs/_ytvis_attnmass_v39/checkpoints/slotcurri_step=step=100000-v1.ckpt",
        "ignore_bg": False,
    },
}

REL_WINDOW = (-3, -2, -1, 0, 1, 2, 3)
EPS = 1e-6


def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _one_hot_gt(seg: torch.Tensor) -> torch.Tensor:
    if seg.ndim == 5:
        return seg.bool()
    ncls = int(seg.max().item()) + 1
    b, t, h, w = seg.shape
    oh = torch.zeros(b, t, ncls, h, w, dtype=torch.bool, device=seg.device)
    for c in range(ncls):
        oh[:, :, c] = seg == c
    return oh


def _resize_bool(masks: torch.Tensor, hw) -> torch.Tensor:
    if masks.shape[-2:] == hw:
        return masks.bool()
    b, t, c, _, _ = masks.shape
    m = masks.float().reshape(b * t, c, *masks.shape[-2:])
    m = F.interpolate(m, size=hw, mode="nearest")
    return m.reshape(b, t, c, *hw).bool()


def _iou_slot_gt(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    t, s, h, w = pred.shape
    c = gt.shape[1]
    p = pred.reshape(t, s, h * w).float()
    g = gt.reshape(t, c, h * w).float()
    inter = torch.einsum("tsn,tcn->tsc", p, g)
    psum = p.sum(-1).unsqueeze(-1)
    gsum = g.sum(-1).unsqueeze(1)
    return inter / (psum + gsum - inter).clamp_min(1.0)


def _summarize(xs: np.ndarray) -> Dict:
    if xs.size == 0:
        return {"n": 0}
    return {
        "n": int(xs.size),
        "mean": float(xs.mean()),
        "median": float(np.median(xs)),
        "p10": float(np.percentile(xs, 10)),
        "p90": float(np.percentile(xs, 90)),
        "std": float(xs.std()),
    }


@torch.no_grad()
def n8_spectrum(att: torch.Tensor, feat: torch.Tensor, chunk: int = 8):
    """att (T,S,N), feat (T,N,D) -> λ1, λ2, π each (T,S). Same solver as v39."""
    z = F.normalize(feat.float(), dim=-1)
    a = att.float()
    t, s, _ = a.shape
    lam1 = a.new_empty(t, s)
    lam2 = a.new_empty(t, s)
    step = max(int(chunk), 1)
    for i in range(0, t, step):
        l1, l2 = _top2_algebraic_n8(z[i : i + step], a[i : i + step], 16, EPS)
        lam1[i : i + step] = l1
        lam2[i : i + step] = l2
    lam2p = lam2.clamp_min(0.0)
    pi = (lam1 - lam2p).clamp_min(0.0)
    return lam1, lam2, pi


def _area_to_patches(mask_thw: torch.Tensor, grid: int) -> torch.Tensor:
    """(T,H,W) bool -> (T, N) area fraction in [0, 1]."""
    t, h, w = mask_thw.shape
    m = mask_thw.float().unsqueeze(1)
    m = F.interpolate(m, size=(grid, grid), mode="area")
    return m.reshape(t, grid * grid)


def _n_cc8(binary_hw: np.ndarray) -> int:
    g = binary_hw.shape[0]
    seen = np.zeros((g, g), dtype=bool)
    n = 0
    ys, xs = np.where(binary_hw)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        n += 1
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < g and 0 <= xx < g and binary_hw[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
    return n


def _near_border(mask_hw: torch.Tensor, margin: float = 0.08) -> bool:
    ys, xs = torch.where(mask_hw)
    if ys.numel() == 0:
        return False
    h, w = mask_hw.shape
    my, mx = margin * h, margin * w
    return bool(
        ys.min() < my
        or ys.max() > h - 1 - my
        or xs.min() < mx
        or xs.max() > w - 1 - mx
    )


def _classify_transition(
    prev: torch.Tensor,
    cur: torch.Tensor,
    cur_full: torch.Tensor,
    obj_id: int,
    occ_frac: float,
    drop_ratio: float,
    stay_ratio: float,
    border: float,
) -> Optional[str]:
    """prev/cur are (H,W) bool for this object; cur_full is (H,W) int ids."""
    n_prev = int(prev.sum().item())
    if n_prev < 8:
        return None
    n_cur = int(cur.sum().item())
    stolen = cur_full[prev]
    n_other = int(((stolen != obj_id) & (stolen != 0)).sum().item())
    n_bg = int((stolen == 0).sum().item())
    frac_other = n_other / n_prev
    frac_bg = n_bg / n_prev
    area_ratio = n_cur / max(n_prev, 1)

    if frac_other >= occ_frac and area_ratio <= drop_ratio:
        return "occlude"
    if (
        area_ratio <= drop_ratio
        and frac_bg >= occ_frac
        and frac_other < occ_frac
        and _near_border(prev, border)
    ):
        return "exit"
    if area_ratio >= stay_ratio and frac_other < 0.05 and n_cur >= 8:
        return "track"
    return None


@torch.no_grad()
def collect(
    model,
    loader,
    device: torch.device,
    max_clips: int,
    iou_thr: float,
    ignore_bg: bool,
    occ_frac: float,
    drop_ratio: float,
    stay_ratio: float,
    border: float,
    cc_thr: float,
    track_max_per_clip: int,
):
    events: List[Dict] = []
    traj = defaultdict(lambda: {rel: {"pi": [], "l1": [], "l2": [], "gt_pi": [], "gt_l1": [], "gt_l2": []} for rel in REL_WINDOW})
    n_clips = 0
    n_unmatched = 0
    pi_agree = []

    for batch in loader:
        if "batch_padding_mask" in batch:
            batch = model._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                continue
        batch = _to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model.forward(batch, train=False, cycle=False)
            aux = model.aux_forward(batch, outputs)

        att = outputs["processor"]["state_attn_mask"].float()[0]  # (T,S,N)
        bind = outputs["encoder"].get("backbone_key")
        if bind is None:
            bind = outputs["encoder"]["backbone_features"]
        bind = bind.float()[0]  # (T,N,C)
        u = outputs["processor"]["state"].float()[0]
        hat_next = outputs["processor"]["state_predicted"].float()[0]
        gate_conf = outputs["processor"].get("gate_conf")
        if gate_conf is not None:
            gate_conf = gate_conf.float()[0]

        t_len, n_slots, n_tokens = att.shape
        grid = int(round(math.sqrt(n_tokens)))
        if grid * grid != n_tokens:
            raise RuntimeError(f"non-square tokens N={n_tokens}")

        lam1, lam2, pi = n8_spectrum(att, bind, chunk=8)
        if gate_conf is not None:
            pi_agree.append(float((pi - gate_conf).abs().mean().cpu()))

        mass = att.sum(dim=-1) / float(n_tokens)
        c_id = torch.zeros(t_len, n_slots, device=att.device)
        if t_len >= 2:
            c_id[1:] = F.cosine_similarity(hat_next[:-1], u[1:], dim=-1).clamp(-1.0, 1.0)

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
        iou = _iou_slot_gt(pred, gt_oh)  # (T,S,C)
        if ignore_bg and iou.shape[-1] > 0:
            iou[:, :, 0] = 0.0

        seg = batch["segmentations"][0]
        if seg.ndim == 4:
            # (T,C,H,W) one-hot / multi-channel. Bool one-hot needs a float argmax.
            seg = seg.float().argmax(dim=1)
        seg = F.interpolate(
            seg.float().unsqueeze(1), size=pred.shape[-2:], mode="nearest"
        ).squeeze(1).long()
        t_len, h, w = seg.shape

        obj_ids = [int(i) for i in seg.unique().tolist() if int(i) != 0]
        if ignore_bg:
            obj_ids = [i for i in obj_ids if i != 0]

        # Oracle GT support spectrum: stack objects as a fake slot dim per frame.
        gt_occ = {}
        gt_spec = {}
        for oid in obj_ids:
            occ = _area_to_patches(seg == oid, grid)
            gt_occ[oid] = occ
            a_gt = occ.unsqueeze(1)  # (T,1,N)
            l1, l2, p = n8_spectrum(a_gt, bind, chunk=8)
            gt_spec[oid] = (l1[:, 0], l2[:, 0], p[:, 0])

        track_taken = 0
        rng = np.random.RandomState(n_clips + 17)

        def emit(oid, t, kind, n_cc, masks, areas, gl1, gl2, gpi):
            nonlocal n_unmatched
            best_s = int(iou[t - 1, :, oid].argmax().item()) if oid < iou.shape[-1] else -1
            best_iou = (
                float(iou[t - 1, best_s, oid].item()) if oid < iou.shape[-1] else 0.0
            )
            matched = best_iou >= iou_thr
            if not matched:
                n_unmatched += 1
                s = -1
            else:
                s = best_s

            stolen = seg[t][masks[t - 1]]
            other = stolen[(stolen != oid) & (stolen != 0)]
            occluder = int(other.mode().values.item()) if other.numel() else 0

            rec = {
                "clip": n_clips,
                "t": t,
                "oid": oid,
                "kind": kind,
                "n_cc": n_cc,
                "area_prev": int(areas[t - 1].item()),
                "area_cur": int(areas[t].item()),
                "area_ratio": float(areas[t].item() / max(int(areas[t - 1].item()), 1)),
                "slot": s,
                "iou_prev": best_iou,
                "matched": int(matched),
                "occluder": occluder,
                "gt_l1_prev": float(gl1[t - 1].cpu()),
                "gt_l2_prev": float(gl2[t - 1].cpu()),
                "gt_pi_prev": float(gpi[t - 1].cpu()),
                "gt_l1": float(gl1[t].cpu()),
                "gt_l2": float(gl2[t].cpu()),
                "gt_pi": float(gpi[t].cpu()),
            }
            rec["d_gt_pi"] = rec["gt_pi"] - rec["gt_pi_prev"]
            rec["d_gt_l1"] = rec["gt_l1"] - rec["gt_l1_prev"]
            rec["d_gt_l2"] = rec["gt_l2"] - rec["gt_l2_prev"]
            if matched:
                rec.update(
                    {
                        "l1_prev": float(lam1[t - 1, s].cpu()),
                        "l2_prev": float(lam2[t - 1, s].cpu()),
                        "pi_prev": float(pi[t - 1, s].cpu()),
                        "l1": float(lam1[t, s].cpu()),
                        "l2": float(lam2[t, s].cpu()),
                        "pi": float(pi[t, s].cpu()),
                        "mass_prev": float(mass[t - 1, s].cpu()),
                        "mass": float(mass[t, s].cpu()),
                        "c_id": float(c_id[t, s].cpu()),
                    }
                )
                rec["d_pi"] = rec["pi"] - rec["pi_prev"]
                rec["d_l1"] = rec["l1"] - rec["l1_prev"]
                rec["d_l2"] = rec["l2"] - rec["l2_prev"]
                rec["iou_occluder"] = (
                    float(iou[t, s, occluder].item())
                    if occluder and occluder < iou.shape[-1]
                    else 0.0
                )
                rec["iou_obj"] = (
                    float(iou[t, s, oid].item()) if oid < iou.shape[-1] else 0.0
                )
            events.append(rec)
            for rel in REL_WINDOW:
                tt = t + rel
                if not (0 <= tt < t_len):
                    continue
                traj[kind][rel]["gt_pi"].append(float(gpi[tt].cpu()))
                traj[kind][rel]["gt_l1"].append(float(gl1[tt].cpu()))
                traj[kind][rel]["gt_l2"].append(float(gl2[tt].cpu()))
                if matched:
                    traj[kind][rel]["pi"].append(float(pi[tt, s].cpu()))
                    traj[kind][rel]["l1"].append(float(lam1[tt, s].cpu()))
                    traj[kind][rel]["l2"].append(float(lam2[tt, s].cpu()))

        for oid in obj_ids:
            masks = seg == oid
            areas = masks.flatten(1).sum(dim=-1)
            occ = gt_occ[oid]
            gl1, gl2, gpi = gt_spec[oid]
            labels: List[Optional[Tuple[str, int]]] = [None] * t_len
            for t in range(1, t_len):
                kind = _classify_transition(
                    masks[t - 1],
                    masks[t],
                    seg[t],
                    oid,
                    occ_frac,
                    drop_ratio,
                    stay_ratio,
                    border,
                )
                if kind is None:
                    continue
                rem = occ[t].reshape(grid, grid).cpu().numpy() > cc_thr
                n_cc = _n_cc8(rem)
                if kind == "occlude":
                    kind = "split" if n_cc >= 2 else "cover"
                labels[t] = (kind, n_cc)

            prev_run = None
            for t in range(1, t_len):
                if labels[t] is None:
                    prev_run = None
                    continue
                kind, n_cc = labels[t]
                if kind == "track":
                    prev_run = None
                    if track_taken >= track_max_per_clip or rng.rand() > 0.25:
                        continue
                    track_taken += 1
                    emit(oid, t, kind, n_cc, masks, areas, gl1, gl2, gpi)
                    continue
                run = "occlude" if kind in ("cover", "split") else kind
                if run == prev_run:
                    continue
                prev_run = run
                emit(oid, t, kind, n_cc, masks, areas, gl1, gl2, gpi)

        n_clips += 1
        if n_clips % 10 == 0:
            print(f"  clips {n_clips}/{max_clips}  events={len(events)}", flush=True)
        if n_clips >= max_clips:
            break

    return {
        "events": events,
        "traj": traj,
        "n_clips": n_clips,
        "n_unmatched": n_unmatched,
        "pi_agree_mae": float(np.mean(pi_agree)) if pi_agree else None,
    }


def _mean_se(xs: List[float]) -> Tuple[Optional[float], Optional[float], int]:
    if not xs:
        return None, None, 0
    a = np.asarray(xs, dtype=np.float64)
    return float(a.mean()), float(a.std() / max(math.sqrt(len(a)), 1.0)), int(len(a))


def _plot_traj(path: Path, traj, keys, title: str, ylabel: str):
    colors = {
        "track": "#2ca02c",
        "cover": "#ff7f0e",
        "split": "#d62728",
        "exit": "#1f77b4",
        "occlude": "#9467bd",
    }
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.6), sharex=True)
    if len(keys) == 1:
        axes = [axes]
    xs = list(REL_WINDOW)
    for ax, key, lab in zip(axes, keys, ylabel if isinstance(ylabel, (list, tuple)) else [ylabel] * len(keys)):
        for kind in ("track", "cover", "split", "exit"):
            means, los, his = [], [], []
            ok = False
            for rel in xs:
                m, se, n = _mean_se(traj[kind][rel][key])
                if m is None or n < 5:
                    means.append(np.nan)
                    los.append(np.nan)
                    his.append(np.nan)
                else:
                    ok = True
                    means.append(m)
                    los.append(m - se)
                    his.append(m + se)
            if not ok:
                continue
            ax.plot(xs, means, color=colors[kind], lw=1.8, label=f"{kind}")
            ax.fill_between(xs, los, his, color=colors[kind], alpha=0.15)
        ax.axvline(0.0, color="0.7", ls="--", lw=0.8)
        ax.set_xlabel("frame relative to onset")
        ax.set_ylabel(lab)
        ax.set_xticks(xs)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_delta_scatter(path: Path, events: List[Dict], src: str):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    colors = {"track": "#2ca02c", "cover": "#ff7f0e", "split": "#d62728", "exit": "#1f77b4"}
    for kind, col in colors.items():
        rows = [e for e in events if e["kind"] == kind and f"d_{src}l1" in e]
        if not rows:
            continue
        ax.scatter(
            [e[f"d_{src}l1"] for e in rows],
            [e[f"d_{src}l2"] for e in rows],
            s=14,
            alpha=0.55,
            c=col,
            label=f"{kind} n={len(rows)}",
            edgecolors="none",
        )
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.axvline(0.0, color="0.7", lw=0.6)
    prefix = "" if src == "" else "gt "
    ax.set_xlabel(rf"$\Delta {prefix}\lambda_1$")
    ax.set_ylabel(rf"$\Delta {prefix}\lambda_2$")
    ax.set_title(f"{prefix}n8 spectrum change at onset".strip())
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_hists(path: Path, events: List[Dict], field: str, title: str, xlim=None):
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    colors = {"track": "#2ca02c", "cover": "#ff7f0e", "split": "#d62728", "exit": "#1f77b4"}
    for kind, col in colors.items():
        xs = np.asarray([e[field] for e in events if e["kind"] == kind and field in e], dtype=np.float64)
        if xs.size == 0:
            continue
        ax.hist(
            xs,
            bins=32,
            range=xlim,
            density=True,
            histtype="step",
            linewidth=1.7,
            color=col,
            label=f"{kind} n={xs.size}  μ={xs.mean():.3f}",
        )
    ax.axvline(0.0, color="0.7", ls="--", lw=0.8)
    ax.set_xlabel(field)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("movi", "ytvis"), default="movi")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--max-clips", type=int, default=250)
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--occ-frac", type=float, default=0.25)
    ap.add_argument("--drop-ratio", type=float, default=0.75)
    ap.add_argument("--stay-ratio", type=float, default=0.85)
    ap.add_argument("--border", type=float, default=0.08)
    ap.add_argument("--cc-thr", type=float, default=0.25)
    ap.add_argument("--track-max-per-clip", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    spec = RUNS[args.dataset]
    config_path = args.config or spec["config"]
    ckpt_path = args.ckpt or spec["ckpt"]
    out_dir = Path(args.out or f"logs/v39_occlusion_spectrum/{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = configuration.load_config(config_path)
    config.model.visualize = False
    config.dataset.val_batch_size = 1
    config.dataset.num_val_workers = 0

    model = models.build(config.model, config.optimizer)
    model.load_weights_from_checkpoint(ckpt_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    model._trainer = _FakeTrainer(args.step)
    model.amc_state_identity_cos = False

    dataset = data.build(config.dataset, data_dir=args.data_dir)
    dataset.setup("validate")
    loader = dataset.val_dataloader()

    print(
        f"dataset={args.dataset} ckpt={ckpt_path} device={device} "
        f"max_clips={args.max_clips} iou_thr={args.iou_thr}",
        flush=True,
    )
    packed = collect(
        model,
        loader,
        device,
        args.max_clips,
        args.iou_thr,
        spec["ignore_bg"],
        args.occ_frac,
        args.drop_ratio,
        args.stay_ratio,
        args.border,
        args.cc_thr,
        args.track_max_per_clip,
    )
    events = packed["events"]
    kinds = ("track", "cover", "split", "exit")
    slot_fields = ("d_pi", "d_l1", "d_l2", "pi", "l1", "l2", "c_id", "mass", "iou_obj", "iou_occluder")
    gt_fields = ("d_gt_pi", "d_gt_l1", "d_gt_l2", "gt_pi", "gt_l1", "gt_l2", "n_cc", "area_ratio")

    report = {
        "n_clips": packed["n_clips"],
        "n_events": len(events),
        "n_unmatched": packed["n_unmatched"],
        "pi_agree_mae": packed["pi_agree_mae"],
        "ckpt": ckpt_path,
        "iou_thr": args.iou_thr,
        "occ_frac": args.occ_frac,
        "drop_ratio": args.drop_ratio,
        "counts": {k: int(sum(e["kind"] == k for e in events)) for k in kinds},
        "matched_counts": {
            k: int(sum(e["kind"] == k and e.get("matched") for e in events)) for k in kinds
        },
    }

    def bucket(kind: str, field: str, matched_only: bool):
        xs = []
        for e in events:
            if e["kind"] != kind:
                continue
            if matched_only and not e.get("matched"):
                continue
            if field in e:
                xs.append(e[field])
        return np.asarray(xs, dtype=np.float64)

    report["slot"] = {
        k: {f: _summarize(bucket(k, f, True)) for f in slot_fields} for k in kinds
    }
    report["gt"] = {k: {f: _summarize(bucket(k, f, False)) for f in gt_fields} for k in kinds}

    def frac_drop_from_l2(kind: str, d_pi: str, d_l1: str, d_l2: str, matched_only: bool):
        """Share of π drops where λ2 rose (split-like) vs λ1 fell (mass-like)."""
        rows = [
            e
            for e in events
            if e["kind"] == kind and d_pi in e and (e.get("matched") or not matched_only)
        ]
        if not rows:
            return None
        n = len(rows)
        n_drop = sum(e[d_pi] < 0 for e in rows)
        n_l1 = sum(e[d_pi] < 0 and e[d_l1] < 0 for e in rows)
        n_l2 = sum(e[d_pi] < 0 and e[d_l2] > 0 for e in rows)
        return {
            "n": n,
            "frac_pi_drop": n_drop / n,
            "frac_drop_with_l1_fall": n_l1 / max(n_drop, 1),
            "frac_drop_with_l2_rise": n_l2 / max(n_drop, 1),
        }

    report["mechanism_slot"] = {
        k: frac_drop_from_l2(k, "d_pi", "d_l1", "d_l2", True) for k in kinds
    }
    report["mechanism_gt"] = {
        k: frac_drop_from_l2(k, "d_gt_pi", "d_gt_l1", "d_gt_l2", False) for k in kinds
    }

    report["traj_mean"] = {
        kind: {
            rel: {
                kk: _mean_se(list(packed["traj"][kind][rel][kk]))[0]
                for kk in ("pi", "l1", "l2", "gt_pi", "gt_l1", "gt_l2")
            }
            for rel in REL_WINDOW
        }
        for kind in kinds
    }

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    with (out_dir / "events.csv").open("w", newline="") as f:
        if events:
            wr = csv.DictWriter(f, fieldnames=list(events[0].keys()))
            wr.writeheader()
            wr.writerows(events)

    _plot_traj(
        out_dir / "traj_slot.png",
        packed["traj"],
        ("pi", "l1", "l2"),
        "Matched-slot n8 spectrum around onset (v39 eval)",
        (r"slot $\pi=\lambda_1-\lambda_2^+$", r"slot $\lambda_1$", r"slot $\lambda_2$"),
    )
    _plot_traj(
        out_dir / "traj_gt.png",
        packed["traj"],
        ("gt_pi", "gt_l1", "gt_l2"),
        "GT-support n8 spectrum around onset (oracle a)",
        (r"GT $\pi$", r"GT $\lambda_1$", r"GT $\lambda_2$"),
    )
    _plot_delta_scatter(out_dir / "delta_slot_l1_l2.png", events, "")
    _plot_delta_scatter(out_dir / "delta_gt_l1_l2.png", events, "gt_")
    _plot_hists(out_dir / "d_pi_slot_hist.png", events, "d_pi", r"matched slot $\Delta\pi$ at onset")
    _plot_hists(out_dir / "d_pi_gt_hist.png", events, "d_gt_pi", r"GT-support $\Delta\pi$ at onset")
    _plot_hists(
        out_dir / "c_id_hist.png",
        events,
        "c_id",
        r"$\cos(\hat u_t, u_t)$ at onset (matched slot)",
        xlim=(-0.2, 1.05),
    )

    def line(kind, group, field):
        d = report[group][kind][field]
        if d.get("n", 0) == 0:
            return f"  {kind:6s} {field:16s}  n=0"
        return (
            f"  {kind:6s} {field:16s}  n={d['n']:5d}  "
            f"mean={d['mean']:+.4f}  med={d['median']:+.4f}  "
            f"p10={d['p10']:+.4f}  p90={d['p90']:+.4f}"
        )

    print(f"\nclips={packed['n_clips']} events={len(events)} unmatched={packed['n_unmatched']} "
          f"π vs gate_conf MAE={packed['pi_agree_mae']}", flush=True)
    print(f"counts={report['counts']}  matched={report['matched_counts']}", flush=True)
    print("\n=== GT support (oracle a) ===", flush=True)
    for field in ("d_gt_pi", "d_gt_l1", "d_gt_l2", "n_cc", "area_ratio"):
        print(f"-- {field} --", flush=True)
        for kind in kinds:
            print(line(kind, "gt", field), flush=True)
    print("\n=== matched slot ===", flush=True)
    for field in ("d_pi", "d_l1", "d_l2", "c_id", "iou_obj", "iou_occluder"):
        print(f"-- {field} --", flush=True)
        for kind in kinds:
            print(line(kind, "slot", field), flush=True)
    print("\n=== π-drop mechanism (frac of drops) ===", flush=True)
    for tag, mech in (("slot", report["mechanism_slot"]), ("gt", report["mechanism_gt"])):
        print(f"  [{tag}]", flush=True)
        for kind in kinds:
            m = mech[kind]
            if not m:
                print(f"    {kind}: n=0", flush=True)
                continue
            print(
                f"    {kind}: n={m['n']}  P(Δπ<0)={m['frac_pi_drop']:.2f}  "
                f"of drops: λ1↓={m['frac_drop_with_l1_fall']:.2f}  "
                f"λ2↑={m['frac_drop_with_l2_rise']:.2f}",
                flush=True,
            )
    print(f"\nwrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
