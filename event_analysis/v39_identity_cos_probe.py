#!/usr/bin/env python3
"""Does cos(û_t, u_t) separate normal tracking from identity switch / occlusion?

Eval-only on a trained v39 checkpoint (identity gate off — this is a diagnostic,
not v39i). Per occupied slot, t>=1:

  c_slot  = cos(û_t, u_t)                 # proposed identity signal (slot residual)
  c_patch = cos(ā_{t-1} Z_{t-1}, ā_t Z_t) # attention-weighted DINO tokens

Labels from decoder-hard vs GT (fg IoU):
  track   same FG object, IoU >= thr both frames
  switch  different FG objects both frames
  occlude occupied -> not occupied

Usage (slotcurri image):
  python event_analysis/v39_identity_cos_probe.py --dataset movi --max-clips 80
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from slotcurri import configuration, data, models


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


def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _one_hot_gt(seg: torch.Tensor) -> torch.Tensor:
    """(B,T,H,W) ids -> (B,T,C,H,W) bool."""
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
    """pred (T,S,H,W) bool, gt (T,C,H,W) bool -> iou (T,S,C)."""
    t, s, h, w = pred.shape
    c = gt.shape[1]
    p = pred.reshape(t, s, h * w).float()
    g = gt.reshape(t, c, h * w).float()
    inter = torch.einsum("tsn,tcn->tsc", p, g)
    psum = p.sum(-1).unsqueeze(-1)
    gsum = g.sum(-1).unsqueeze(1)
    return inter / (psum + gsum - inter).clamp_min(1.0)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg). For switch-as-positive use -c so lower cosine ranks as switch."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort().astype(np.float64) + 1.0
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


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


def _prototypes(att: torch.Tensor, bind: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """e_{t,s} = Σ_i α̃_{t,s,i} z_{t,i}, L2-normalized. att (T,S,N), bind (T,N,C)."""
    mass = att.sum(dim=-1, keepdim=True).clamp_min(eps)
    e = torch.einsum("tsn,tnc->tsc", att / mass, bind)
    return F.normalize(e, dim=-1)


def _identity_margin(e: torch.Tensor) -> tuple:
    """q+, q-, Δ for t>=1. e: (T,S,C) unit vectors.

    q+_s = cos(e_t,s, e_{t-1,s})
    q-_s = max_{r≠s} cos(e_t,s, e_{t-1,r})
    Δ_s  = q+ - q-
    """
    sim = torch.matmul(e[1:], e[:-1].transpose(-1, -2))  # (T-1,S,S)
    q_plus = sim.diagonal(dim1=-2, dim2=-1)
    s = sim.shape[-1]
    off = sim.masked_fill(torch.eye(s, dtype=torch.bool, device=sim.device), -1.0)
    q_minus = off.max(dim=-1).values
    return q_plus, q_minus, q_plus - q_minus


@torch.no_grad()
def collect(model, loader, device, max_clips: int, iou_thr: float, ignore_bg: bool):
    keys = ("slot", "patch", "qplus", "qminus", "delta", "delta_win")
    buckets = {k: {kk: [] for kk in keys} for k in ("track", "switch", "occlude")}
    n_clips = 0
    for batch in loader:
        if "batch_padding_mask" in batch:
            batch = model._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                continue
        batch = _to_device(batch, device)
        outputs = model.forward(batch, train=False, cycle=False)
        aux = model.aux_forward(batch, outputs)

        u = outputs["processor"]["state"].float()  # (B,T,S,D)
        hat_next = outputs["processor"]["state_predicted"].float()
        att = outputs["processor"]["state_attn_mask"].float()  # (B,T,S,N)
        bind = outputs["encoder"].get("backbone_key")
        if bind is None:
            bind = outputs["encoder"]["backbone_features"]
        bind = F.normalize(bind.float(), dim=-1)  # (B,T,N,C)

        key = "decoder_masks_vis_hard" if "decoder_masks_vis_hard" in aux else "decoder_masks_hard"
        pred = aux[key].bool()[0]  # (T,S,H,W)
        gt = _one_hot_gt(batch["segmentations"])[0]
        gt = _resize_bool(gt.unsqueeze(0), pred.shape[-2:])[0]

        iou = _iou_slot_gt(pred.cpu(), gt.cpu())  # (T,S,C)
        if ignore_bg:
            iou[:, :, 0] = 0.0
        best_iou, best_id = iou.max(dim=-1)  # (T,S)
        occupied = best_iou >= iou_thr

        bsz, t, s, _ = u.shape
        assert bsz == 1
        u_t = u[0, 1:]
        hat_t = hat_next[0, :-1]
        c_slot = F.cosine_similarity(hat_t, u_t, dim=-1).cpu()  # (T-1,S)

        e = _prototypes(att[0], bind[0])
        q_plus, q_minus, delta = _identity_margin(e)
        q_plus, q_minus, delta = q_plus.cpu(), q_minus.cpu(), delta.cpu()
        c_patch = q_plus  # same as cos(e_t, e_{t-1})

        winner = att[0].argmax(dim=1)  # (T,N)
        hard = F.one_hot(winner, num_classes=s).permute(0, 2, 1).float()
        e_win = _prototypes(hard, bind[0])
        _qp_w, _qm_w, delta_win = _identity_margin(e_win)
        delta_win = delta_win.cpu()

        occ_prev = occupied[:-1]
        occ_cur = occupied[1:]
        id_prev = best_id[:-1]
        id_cur = best_id[1:]
        same = id_prev == id_cur

        track = occ_prev & occ_cur & same
        switch = occ_prev & occ_cur & ~same
        occlude = occ_prev & ~occ_cur

        packed = {
            "slot": c_slot,
            "patch": c_patch,
            "qplus": q_plus,
            "qminus": q_minus,
            "delta": delta,
            "delta_win": delta_win,
        }
        for name, mask in (("track", track), ("switch", switch), ("occlude", occlude)):
            m = mask.cpu()
            if m.any():
                for kk, tensor in packed.items():
                    buckets[name][kk].append(tensor[m].numpy())

        n_clips += 1
        if n_clips % 10 == 0:
            print(f"  clips {n_clips}/{max_clips}", flush=True)
        if n_clips >= max_clips:
            break

    out = {}
    for name, store in buckets.items():
        out[name] = {
            kk: np.concatenate(store[kk]) if store[kk] else np.array([])
            for kk in keys
        }
    return out, n_clips


def _hist(path: Path, groups: Dict[str, np.ndarray], title: str, xlim=(-0.2, 1.05), xlabel="cosine", vlines=()):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = {"track": "#2ca02c", "switch": "#d62728", "occlude": "#ff7f0e"}
    for name, xs in groups.items():
        if xs.size == 0:
            continue
        ax.hist(
            xs,
            bins=40,
            range=xlim,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=colors.get(name, "gray"),
            label=f"{name} n={xs.size}  μ={xs.mean():.3f}",
        )
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(frameon=False)
    for x, ls in vlines:
        ax.axvline(x, color="0.7", ls=ls, lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("movi", "ytvis"), default="movi")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--max-clips", type=int, default=80)
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--out", default="logs/v39_identity_cos_probe")
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    spec = RUNS[args.dataset]
    config_path = args.config or spec["config"]
    ckpt_path = args.ckpt or spec["ckpt"]

    config = configuration.load_config(config_path)
    config.model.visualize = False
    config.dataset.val_batch_size = 1
    config.dataset.num_val_workers = 0

    model = models.build(config.model, config.optimizer)
    model.load_weights_from_checkpoint(ckpt_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    model._trainer = _FakeTrainer(args.step)
    # diagnostic: do not apply the v39i mix; we only read û, u
    model.amc_state_identity_cos = False

    dataset = data.build(config.dataset, data_dir=args.data_dir)
    dataset.setup("validate")
    loader = dataset.val_dataloader()

    print(
        f"dataset={args.dataset} ckpt={ckpt_path} device={device} "
        f"max_clips={args.max_clips} iou_thr={args.iou_thr}",
        flush=True,
    )
    buckets, n_clips = collect(
        model, loader, device, args.max_clips, args.iou_thr, spec["ignore_bg"]
    )

    report = {"n_clips": n_clips, "iou_thr": args.iou_thr, "ckpt": ckpt_path}
    metrics = ("slot", "patch", "qplus", "qminus", "delta", "delta_win")
    for kind in ("track", "switch", "occlude"):
        report[kind] = {m: _summarize(buckets[kind][m]) for m in metrics}

    def gap(kind_hi, kind_lo, metric):
        a, b = buckets[kind_hi][metric], buckets[kind_lo][metric]
        if a.size == 0 or b.size == 0:
            return None
        return float(a.mean() - b.mean())

    def bad(metric):
        parts = [buckets[k][metric] for k in ("switch", "occlude") if buckets[k][metric].size]
        return np.concatenate(parts) if parts else np.array([])

    report["auc_track_gt_switchocc"] = {
        m: _auc(buckets["track"][m], bad(m)) for m in metrics
    }
    report["gap_mean_track_minus_switch"] = {m: gap("track", "switch", m) for m in metrics}
    report["gap_mean_track_minus_occlude"] = {m: gap("track", "occlude", m) for m in metrics}

    def sign_stats(metric):
        out = {}
        for kind in ("track", "switch", "occlude"):
            xs = buckets[kind][metric]
            if xs.size == 0:
                out[kind] = {"frac_pos": None, "frac_neg": None}
            else:
                out[kind] = {
                    "frac_pos": float((xs > 0).mean()),
                    "frac_neg": float((xs < 0).mean()),
                }
        return out

    report["sign"] = {
        "delta": sign_stats("delta"),
        "delta_win": sign_stats("delta_win"),
    }

    def sigmoid_gate(xs, tau):
        return 1.0 / (1.0 + np.exp(-xs / max(tau, 1e-8)))

    report["sigmoid_mean_g"] = {}
    for tau in (0.02, 0.05, 0.10):
        report["sigmoid_mean_g"][str(tau)] = {
            kind: float(sigmoid_gate(buckets[kind]["delta"], tau).mean())
            if buckets[kind]["delta"].size
            else None
            for kind in ("track", "switch", "occlude")
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    groups3 = {k: buckets[k] for k in ("track", "switch", "occlude")}
    _hist(
        out_dir / "c_slot_hist.png",
        {k: v["slot"] for k, v in groups3.items()},
        r"$c_{slot}=\cos(\hat u_t, u_t)$  (v39, eval)",
        vlines=((0.9, "--"), (0.4, ":")),
    )
    _hist(
        out_dir / "c_patch_hist.png",
        {k: v["patch"] for k, v in groups3.items()},
        r"$q^+=\cos(e_t, e_{t-1})$  visual prototype",
        vlines=((0.9, "--"), (0.4, ":")),
    )
    _hist(
        out_dir / "delta_hist.png",
        {k: v["delta"] for k, v in groups3.items()},
        r"$\Delta=q^+-q^-$  identity margin (soft $e$)",
        xlim=(-0.4, 0.4),
        xlabel=r"$\Delta$",
        vlines=((0.0, "-"),),
    )
    _hist(
        out_dir / "delta_win_hist.png",
        {k: v["delta_win"] for k, v in groups3.items()},
        r"$\Delta$  winner-only prototypes (argmax $\alpha$)",
        xlim=(-0.4, 0.4),
        xlabel=r"$\Delta$",
        vlines=((0.0, "-"),),
    )

    def line(kind, which):
        d = report[kind][which]
        if d.get("n", 0) == 0:
            return f"  {kind:8s} {which:9s}  n=0"
        return (
            f"  {kind:8s} {which:9s}  n={d['n']:6d}  "
            f"mean={d['mean']:.3f}  med={d['median']:.3f}  "
            f"p10={d['p10']:.3f}  p90={d['p90']:.3f}"
        )

    print(f"\nclips={n_clips}", flush=True)
    for which in metrics:
        print(f"\n=== {which} ===", flush=True)
        for kind in ("track", "switch", "occlude"):
            print(line(kind, which), flush=True)
        print(
            f"  AUC P(track>switch∪occ) = {report['auc_track_gt_switchocc'][which]:.3f}",
            flush=True,
        )
        print(
            f"  mean gap track-switch={report['gap_mean_track_minus_switch'][which]}  "
            f"track-occlude={report['gap_mean_track_minus_occlude'][which]}",
            flush=True,
        )

    print("\n=== sign(Δ) as a hard gate ===", flush=True)
    for metric in ("delta", "delta_win"):
        print(f"  [{metric}]", flush=True)
        for kind in ("track", "switch", "occlude"):
            st = report["sign"][metric][kind]
            print(
                f"    {kind:8s}  P(Δ>0)={st['frac_pos']}  P(Δ<0)={st['frac_neg']}",
                flush=True,
            )
    print("\n=== mean σ(Δ/τ) (soft e) ===", flush=True)
    for tau, row in report["sigmoid_mean_g"].items():
        print(f"  τ={tau}  track={row['track']}  switch={row['switch']}  occlude={row['occlude']}", flush=True)

    dgap = report["gap_mean_track_minus_switch"]["delta"]
    if dgap is None:
        verdict = "inconclusive"
    elif dgap >= 0.08 and report["auc_track_gt_switchocc"]["delta"] >= 0.80:
        verdict = "margin is a useful identity signal"
    elif dgap <= 0.03 or report["auc_track_gt_switchocc"]["delta"] < 0.65:
        verdict = "margin still too weak to gate"
    else:
        verdict = "margin helps vs raw cosine but gap is modest"
    report["verdict_margin"] = verdict
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(f"\nverdict (identity margin): {verdict}", flush=True)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
