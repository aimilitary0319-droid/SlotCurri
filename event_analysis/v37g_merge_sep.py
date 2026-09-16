"""Does v37g gap drop when a slot owns two GT objects?

v37 100k eval A, C_s on raw X. Labels match v39_merge_spectrum:
  exclusive  IoU1>=0.30 and IoU2<0.12
  merge      IoU1>=0.20 and IoU2>=0.15  (touch vs apart)
Oracle: GT exclusive mask vs A∪B as a, same C_s.

Usage:
  python event_analysis/v37g_merge_sep.py --dataset ytvis --max-clips 40
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from slotcurri import configuration, data, metrics, models
from slotcurri.modules.video import spectral_slot_purity

from v37g_gt_size_sep import (
    RUNS,
    _FakeTrainer,
    _iou_slot_gt,
    _one_hot_gt,
    auc_rank,
    summarize,
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


def report(name, score, exclusive, merge):
    print(f"\n== {name} ==")
    print(f"  exclusive  {summarize(score[exclusive])}")
    print(f"  merge      {summarize(score[merge])}")
    if exclusive.any() and merge.any():
        ratio = float(score[merge].mean() / max(score[exclusive].mean(), 1e-8))
        print(
            f"  merge/excl mean={ratio:.2f}  "
            f"AUC excl>merge={auc_rank(score[exclusive], score[merge]):.3f}"
        )


@torch.no_grad()
def collect(model, loader, max_clips: int, ignore_bg: bool, proj: int):
    slot = {k: [] for k in ("kind", "gap", "pi", "occ", "mass", "iou1", "iou2", "win")}
    oracle = {k: [] for k in ("kind", "gap", "pi")}
    n_clips = 0
    for batch in loader:
        if "batch_padding_mask" in batch:
            batch = model._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                continue
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
        outputs = model.forward(batch, train=False, cycle=False)
        att = outputs["processor"]["state_attn_mask"].float()[0]
        raw = outputs["encoder"]["backbone_features"].float()[0]
        t, s, n = att.shape
        grid = int(round(n ** 0.5))
        occ = (att * att).sum(-1) / att.sum(-1).clamp_min(1e-8)
        pi = spectral_slot_purity(att, raw, divide_by_mass=True, proj_dim=proj)
        gap = spectral_slot_purity(att, raw, divide_by_mass=False, proj_dim=proj)
        mass = att.sum(-1)
        winner = att.argmax(dim=1)
        pred = torch.zeros(t, s, grid, grid, dtype=torch.bool, device=att.device)
        for si in range(s):
            pred[:, si] = (winner == si).reshape(t, grid, grid)
        win = pred.flatten(-2).float().mean(-1)

        gt_oh = _one_hot_gt(batch["segmentations"])[0]
        gt_p = F.interpolate(gt_oh.float(), size=(grid, grid), mode="nearest").bool()
        c0 = 1 if ignore_bg else 0
        if gt_p.shape[1] <= c0:
            n_clips += 1
            if n_clips >= max_clips:
                break
            continue
        fg = gt_p[:, c0:]
        iou = _iou_slot_gt(pred, fg)
        hw = float(grid * grid)

        # oracle: exclusive GT vs unions of the two largest objects
        for ti in range(t):
            areas = [(ci, float(fg[ti, ci].float().sum())) for ci in range(fg.shape[1])]
            areas = [(ci, a) for ci, a in areas if a >= 4]
            areas.sort(key=lambda x: -x[1])
            if not areas:
                continue
            masks = []
            tags = []
            c_a = areas[0][0]
            masks.append(fg[ti, c_a].reshape(-1).float())
            tags.append("excl")
            for c_b, _ in areas[1:3]:
                uni = (fg[ti, c_a] | fg[ti, c_b]).reshape(-1).float()
                pa = fg[ti, c_a].cpu().numpy()
                pb = fg[ti, c_b].cpu().numpy()
                tags.append("union_touch" if _touch8(pa, pb) else "union_apart")
                masks.append(uni)
            a_or = torch.stack(masks, dim=0).unsqueeze(0)
            feat = raw[ti : ti + 1]
            og = spectral_slot_purity(a_or, feat, divide_by_mass=False, proj_dim=proj)[0]
            op = spectral_slot_purity(a_or, feat, divide_by_mass=True, proj_dim=proj)[0]
            for i, tag in enumerate(tags):
                oracle["kind"].append(tag)
                oracle["gap"].append(float(og[i]))
                oracle["pi"].append(float(op[i]))

        for ti in range(t):
            for si in range(s):
                scores = iou[ti, si]
                if scores.numel() == 0:
                    continue
                topv, topi = scores.topk(min(2, scores.numel()))
                v1 = float(topv[0])
                v2 = float(topv[1]) if topv.numel() > 1 else 0.0
                if v1 < 0.10:
                    kind = "ghost"
                elif v1 >= 0.30 and v2 < 0.12:
                    kind = "exclusive"
                elif v1 >= 0.20 and v2 >= 0.15:
                    i1, i2 = int(topi[0]), int(topi[1])
                    pa = fg[ti, i1].cpu().numpy()
                    pb = fg[ti, i2].cpu().numpy()
                    kind = "merge_touch" if _touch8(pa, pb) else "merge_apart"
                else:
                    continue
                slot["kind"].append(kind)
                slot["gap"].append(float(gap[ti, si]))
                slot["pi"].append(float(pi[ti, si]))
                slot["occ"].append(float(occ[ti, si]))
                slot["mass"].append(float(mass[ti, si]))
                slot["iou1"].append(v1)
                slot["iou2"].append(v2)
                slot["win"].append(float(win[ti, si]))

        n_clips += 1
        if n_clips % 10 == 0:
            print(f"  clips={n_clips}", flush=True)
        if n_clips >= max_clips:
            break
    out = {k: np.asarray(v) for k, v in slot.items()}
    out["n_clips"] = n_clips
    out["oracle"] = {k: np.asarray(v) for k, v in oracle.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(RUNS), required=True)
    ap.add_argument("--max-clips", type=int, default=40)
    ap.add_argument("--data-dir", default="/workspace/dataset")
    args = ap.parse_args()
    spec = RUNS[args.dataset]
    print(f"loading {spec['ckpt']}", flush=True)
    config = configuration.load_config(spec["config"])
    config.model.visualize = False
    config.dataset.val_batch_size = 1
    config.dataset.num_val_workers = 0
    built = {n: metrics.build(c) for n, c in config.val_metrics.items()}
    model = models.build(config.model, config.optimizer, None, built)
    sd = torch.load(spec["ckpt"], map_location="cpu")
    model.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd, strict=False)
    model = model.cuda().eval()
    model._trainer = _FakeTrainer(100000)
    proj = int(getattr(model, "amc_spectral_proj_dim", 64) or 64)
    dm = data.build(config.dataset, data_dir=args.data_dir)
    dm.setup("validate")
    pack = collect(model, dm.val_dataloader(), args.max_clips, spec["ignore_bg"], proj)
    kind = pack["kind"]
    print(f"\n{args.dataset} clips={pack['n_clips']}  C_s=raw X  proj={proj}")
    for k in ("exclusive", "merge_touch", "merge_apart", "ghost"):
        print(f"  {k:14s} n={(kind == k).sum()}")
    excl = kind == "exclusive"
    merge = (kind == "merge_touch") | (kind == "merge_apart")
    report("v37g gap  (λ1-λ2)", pack["gap"].astype(float), excl, merge)
    report("v37 π     (gap/mass)", pack["pi"].astype(float), excl, merge)
    report("occupancy", pack["occ"].astype(float), excl, merge)
    for tag in ("merge_touch", "merge_apart"):
        m = kind == tag
        if m.any():
            print(
                f"  {tag}: gap={pack['gap'][m].astype(float).mean():.1f}  "
                f"π={pack['pi'][m].astype(float).mean():.3f}  "
                f"occ={pack['occ'][m].astype(float).mean():.3f}  n={m.sum()}"
            )

    okind = pack["oracle"]["kind"]
    ogap = pack["oracle"]["gap"].astype(float)
    opi = pack["oracle"]["pi"].astype(float)
    print("\n--- oracle GT masks as a ---")
    for tag in ("excl", "union_touch", "union_apart"):
        m = okind == tag
        if m.any():
            print(
                f"  {tag:12s} n={m.sum():4d}  gap={ogap[m].mean():.1f}  "
                f"π={opi[m].mean():.3f}"
            )
    oe, ou = okind == "excl", (okind == "union_touch") | (okind == "union_apart")
    if oe.any() and ou.any():
        print(
            f"  union/excl gap={ogap[ou].mean()/max(ogap[oe].mean(), 1e-8):.2f}  "
            f"π={opi[ou].mean()/max(opi[oe].mean(), 1e-8):.2f}  "
            f"AUC excl>union gap={auc_rank(ogap[oe], ogap[ou]):.3f}  "
            f"π={auc_rank(opi[oe], opi[ou]):.3f}"
        )


if __name__ == "__main__":
    main()
