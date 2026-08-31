"""Compare ReLU(S) vs ReLU(S - μ_i) global P mass on MOVi-C DINO tokens."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featcur_strength_probe import extract_tokens, patch_ids_movi  # noqa: E402
from slotcurri import configuration, data
from slotcurri.modules.encoders import TimmExtractor


def cosine_s(x):
    z = F.normalize(x.float(), dim=-1)
    s = z @ z.T
    s.fill_diagonal_(0.0)
    return s


def p_from_w(w, eps=1e-6):
    row = w.sum(dim=-1, keepdim=True)
    p = w / row.clamp_min(eps)
    rel = p @ x_hold["x"] if False else None  # placeholder
    return p, row


def masses(p, gt, obj_ids, bg_id=0):
    bg = (gt == bg_id).nonzero(as_tuple=False).squeeze(-1)
    recs = []
    for oid in obj_ids:
        own = (gt == oid).nonzero(as_tuple=False).squeeze(-1)
        if own.numel() < 4:
            continue
        others = [((gt == j).nonzero(as_tuple=False).squeeze(-1)) for j in obj_ids if j != oid]
        other = torch.cat(others) if others else own.new_empty(0)
        po = p.index_select(0, own)
        recs.append(
            {
                "p_own": float(po.index_select(1, own).sum() / own.numel()),
                "p_oth": float(po.index_select(1, other).sum() / own.numel()) if other.numel() else 0.0,
                "p_bg": float(po.index_select(1, bg).sum() / own.numel()) if bg.numel() else 0.0,
            }
        )
    if not recs:
        return None
    return {k: float(sum(r[k] for r in recs) / len(recs)) for k in recs[0]}


def obj_cos(x, gt, obj_ids):
    means = []
    for oid in obj_ids:
        own = (gt == oid).nonzero(as_tuple=False).squeeze(-1)
        if own.numel() >= 4:
            means.append(x[own].mean(0))
    if len(means) < 2:
        return float("nan")
    m = F.normalize(torch.stack(means), dim=-1)
    sim = m @ m.T
    n = sim.shape[0]
    return float(((sim.sum() - n) / (n * (n - 1))).item())


def unique_keys(tokens, thr=0.95):
    xn = F.normalize(tokens.float(), dim=-1)
    kept = []
    for i in range(xn.shape[0]):
        if not kept or float((torch.stack(kept) @ xn[i]).max()) < thr:
            kept.append(xn[i])
    return len(kept)


@torch.no_grad()
def level(x, kind):
    s = cosine_s(x)
    if kind == "relu0":
        w = s.clamp_min(0.0)
    elif kind == "adaptive":
        mu = s.sum(dim=-1, keepdim=True) / max(s.shape[0] - 1, 1)
        w = (s - mu).clamp_min(0.0)
    elif kind == "degcorr":
        a = s.clamp_min(0.0)
        a.fill_diagonal_(0.0)
        deg = a.sum(dim=-1)
        expected = deg.unsqueeze(-1) * deg.unsqueeze(-2) / deg.sum().clamp_min(1e-6)
        w = (a - expected).clamp_min(0.0)
    elif kind == "symnorm":
        a = s.clamp_min(0.0)
        a.fill_diagonal_(0.0)
        d_inv_sqrt = a.sum(dim=-1).clamp_min(1e-6).rsqrt()
        w = d_inv_sqrt.unsqueeze(-1) * a * d_inv_sqrt.unsqueeze(-2)
    else:
        raise ValueError(kind)
    w.fill_diagonal_(0.0)
    row = w.sum(dim=-1, keepdim=True)
    p = w / row.clamp_min(1e-6)
    # Proposal is X_low = S X (S already degree-normalized). Others use row-stochastic P X.
    rel = (w @ x.float()) if kind == "symnorm" else (p @ x.float())
    rel = torch.where(row < 1e-6, x.float(), rel)
    nnz = float((w > 0).float().mean())
    return p, rel, nnz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--max-clips", type=int, default=20)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = configuration.load_config("configs/slotcurri/movi_c_attnmass_v36.yaml")
    cfg.dataset.val_batch_size = 1
    cfg.dataset.num_val_workers = 0
    dm = data.build(cfg.dataset, data_dir=args.data_dir)
    dm.setup("validate")
    bb = cfg.model.encoder.backbone
    kwargs = dict(bb.get("model_kwargs") or {})
    backbone = (
        TimmExtractor(
            model=bb.model, pretrained=True, frozen=True, features=bb.features, model_kwargs=kwargs or None
        )
        .to(device)
        .eval()
    )
    kinds = ("relu0", "symnorm")
    acc = {k: [] for k in kinds}
    n = 0
    for batch in dm.val_dataloader():
        if n >= args.max_clips:
            break
        if "batch_padding_mask" in batch and torch.is_tensor(batch["batch_padding_mask"]) and bool(
            batch["batch_padding_mask"].any()
        ):
            n += 1
            continue
        video, seg = batch["video"][0], batch["segmentations"][0]
        ti = int(video.shape[0] // 2)
        frame = video[ti].to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            raw = extract_tokens(backbone, frame.unsqueeze(0), bb.features).float()[0]
        grid = int(round(math.sqrt(raw.shape[0])))
        gt = patch_ids_movi(seg[ti].to(device), grid)
        obj_ids = [int(i) for i in gt.unique().tolist() if int(i) != 0]
        if len(obj_ids) < 2:
            n += 1
            continue
        raw_cos = obj_cos(raw, gt, obj_ids)
        line = f"clip {n} n_obj={len(obj_ids)} raw_cos={raw_cos:.2f}"
        for kind in kinds:
            p, rel, nnz = level(raw, kind)
            m = masses(p, gt, obj_ids)
            if m is None:
                continue
            rec = {
                **m,
                "cos_rel": obj_cos(rel, gt, obj_ids),
                "keys": unique_keys(rel),
                "nnz": nnz,
            }
            acc[kind].append(rec)
            line += (
                f"  {kind}: P {m['p_own']:.2f}/{m['p_oth']:.2f}/{m['p_bg']:.2f} "
                f"cos={rec['cos_rel']:.2f} keys={rec['keys']} nnz={nnz:.2f}"
            )
        print(line, flush=True)
        n += 1

    out = {}
    for kind, recs in acc.items():
        if not recs:
            continue
        keys = recs[0].keys()
        out[kind] = {k: float(sum(r[k] for r in recs) / len(recs)) for k in keys}
        out[kind]["n"] = len(recs)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
