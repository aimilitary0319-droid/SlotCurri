"""Where does one-step global P actually send mass on MOVi-C?

Same-region does not imply one Key. P only mixes along ReLU-cosine W inside
the Ncut region. This measures own-object vs other-object vs bg mass.
"""

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
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor


@torch.no_grad()
def graph(x, ncut: NcutRelationalLeveling):
    x = x.float().unsqueeze(0)
    z = F.normalize(x, dim=-1)
    w = torch.bmm(z, z.transpose(1, 2)).clamp_min(0.0)
    w.diagonal(dim1=-2, dim2=-1).zero_()
    exp = ncut.explain(x)
    region = exp["region"][0]
    same = region.unsqueeze(-1) == region.unsqueeze(-2)
    w_t = w[0].masked_fill(~same, 0.0)
    row = w_t.sum(dim=-1, keepdim=True)
    p = w_t / row.clamp_min(ncut.eps)
    return w[0], p, region, exp["rel"][0]


def mean_pair(w, a, b):
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    block = w.index_select(0, a).index_select(1, b)
    if a is b or torch.equal(a, b):
        n = a.numel()
        if n < 2:
            return float("nan")
        return float((block.sum() / (n * (n - 1))).item())
    return float(block.mean().item())


@torch.no_grad()
def frame_stats(w, p, region, raw, rel, gt, bg_id=0):
    obj_ids = [int(i) for i in gt.unique().tolist() if int(i) != bg_id]
    if len(obj_ids) < 2:
        return None
    bg = (gt == bg_id).nonzero(as_tuple=False).squeeze(-1)
    recs = []
    obj_means_raw, obj_means_rel = [], []
    for oid in obj_ids:
        own = (gt == oid).nonzero(as_tuple=False).squeeze(-1)
        if own.numel() < 4:
            continue
        others = []
        for j in obj_ids:
            if j == oid:
                continue
            idx = (gt == j).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel():
                others.append(idx)
        other = torch.cat(others) if others else own.new_empty(0)
        # restrict "other" / bg to same Ncut region as the object's majority
        maj = bool(region[own].float().mean() >= 0.5)
        same_reg = region == maj
        other_s = other[same_reg[other]] if other.numel() else other
        bg_s = bg[same_reg[bg]] if bg.numel() else bg
        p_own = p.index_select(0, own)
        mass_own = float(p_own.index_select(1, own).sum() / own.numel())
        mass_oth = float(p_own.index_select(1, other_s).sum() / own.numel()) if other_s.numel() else 0.0
        mass_bg = float(p_own.index_select(1, bg_s).sum() / own.numel()) if bg_s.numel() else 0.0
        recs.append(
            {
                "w_own": mean_pair(w, own, own),
                "w_oth": mean_pair(w, own, other_s) if other_s.numel() else float("nan"),
                "w_bg": mean_pair(w, own, bg_s) if bg_s.numel() else float("nan"),
                "p_own": mass_own,
                "p_oth": mass_oth,
                "p_bg": mass_bg,
            }
        )
        obj_means_raw.append(raw[own].mean(0))
        obj_means_rel.append(rel[own].mean(0))

    def pairwise_cos(means):
        if len(means) < 2:
            return float("nan")
        m = F.normalize(torch.stack(means), dim=-1)
        sim = m @ m.T
        n = sim.shape[0]
        return float(((sim.sum() - n) / (n * (n - 1))).item())

    avg = {k: float(torch.tensor([r[k] for r in recs if r[k] == r[k]]).mean()) for k in recs[0]}
    avg["n_obj"] = len(obj_ids)
    avg["cos_obj_raw"] = pairwise_cos(obj_means_raw)
    avg["cos_obj_rel"] = pairwise_cos(obj_means_rel)
    return avg


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
    ncut = NcutRelationalLeveling(chunk_size=8, n_iter=16).to(device)
    acc = []
    n = 0
    for batch in dm.val_dataloader():
        if n >= args.max_clips:
            break
        if "batch_padding_mask" in batch and torch.is_tensor(batch["batch_padding_mask"]) and bool(
            batch["batch_padding_mask"].any()
        ):
            n += 1
            continue
        video = batch["video"][0]
        seg = batch["segmentations"][0]
        ti = int(video.shape[0] // 2)
        frame = video[ti].to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            raw = extract_tokens(backbone, frame.unsqueeze(0), bb.features).float()[0]
        grid = int(round(math.sqrt(raw.shape[0])))
        gt = patch_ids_movi(seg[ti].to(device), grid)
        w, p, region, rel = graph(raw, ncut)
        st = frame_stats(w, p, region, raw, rel, gt)
        if st is not None:
            acc.append(st)
            print(
                f"clip {n}: n_obj={st['n_obj']}  P own={st['p_own']:.2f} oth={st['p_oth']:.2f} bg={st['p_bg']:.2f}  "
                f"cos obj {st['cos_obj_raw']:.2f}->{st['cos_obj_rel']:.2f}",
                flush=True,
            )
        n += 1
    keys = ["p_own", "p_oth", "p_bg", "w_own", "w_oth", "w_bg", "cos_obj_raw", "cos_obj_rel"]
    mean = {k: float(sum(a[k] for a in acc) / len(acc)) for k in keys}
    print(json.dumps({"n": len(acc), **mean}, indent=2))


if __name__ == "__main__":
    main()
