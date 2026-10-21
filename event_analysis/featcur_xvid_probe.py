"""Cross-video-aware feature curriculum candidates vs the v39lam1gu dense P X.

Within-frame statistics of X^rel are the same on YouTube-VIS, MOVi-C and MOVi-E
(featcur_xrel). They differ only in how much of a frame is shared with other
videos. Every candidate below uses the minibatch (one frame per video) as the
reference, as training would:

  dense      W = ReLU(z_i^T z_j)                                  (current FC)
  xcenter    W = ReLU((z_i - zbar)^T (z_j - zbar)),  zbar = batch mean of z
  xcenter_n  same, centered z re-normalized before the dot product
  xshared    mu + g (P_dense X - mu),  mu = batch mean of X,
             g^2 = E||X - mu||^2 / E||P_dense X - mu||^2
  xframe     mu + g (m_f - mu) + (P_dense X - m_f),  m_f = frame mean of P_dense X,
             g^2 = (E||X - mu||^2 - E||PX - m_f||^2) / E||m_f - mu||^2
  rowcenter  existing anneal=rowcenter (within-frame tau)

A: LN + L2 key-token cosine, within frame and across videos (global_winner A).
B: real config modules at random init, frame-0 Slot Attention (global_winner B).

Usage (inside the slotcurri image):
  python event_analysis/featcur_xvid_probe.py --dataset movi_e
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from global_winner_probe import CONFIGS, build_arch, load_frames  # noqa: E402
from slotcurri import configuration  # noqa: E402
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor  # noqa: E402

VARIANTS = ("raw", "dense_s0.5", "dense", "xcenter", "xcenter_n", "xshared", "xframe", "rowcenter")


def xframe_rescale(x: torch.Tensor, rel: torch.Tensor):
    """mu + g (m_f - mu) + (rel - m_f). Within-frame residual sums to 0 per frame,
    so it is orthogonal to (m_f - mu) and the energy match is closed form."""
    mu = x.mean(dim=(0, 1), keepdim=True)
    m = rel.mean(dim=1, keepdim=True)
    e = rel - m
    num = (x - mu).square().sum(-1).mean() - e.square().sum(-1).mean()
    den = (m - mu).square().sum(-1).mean()
    g = (num.clamp_min(0.0) / den.clamp_min(1e-12)).sqrt()
    return mu + g * (m - mu) + e, float(g)


def _row_level(w: torch.Tensor, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = w.clone()
    w.diagonal(dim1=-2, dim2=-1).zero_()
    row = w.sum(-1, keepdim=True)
    rel = torch.bmm(w, x) / row.clamp_min(eps)
    return torch.where(row < eps, x, rel)


@torch.no_grad()
def candidates(x: torch.Tensor, rowcenter: NcutRelationalLeveling) -> dict:
    """x: (V, N, D) one frame per video. Returns key tokens per variant."""
    z = F.normalize(x, dim=-1)
    dense = _row_level(torch.bmm(z, z.transpose(1, 2)).clamp_min(0.0), x)

    zbar = z.mean(dim=(0, 1), keepdim=True)
    zc = z - zbar
    xcenter = _row_level(torch.bmm(zc, zc.transpose(1, 2)).clamp_min(0.0), x)
    zcn = F.normalize(zc, dim=-1)
    xcenter_n = _row_level(torch.bmm(zcn, zcn.transpose(1, 2)).clamp_min(0.0), x)

    mu = x.mean(dim=(0, 1), keepdim=True)
    g = ((x - mu).square().sum(-1).mean() / (dense - mu).square().sum(-1).mean()).sqrt()
    xshared = mu + g * (dense - mu)
    xframe, g_frame = xframe_rescale(x, dense)

    return {
        "xframe": xframe,
        "_g_frame": g_frame,
        "raw": x,
        "dense_s0.5": 0.5 * dense + 0.5 * x,
        "dense": dense,
        "xcenter": xcenter,
        "xcenter_n": xcenter_n,
        "xshared": xshared,
        "rowcenter": rowcenter(x, 0.0),
        "_g": float(g),
        "_pos_dense": float((torch.bmm(z, z.transpose(1, 2)) > 0).float().mean()),
        "_pos_xcenter": float((torch.bmm(zc, zc.transpose(1, 2)) > 0).float().mean()),
    }


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(CONFIGS))
    p.add_argument("--data-dir", default="/workspace/dataset")
    p.add_argument("--out-dir", default="event_analysis/featcur_xvid")
    p.add_argument("--n-videos", type=int, default=256)
    p.add_argument("--per-shard", type=int, default=6)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--batch-videos", type=int, default=32)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    cfg = configuration.load_config(CONFIGS[args.dataset])
    bb = cfg.model.encoder.backbone
    feat_dim = int(cfg.globals.FEAT_DIM)
    backbone = TimmExtractor(
        model=bb.model, pretrained=True, frozen=True, features=bb.features,
        model_kwargs=dict(bb.get("model_kwargs") or {}) or None,
    ).to(device).eval()
    rowcenter = NcutRelationalLeveling(chunk_size=4, barrier=False, row_center=True).to(device)

    archs = {}
    for name, path in CONFIGS.items():
        acfg = configuration.load_config(path)
        if int(acfg.globals.FEAT_DIM) != feat_dim:
            continue
        tag = f"{name}_arch(K={acfg.globals.NUM_SLOTS},d={acfg.globals.SLOT_DIM})"
        archs[tag] = [build_arch(acfg, seed, device) for seed in range(args.seeds)]

    t0 = time.time()
    frames = load_frames(cfg, args.data_dir, args.n_videos, args.per_shard, rng)
    print(f"{args.dataset}: {len(frames)} frames ({time.time() - t0:.0f}s)", flush=True)

    feat = {v: {"S": None, "Q": 0.0, "cin": [], "V": 0} for v in VARIANTS}
    acc = {(t, v, s): {"dom": [], "share": [], "lv": [], "pv": []}
           for t in archs for v in VARIANTS for s in range(args.seeds)}
    extras = {"g": [], "g_frame": [], "pos_dense": [], "pos_xcenter": []}

    for b0 in range(0, len(frames), args.batch_videos):
        x_img = torch.stack(frames[b0 : b0 + args.batch_videos], 0).to(device)
        if x_img.shape[0] < 4:
            continue
        tok = []
        for c0 in range(0, x_img.shape[0], 8):
            t = backbone(x_img[c0 : c0 + 8])
            tok.append((t[bb.features] if isinstance(t, dict) else t).float())
        tok = torch.cat(tok, 0)
        n_tok = tok.shape[1]
        cand = candidates(tok, rowcenter)
        extras["g"].append(cand["_g"])
        extras["g_frame"].append(cand["_g_frame"])
        extras["pos_dense"].append(cand["_pos_dense"])
        extras["pos_xcenter"].append(cand["_pos_xcenter"])
        for v in VARIANTS:
            key_tok = cand[v]
            u = F.normalize(F.layer_norm(key_tok, (key_tok.shape[-1],)), dim=-1)
            m = u.mean(1)
            st = feat[v]
            st["S"] = m.sum(0) if st["S"] is None else st["S"] + m.sum(0)
            st["Q"] += float(m.square().sum())
            st["V"] += m.shape[0]
            st["cin"] += ((n_tok * m.square().sum(-1) - 1.0) / (n_tok - 1)).tolist()
            for tag, seeds in archs.items():
                for si, (ot, sa, init) in enumerate(seeds):
                    for c0 in range(0, tok.shape[0], 8):
                        tv = tok[c0 : c0 + 8]
                        fv = ot(tv)
                        fk = fv if v == "raw" else ot(key_tok[c0 : c0 + 8])
                        slots0 = init(tv.shape[0])
                        q = sa.to_q(sa.norm_slots(slots0))
                        k = sa.to_k(sa.norm_features(fk))
                        logit = torch.einsum("bsd,bnd->bsn", q, k) * sa.scale / sa.logit_div
                        lv = logit.mean(-1)
                        masks = sa(slots0, fv, n_iters=3, key_features=fk)["masks"]
                        dom = masks.mean(-1).argmax(-1)
                        share = (masks.argmax(1) == dom.unsqueeze(-1)).float().mean(-1)
                        a = acc[(tag, v, si)]
                        a["dom"] += dom.tolist()
                        a["share"] += share.tolist()
                        a["lv"].append(lv.cpu())
                        a["pv"] += (logit - lv.unsqueeze(-1)).var(1).mean(-1).tolist()
        print(f"  batch {b0 // args.batch_videos}: g={cand['_g']:.2f} "
              f"pos dense/xcenter={cand['_pos_dense']:.3f}/{cand['_pos_xcenter']:.3f}", flush=True)

    out = {"dataset": args.dataset, "n_videos": len(frames), "batch_videos": args.batch_videos,
           "features": {}, "models": {}, "extras": {k: float(np.mean(v)) for k, v in extras.items()}}
    print(f"\n[A] {args.dataset}: key-token cosine (LN + L2), batch of {args.batch_videos} videos")
    for v in VARIANTS:
        st = feat[v]
        V = st["V"]
        cross = (float(st["S"].square().sum()) - st["Q"]) / (V * (V - 1))
        cin = float(np.mean(st["cin"]))
        out["features"][v] = {"within_frame": cin, "cross_video": cross}
        print(f"  {v:11s} within-frame {cin:.3f}  cross-video {cross:.3f}  ratio {cross / cin:.3f}")

    print(f"\n[B] {args.dataset}: frame-0 Slot Attention at random init ({args.seeds} seeds)")
    for tag in archs:
        for v in VARIANTS:
            agree, share, gfrac = [], [], []
            for si in range(args.seeds):
                a = acc[(tag, v, si)]
                dom = np.asarray(a["dom"])
                agree.append(np.bincount(dom).max() / dom.size)
                share.append(float(np.mean(a["share"])))
                lv = torch.cat(a["lv"], 0)
                b = lv.mean(0)
                gv, vv, pv = float(b.var()), float((lv - b).var(1).mean()), float(np.mean(a["pv"]))
                gfrac.append(gv / (gv + vv + pv))
            row = {"same_slot_across_videos": float(np.mean(agree)),
                   "dominant_patch_share": float(np.mean(share)),
                   "global_frac": float(np.mean(gfrac))}
            out["models"][f"{tag}|{v}"] = row
            print(f"  {tag:26s} {v:11s} same-slot {row['same_slot_across_videos']:.2f}  "
                  f"dom-share {row['dominant_patch_share']:.2f}  global {row['global_frac']:.2f}")
    print("extras", out["extras"])

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"xvid_{args.dataset}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
