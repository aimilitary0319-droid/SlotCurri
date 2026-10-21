"""Does the same slot index win in every video? (MOVi-E collapse probe)

A. Data only. Key tokens at FC mix s are LayerNormed and L2-normalized. Mean
   pairwise cosine within a frame vs between frames of different videos.
B. Real config modules at random init (output_transform, SlotAttention,
   FixedLearnedInit), several seeds. Frame-0 Slot Attention (3 iters) on one
   random frame per video. Reports how often the frame's dominant slot is the
   same index across videos, and how the iteration-1 slot logits split into
   a part shared by all videos, a per-video part and a per-patch part.
   Architectures with the same FEAT_DIM are crossed with the data.
C. Optional trained checkpoint: which slot index carries the largest π per video.
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
import webdataset as wds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slotcurri import configuration, modules
from slotcurri.data import transforms
from slotcurri.featcur_start import _random_frame, _shard_paths
from slotcurri.modules.encoders import NcutRelationalLeveling, TimmExtractor

CONFIGS = {
    "ytvis": "configs/slotcurri/ytvis2021_attnmass_v39lam1gu.yaml",
    "movi_c": "configs/slotcurri/movi_c_attnmass_v39lam1gu.yaml",
    "movi_e": "configs/slotcurri/movi_e_attnmass_v39lam1gu.yaml",
}
S_LIST = (0.0, 0.5, 1.0)


def load_frames(cfg, data_dir, n_videos, per_shard, rng):
    tf = transforms.build(cfg.dataset.val_pipeline.transforms)["video"]
    paths = list(_shard_paths(str(cfg.dataset.train_shards), data_dir))
    paths = [paths[i] for i in rng.permutation(len(paths))]
    frames = []
    for path in paths:
        ds = wds.WebDataset([path], shardshuffle=False, nodesplitter=lambda u: u)
        ds = ds.map(
            lambda s: {k: v for k, v in s.items() if "video" in k or str(k).startswith("__")}
        ).decode("rgb")
        taken = 0
        for sample in ds:
            video = next(v for k, v in sample.items() if "video" in k and not str(k).startswith("__"))
            x = tf(_random_frame(np.asarray(video), rng))
            frames.append(x[0] if x.ndim == 4 else x)
            taken += 1
            if taken >= per_shard or len(frames) >= n_videos:
                break
        if len(frames) >= n_videos:
            break
    return frames


def build_arch(cfg, seed, device):
    torch.manual_seed(seed)
    ot = modules.build_module(cfg.model.encoder.output_transform).to(device).eval()
    sa = modules.build_grouper(cfg.model.grouper).to(device).eval()
    init = modules.build_initializer(cfg.model.initializer).to(device).eval()
    return ot, sa, init


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(CONFIGS))
    p.add_argument("--data-dir", default="/workspace/dataset")
    p.add_argument("--out-dir", default="event_analysis/global_winner")
    p.add_argument("--n-videos", type=int, default=256)
    p.add_argument("--per-shard", type=int, default=6)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--mem-fraction", type=float, default=0.25)
    p.add_argument("--ckpt", default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and args.mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    rng = np.random.default_rng(0)

    cfg = configuration.load_config(CONFIGS[args.dataset])
    bb = cfg.model.encoder.backbone
    feat_dim = int(cfg.globals.FEAT_DIM)
    backbone = TimmExtractor(
        model=bb.model, pretrained=True, frozen=True, features=bb.features,
        model_kwargs=dict(bb.get("model_kwargs") or {}) or None,
    ).to(device).eval()
    level = NcutRelationalLeveling(chunk_size=4, barrier=False).to(device)

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

    feat = {s: {"S": None, "Q": 0.0, "cin": [], "V": 0} for s in S_LIST}
    model_acc = {
        (tag, s, si): {"dom": [], "share": [], "lv": [], "patch_var": []}
        for tag in archs for s in S_LIST for si in range(args.seeds)
    }

    for b0 in range(0, len(frames), args.batch_size):
        x = torch.stack(frames[b0 : b0 + args.batch_size], 0).to(device)
        tok = backbone(x)
        tok = (tok[bb.features] if isinstance(tok, dict) else tok).float()
        n_tok = tok.shape[1]
        for s in S_LIST:
            key_tok = tok if s >= 1.0 else level(tok, s)
            u = F.normalize(F.layer_norm(key_tok, (key_tok.shape[-1],)), dim=-1)
            m = u.mean(1)
            st = feat[s]
            st["S"] = m.sum(0) if st["S"] is None else st["S"] + m.sum(0)
            st["Q"] += float(m.square().sum())
            st["V"] += m.shape[0]
            st["cin"] += ((n_tok * m.square().sum(-1) - 1.0) / (n_tok - 1)).tolist()
            for tag, seeds in archs.items():
                for si, (ot, sa, init) in enumerate(seeds):
                    fv = ot(tok)
                    fk = fv if s >= 1.0 else ot(key_tok)
                    slots0 = init(x.shape[0])
                    q = sa.to_q(sa.norm_slots(slots0))
                    k = sa.to_k(sa.norm_features(fk))
                    logit = torch.einsum("bsd,bnd->bsn", q, k) * sa.scale / sa.logit_div
                    lv = logit.mean(-1)
                    resid = logit - lv.unsqueeze(-1)
                    masks = sa(slots0, fv, n_iters=3, key_features=fk)["masks"]
                    dom = masks.mean(-1).argmax(-1)
                    win = masks.argmax(1)
                    share = (win == dom.unsqueeze(-1)).float().mean(-1)
                    acc = model_acc[(tag, s, si)]
                    acc["dom"] += dom.tolist()
                    acc["share"] += share.tolist()
                    acc["lv"].append(lv.cpu())
                    acc["patch_var"] += resid.var(1).mean(-1).tolist()

    out = {"dataset": args.dataset, "n_videos": len(frames), "features": {}, "models": {}}
    print(f"\n[A] {args.dataset}: key-token cosine (LN + L2)")
    for s in S_LIST:
        st = feat[s]
        V = st["V"]
        cross = (float(st["S"].square().sum()) - st["Q"]) / (V * (V - 1))
        cin = float(np.mean(st["cin"]))
        out["features"][str(s)] = {"within_frame": cin, "cross_video": cross}
        print(f"  s={s:.1f}  within-frame {cin:.3f}  cross-video {cross:.3f}  ratio {cross / cin:.3f}")

    print(f"\n[B] {args.dataset}: frame-0 Slot Attention at random init ({args.seeds} seeds)")
    for tag in archs:
        for s in S_LIST:
            agree, share, gvar, vvar, pvar = [], [], [], [], []
            for si in range(args.seeds):
                acc = model_acc[(tag, s, si)]
                dom = np.asarray(acc["dom"])
                counts = np.bincount(dom, minlength=1)
                agree.append(counts.max() / dom.size)
                share.append(float(np.mean(acc["share"])))
                lv = torch.cat(acc["lv"], 0)
                b = lv.mean(0)
                gvar.append(float(b.var()))
                vvar.append(float((lv - b).var(1).mean()))
                pvar.append(float(np.mean(acc["patch_var"])))
            tot = np.array(gvar) + np.array(vvar) + np.array(pvar)
            row = {
                "same_slot_across_videos": float(np.mean(agree)),
                "dominant_patch_share": float(np.mean(share)),
                "logit_var_global": float(np.mean(gvar)),
                "logit_var_video": float(np.mean(vvar)),
                "logit_var_patch": float(np.mean(pvar)),
                "global_frac": float(np.mean(np.array(gvar) / tot)),
            }
            out["models"][f"{tag}|s={s}"] = row
            print(
                f"  {tag:28s} s={s:.1f}  same-slot-across-videos {row['same_slot_across_videos']:.2f}"
                f"  dom-patch-share {row['dominant_patch_share']:.2f}"
                f"  logit var global/video/patch {row['logit_var_global']:.3f}/"
                f"{row['logit_var_video']:.3f}/{row['logit_var_patch']:.3f}"
                f"  (global {row['global_frac']:.2f})"
            )

    if args.ckpt:
        from slotcurri import models

        mcfg = configuration.load_config(CONFIGS[args.dataset])
        model = models.build(mcfg.model, mcfg.optimizer)
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=False)
        model = model.to(device).eval()
        top, val = [], []
        for b0 in range(0, len(frames), args.batch_size):
            x = torch.stack(frames[b0 : b0 + args.batch_size], 0).to(device)
            o = model({"video": x.unsqueeze(1)}, train=False)
            pi = o["processor"]["gate_conf"][:, 0].float()
            v, i = pi.max(-1)
            top += i.tolist()
            val += v.tolist()
        top = np.asarray(top)
        counts = np.bincount(top, minlength=int(mcfg.globals.NUM_SLOTS))
        out["ckpt"] = {
            "path": args.ckpt,
            "argmax_pi_counts": counts.tolist(),
            "max_pi_mean": float(np.mean(val)),
        }
        print(f"\n[C] {os.path.basename(args.ckpt)}: argmax-π slot counts {counts.tolist()}"
              f"  mean max π {np.mean(val):.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"global_winner_{args.dataset}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
