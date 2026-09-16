"""Score v37 C_s statistics on a trained v37 ckpt vs feature-curriculum Z.

Holds last-iter attention A from a forward, then evaluates
  occupancy c = Σa²/ΣA
  π_v37 = (λ1-λ2)/mass
  π_v37g = λ1-λ2
  ρ = λ2/λ1
on Z from raw X and from X^bind at mix s ∈ {0, 0.5, 1}.

Winner slot is argmax occupancy (A-only, mix-independent) so the same
slot is compared across s.

Two A sources:
  eval: mix=1 Keys (as-trained eval). Isolates Z; same partition.
  train_s0: model.train() with feature_ncut_mix=0 so A is from leveled Keys.

Usage (container):
  python event_analysis/v37_cs_curriculum_probe.py --dataset ytvis --max-clips 24
  python event_analysis/v37_cs_curriculum_probe.py --dataset movi --max-clips 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from slotcurri import configuration, data, metrics, models
from slotcurri.modules.video import spectral_cs_impurity, spectral_slot_purity


class _FakeTrainer:
    def __init__(self, step: int):
        self.global_step = step


RUNS = {
    "ytvis": {
        "config": "configs/slotcurri/ytvis2021_attnmass_v37.yaml",
        "ckpt": "logs/_ytvis_attnmass_v37/checkpoints/slotcurri_step=step=100000-v1.ckpt",
    },
    "movi": {
        "config": "configs/slotcurri/movi_c_attnmass_v37.yaml",
        "ckpt": "logs/_movi_c_attnmass_v37/checkpoints/slotcurri_step=step=100000-v1.ckpt",
    },
}

MIXES = (0.0, 0.5, 1.0)


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    if float(den) < 1e-8:
        return float("nan")
    return float((a * b).sum() / den)


@torch.no_grad()
def _stats(att, z, proj_dim):
    a = att.float()
    mass = a.sum(-1)
    occ = (a * a).sum(-1) / mass.clamp_min(1e-8)
    pi = spectral_slot_purity(a, z, divide_by_mass=True, proj_dim=proj_dim)
    gap = spectral_slot_purity(a, z, divide_by_mass=False, proj_dim=proj_dim)
    rho = spectral_cs_impurity(a, z, proj_dim=proj_dim)
    return {
        "occ": occ,
        "pi": pi,
        "gap": gap,
        "rho": rho,
        "purity": (1.0 - rho).clamp(0.0, 1.0),
        "mass": mass,
    }


def _pack_frames(st):
    """st values are (F, S). Winner = argmax occupancy (A-only)."""
    occ = st["occ"]
    mx = occ.argmax(dim=-1)
    f = torch.arange(occ.shape[0], device=occ.device)
    out = {}
    rest_mask = torch.ones_like(occ, dtype=torch.bool)
    rest_mask[f, mx] = False
    for k, v in st.items():
        out[f"all_{k}"] = v.mean()
        out[f"win_{k}"] = v[f, mx].mean()
        out[f"rest_{k}"] = v[rest_mask].mean()
    q = st["pi"] / st["pi"].sum(-1, keepdim=True).clamp_min(1e-8)
    ent = -(q * q.clamp_min(1e-8).log()).sum(-1)
    out["rel_neff"] = ent.exp().mean()
    out["rel_sr"] = (st["pi"].sum(-1) / st["pi"].amax(-1).clamp_min(1e-8)).mean()
    out["rel_q1"] = (st["pi"].amax(-1) / st["pi"].sum(-1).clamp_min(1e-8)).mean()
    out["n_half"] = (st["pi"] > 0.5).float().sum(-1).mean()
    out["n_rho_low"] = (st["rho"] < 0.2).float().sum(-1).mean()
    return {k: float(v) for k, v in out.items()}


def load_model(cfg_path, ckpt_path):
    config = configuration.load_config(cfg_path)
    config.model.visualize = False
    config.dataset.val_batch_size = 1
    config.dataset.num_val_workers = 0
    built = {n: metrics.build(c) for n, c in config.val_metrics.items()}
    model = models.build(config.model, config.optimizer, None, built)
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd, strict=False)
    model = model.cuda().eval()
    model._trainer = _FakeTrainer(100000)
    return model, config


def bind_from_raw(ncut, raw, s):
    return ncut(raw, float(s))


def run_split(model, loader, max_clips, proj_dim, train_s0):
    inner = model._frame_encoder()
    ncut = inner.feature_ncut
    saved_mix = inner.feature_ncut_mix
    acc = {s: [] for s in MIXES}
    all_occ = []
    all_pi = {s: [] for s in MIXES}
    all_gap = {s: [] for s in MIXES}
    all_rho = {s: [] for s in MIXES}
    n_frames = 0
    n_clips = 0
    try:
        if train_s0:
            model.train()
            if hasattr(inner, "backbone"):
                inner.backbone.eval()
            inner.feature_ncut_mix = 0.0
        else:
            model.eval()
            inner.feature_ncut_mix = 1.0
        for batch in loader:
            batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
            if "batch_padding_mask" in batch:
                batch = model._remove_padding(batch, batch["batch_padding_mask"])
                if batch is None:
                    continue
            with torch.no_grad():
                outputs = model.forward(batch, train=train_s0, cycle=False)
            att = outputs["processor"]["state_attn_mask"].float()
            raw = outputs["encoder"]["backbone_features"].float()
            b, t, s_n, n = att.shape
            att_f = att.reshape(b * t, s_n, n)
            raw_f = raw.reshape(b * t, n, raw.shape[-1])
            occ = (att_f * att_f).sum(-1) / att_f.sum(-1).clamp_min(1e-8)
            all_occ.append(occ.cpu())
            for mix in MIXES:
                z = bind_from_raw(ncut, raw_f, mix)
                st = _stats(att_f, z, proj_dim)
                acc[mix].append(_pack_frames(st))
                all_pi[mix].append(st["pi"].cpu())
                all_gap[mix].append(st["gap"].cpu())
                all_rho[mix].append(st["rho"].cpu())
            n_frames += b * t
            n_clips += b
            if n_clips >= max_clips:
                break
    finally:
        inner.feature_ncut_mix = saved_mix
        model.eval()

    summary = {}
    for mix in MIXES:
        keys = acc[mix][0].keys()
        pack = {k: float(sum(d[k] for d in acc[mix]) / len(acc[mix])) for k in keys}
        pi = torch.cat(all_pi[mix])
        gap = torch.cat(all_gap[mix])
        rho = torch.cat(all_rho[mix])
        occ = torch.cat(all_occ)
        pack["corr_pi_occ"] = _corr(pi, occ)
        pack["corr_gap_occ"] = _corr(gap, occ)
        pack["corr_purity_occ"] = _corr(1.0 - rho, occ)
        pack["corr_pi_gap"] = _corr(pi, gap)
        summary[str(mix)] = pack

    pi0 = torch.cat(all_pi[0.0])
    pi1 = torch.cat(all_pi[1.0])
    rho0 = torch.cat(all_rho[0.0])
    rho1 = torch.cat(all_rho[1.0])
    occ = torch.cat(all_occ)
    mx = occ.argmax(dim=-1)
    f = torch.arange(occ.shape[0])
    summary["mean_dpi_s0_minus_raw"] = float((pi0 - pi1).mean())
    summary["win_dpi_s0_minus_raw"] = float((pi0[f, mx] - pi1[f, mx]).mean())
    summary["mean_drho_s0_minus_raw"] = float((rho0 - rho1).mean())
    summary["win_drho_s0_minus_raw"] = float((rho0[f, mx] - rho1[f, mx]).mean())
    summary["n_clips"] = n_clips
    summary["n_frames"] = n_frames
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(RUNS), required=True)
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--max-clips", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    spec = RUNS[args.dataset]
    print(f"loading {args.dataset} {spec['ckpt']}", flush=True)
    model, config = load_model(spec["config"], spec["ckpt"])
    proj = int(getattr(model, "amc_spectral_proj_dim", 64) or 64)
    print(f"spectral_proj_dim={proj}", flush=True)

    def make_loader():
        dm = data.build(config.dataset, data_dir=args.data_dir)
        dm.setup("validate")
        return dm.val_dataloader()

    print("eval A (mix=1 keys), Z at s=0/0.5/1", flush=True)
    eval_sum = run_split(model, make_loader(), args.max_clips, proj, train_s0=False)
    print("train A at s=0 keys, Z at s=0/0.5/1", flush=True)
    s0_sum = run_split(model, make_loader(), args.max_clips, proj, train_s0=True)
    out = {
        "dataset": args.dataset,
        "proj_dim": proj,
        "eval_A": eval_sum,
        "train_s0_A": s0_sum,
    }
    text = json.dumps(out, indent=2)
    print(text)
    path = args.out or f"logs/v37_cs_probe_{args.dataset}.json"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
