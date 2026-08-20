#!/usr/bin/env python3
"""Per-slot attention mass vs gate vs hard-mask pixels on MOVi-C v29 ckpt."""

from pathlib import Path

import torch

from slotcurri import configuration, data, models


@torch.no_grad()
def main():
    root = Path("/workspace/SlotCurri")
    if not (root / "slotcurri").exists():
        root = Path("/mnt/ssd2/hmlee/SlotCurri")
    settings = root / "logs/_movi_c_attnmass_v29/settings/slotcurri/settings.yaml"
    ckpt = root / "logs/_movi_c_attnmass_v29/checkpoints/slotcurri_step=step=55000.ckpt"
    device = torch.device("cpu")

    cfg = configuration.load_config(str(settings))
    cfg.model.visualize = False
    model = models.build(cfg.model, cfg.optimizer)
    raw = torch.load(str(ckpt), map_location="cpu")
    state = raw["state_dict"] if "state_dict" in raw else raw
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    dcfg = configuration.load_config(str(settings))
    dcfg.dataset.num_val_workers = 0
    dcfg.dataset.val_batch_size = 1
    dm = data.build(dcfg.dataset, data_dir="/workspace/dataset")
    dm.setup("fit")

    n = 3
    for si, batch in enumerate(dm.val_dataloader()):
        if si >= n:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model.forward(batch, train=False, cycle=False)
        aux = model.aux_forward(batch, out)
        proc = out["processor"]
        gate = proc["active_mask"].float()[0]  # T,S
        conf = proc.get("gate_conf")
        conf = conf.float()[0] if conf is not None else None

        grp = aux["grouping_masks"].float()[0]  # T,S,H,W
        att = grp / grp.sum(dim=1, keepdim=True).clamp_min(1e-8)
        mass = att.flatten(2).sum(-1) / att.shape[-1]  # T,S  (already ~sum_s = 1)
        # grouping may already be softmax; renormalize over slots to be sure
        mass = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        hard = aux.get("decoder_masks_vis_hard", aux["decoder_masks_hard"])[0]
        if hard.ndim == 4:  # T,S,H,W
            pix = hard.float().flatten(2).sum(-1)
            pix = pix / pix.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            pix = None

        print(f"\n=== sample {si}  (time-mean) ===")
        print(f"{'s':>3} {'mass':>8} {'gate':>8} {'hard%':>8} {'conf':>8}")
        S = mass.shape[-1]
        for s in range(S):
            m = float(mass[:, s].mean())
            g = float(gate[:, s].mean())
            p = float(pix[:, s].mean()) * 100 if pix is not None else float("nan")
            c = float(conf[:, s].mean()) if conf is not None else float("nan")
            print(f"{s:3d} {m:8.4f} {g:8.3f} {p:8.2f} {c:8.3f}")
        print(
            f"  mass sum={float(mass.mean(0).sum()):.3f}  "
            f"top1={float(mass.mean(0).max()):.3f}  "
            f"n_m>0.02={(mass.mean(0)>0.02).sum().item()}  "
            f"n_m>0.009={(mass.mean(0)>0.009).sum().item()}"
        )


if __name__ == "__main__":
    main()
