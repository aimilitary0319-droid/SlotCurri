#!/usr/bin/env python3
"""FG object pixel-fraction distribution on MOVi-C val (class 0 = background)."""

from pathlib import Path

import numpy as np
import torch

from slotcurri import configuration, data


def main():
    root = Path("/workspace/SlotCurri")
    if not (root / "slotcurri").exists():
        root = Path("/mnt/ssd2/hmlee/SlotCurri")
    settings = root / "logs/_movi_c_attnmass_v29/settings/slotcurri/settings.yaml"
    if not settings.exists():
        settings = root / "configs/slotcurri/movi_c_attnmass_v29.yaml"
    cfg = configuration.load_config(str(settings))
    cfg.dataset.num_val_workers = 0
    cfg.dataset.val_batch_size = 1
    dm = data.build(cfg.dataset, data_dir="/workspace/dataset")
    dm.setup("fit")
    loader = dm.val_dataloader()

    fracs = []
    n_clips = 0
    for batch in loader:
        seg = batch["segmentations"]  # (B,T,C,H,W) bool or (B,T,H,W)
        if seg.ndim == 5:
            # one-hot, class 0 = bg
            pix = seg[0].reshape(seg.shape[1], seg.shape[2], -1).sum(dim=-1)  # T,C
            tot = float(seg[0, 0, 0].numel()) * seg.shape[1]
            for c in range(1, seg.shape[2]):
                s = float(pix[:, c].sum())
                if s > 0:
                    fracs.append(s / tot)
        else:
            t, h, w = seg.shape[-3:]
            tot = float(t * h * w)
            ids = seg[0]
            for c in ids.unique().tolist():
                if int(c) == 0:
                    continue
                s = float((ids == c).sum())
                if s > 0:
                    fracs.append(s / tot)
        n_clips += 1

    a = np.array(fracs, dtype=np.float64)
    print(f"clips={n_clips} fg_instances={len(a)}")
    qs = [0, 10, 25, 50, 75, 90, 100]
    print("percentiles %:", {q: round(float(np.percentile(a, q)) * 100, 3) for q in qs})
    for thr in (0.005, 0.01, 0.02, 0.03, 0.05, 0.10):
        print(f"  <{thr*100:4.1f}%: {(a<thr).mean()*100:5.1f}%   >={thr*100:4.1f}%: {(a>=thr).mean()*100:5.1f}%")


if __name__ == "__main__":
    main()
