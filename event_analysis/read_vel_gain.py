"""Report the learned velocity parameters from a checkpoint.

v25 makes velocity an optional input rather than a supervised one, so the question it asks is
answered by a single number: did the predictor drive vel_gain away from its init (velocity is
useful) or toward zero (it is not)? Nothing logs vel_gain during training, so read it from the
checkpoint instead.

Usage:
  python event_analysis/read_vel_gain.py CKPT [CKPT ...]
"""

import sys

import torch


def main() -> int:
    for path in sys.argv[1:]:
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = [k for k in sd if "vel" in k.lower()]
        print(f"\n{path}")
        if not keys:
            print("  no velocity parameters in this checkpoint")
            continue
        for k in keys:
            t = sd[k].float()
            if t.numel() == 1:
                print(f"  {k:<58} {t.item():+.5f}")
            else:
                print(f"  {k:<58} shape {tuple(t.shape)}  absmean {t.abs().mean():.5f}  "
                      f"norm {t.norm():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
