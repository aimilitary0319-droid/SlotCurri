"""Audit how hard the v26 gate actually suppresses ghost slots.

Uses the cached v20 statistics (m, c, wins) and answers, at several curriculum
stages lambda:
  - what gate value g do ghost slots receive (distribution)?
  - what fraction of ghosts "pass" (g > 0.5)? how open are the passing ones?
  - what relative weight do ghosts keep after per-frame normalization
    (temporal alpha = g / max_j g, and decoder-side share g_i / sum_j g_j)?
"""
import math

import numpy as np

EPS = 1e-6
CACHE = "event_analysis/tau_g_stats_v20.npz"

P_START_MULT, P_END_MULT = 1.5, 0.1
BETA_FINAL = 0.7
TAU_G = 0.25
SMALL_FRAC = 0.05

blob = np.load(CACHE)
m, c, wins = blob["m"], blob["c"], blob["wins"]  # (N_frames, S)
n_frames, n_slots = m.shape
f_total = wins.sum(axis=1).max()

ghost = wins == 0
small = (wins > 0) & (wins <= SMALL_FRAC * f_total)
large = wins > SMALL_FRAC * f_total

print(f"frames={n_frames}  slots/frame={n_slots}  patches/frame={int(f_total)}")
print(f"slot-frame counts: ghost={ghost.sum()} ({ghost.mean():.1%})  "
      f"small={small.sum()} ({small.mean():.1%})  large={large.sum()} ({large.mean():.1%})")
print(f"frames with >=1 ghost slot: {(ghost.any(axis=1)).mean():.1%}")
print()

header = (f"{'lam':>5s} {'p_mult':>6s} {'beta':>5s} | {'ghost g med':>11s} {'g p90':>6s}"
          f" {'g>0.5':>6s} {'g>0.9':>6s} | {'small g med':>11s} {'large g med':>11s}")
print(header)
print("-" * len(header))
for lam in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
    p_mult = (1 - lam) * P_START_MULT + lam * P_END_MULT
    beta = 1.0 - (1.0 - BETA_FINAL) * lam
    p = p_mult / n_slots
    log_r = beta * np.log(m + EPS) + (1 - beta) * np.log(c + EPS)
    g = 1.0 / (1.0 + np.exp(-(log_r - math.log(p + EPS)) / TAU_G))
    gg, gs, gl = g[ghost], g[small], g[large]
    print(f"{lam:5.2f} {p_mult:6.3f} {beta:5.2f} | {np.median(gg):11.3f}"
          f" {np.quantile(gg, 0.9):6.3f} {float((gg > 0.5).mean()):6.1%}"
          f" {float((gg > 0.9).mean()):6.1%} | {np.median(gs):11.3f} {np.median(gl):11.3f}")

# ---- final stage, per-frame relative weights ----
lam = 1.0
p = P_END_MULT / n_slots
log_r = BETA_FINAL * np.log(m + EPS) + (1 - BETA_FINAL) * np.log(c + EPS)
g = 1.0 / (1.0 + np.exp(-(log_r - math.log(p + EPS)) / TAU_G))

alpha = g / g.max(axis=1, keepdims=True)          # temporal update fraction
share = g / g.sum(axis=1, keepdims=True)          # decoder-side gate share
uniform_share = 1.0 / n_slots

print()
print("=== final stage (lambda=1): what do PASSING ghosts actually get? ===")
gg = g[ghost]
passing = ghost & (g > 0.5)
print(f"ghost gate quantiles: p10={np.quantile(gg, .1):.3f}  med={np.median(gg):.3f}"
      f"  p90={np.quantile(gg, .9):.3f}  p99={np.quantile(gg, .99):.3f}")
print(f"ghosts with g>0.5: {float((gg > 0.5).mean()):.1%}   g>0.9: {float((gg > 0.9).mean()):.1%}")
if passing.sum():
    print(f"passing ghosts' g: med={np.median(g[passing]):.3f}  p90={np.quantile(g[passing], .9):.3f}")
    print(f"passing ghosts' temporal alpha (g/max): med={np.median(alpha[passing]):.3f}")
    print(f"passing ghosts' decoder gate share: med={np.median(share[passing]):.4f}"
          f"  (uniform=1/{n_slots}={uniform_share:.4f})")
print()
print(f"ALL ghosts   temporal alpha: med={np.median(alpha[ghost]):.3f}  p90={np.quantile(alpha[ghost], .9):.3f}")
print(f"ALL ghosts   decoder share : med={np.median(share[ghost]):.4f}  p90={np.quantile(share[ghost], .9):.4f}")
print(f"small slots  decoder share : med={np.median(share[small]):.4f}")
print(f"large slots  decoder share : med={np.median(share[large]):.4f}")
print()
frames_with_pass = (passing.any(axis=1)).mean()
print(f"frames with >=1 passing ghost: {frames_with_pass:.1%}")
n_pass_per_frame = passing.sum(axis=1)
print(f"passing ghosts per frame: mean={n_pass_per_frame.mean():.2f}  max={n_pass_per_frame.max()}")
