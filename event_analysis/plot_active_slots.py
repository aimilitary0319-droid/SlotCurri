"""Plot the active_slots (and gate_p) trajectory over training for a run.

Usage: python event_analysis/plot_active_slots.py <metrics.csv> <out.png> [label]
"""
import sys
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else \
    "logs/_ytvis_attnmass_v2/metrics/slotcurri/metrics.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "event_analysis/ytvis_v2_active_slots.png"
LABEL = sys.argv[3] if len(sys.argv) > 3 else "ytvis soft v2 (no L1)"

steps, active, gate_p, featrec, ss = [], [], [], [], []
for r in csv.DictReader(open(CSV)):
    if not r.get("train/active_slots"):
        continue
    steps.append(int(r["step"]))
    active.append(float(r["train/active_slots"]))
    gate_p.append(float(r["train/gate_p"]) if r.get("train/gate_p") else np.nan)
    featrec.append(float(r["train/loss_featrec"]) if r.get("train/loss_featrec") else np.nan)
    ss.append(float(r["train/loss_ss"]) if r.get("train/loss_ss") else np.nan)

steps = np.array(steps); active = np.array(active); gate_p = np.array(gate_p)

print("%s: %d points, step %d..%d" % (LABEL, len(steps), steps.min(), steps.max()))
print("active_slots: start %.2f | min %.2f | max %.2f | end %.2f | mean %.2f"
      % (active[0], active.min(), active.max(), active[-1], active.mean()))
# summary at deciles of training
for q in [0, 10, 25, 50, 75, 90, 100]:
    i = min(int(q / 100 * (len(steps) - 1)), len(steps) - 1)
    print("  step %6d (%3d%%): active %.2f | gate_p %.4f" % (steps[i], q, active[i], gate_p[i]))

fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.plot(steps, active, color="tab:blue", lw=1.2, label="active_slots")
# moving average
if len(active) > 20:
    w = max(5, len(active) // 50)
    ma = np.convolve(active, np.ones(w) / w, mode="valid")
    ax1.plot(steps[w - 1:], ma, color="navy", lw=2.2, label=f"active_slots (MA{w})")
ax1.set_xlabel("training step")
ax1.set_ylabel("active_slots (sum of gates)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_ylim(bottom=0)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(steps, gate_p, color="tab:red", ls="--", lw=1.5, label="gate_p (threshold)")
ax2.set_ylabel("gate_p", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_ylim(bottom=0)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
ax1.set_title(f"{LABEL}: active_slots vs gate_p over training")
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print("saved ->", OUT)
