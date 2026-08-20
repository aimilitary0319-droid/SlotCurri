"""Plot the relative distribution of active slots over training for v26 family runs.

Reads the Lightning metrics.csv of each run and visualizes how gate mass is
spread across the 7 slots: winner share, top-2 share, effective slot count
exp(gate entropy), and the hard active-slot count.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/workspace/SlotCurri" if os.path.isdir("/workspace/SlotCurri") else "/mnt/ssd2/hmlee/SlotCurri"
RUNS = {
    "v26 (final)": ("_ytvis_attnmass_v26", "#1f77b4"),
    "v26f (fixed beta)": ("_ytvis_attnmass_v26f", "#ff7f0e"),
    "v26m (mass-only)": ("_ytvis_attnmass_v26m", "#2ca02c"),
}
N_SLOTS = 7


def load(run_dir):
    path = os.path.join(ROOT, "logs", run_dir, "metrics", "slotcurri", "metrics.csv")
    rows = list(csv.DictReader(open(path)))
    out = {}
    for key in ["train/active_slots", "train/gate_max", "train/gate_top2", "train/gate_entropy", "train/gate_n_half"]:
        pts = [(int(r["step"]), float(r[key])) for r in rows if r.get(key) not in (None, "")]
        steps = np.array([p[0] for p in pts], dtype=float)
        vals = np.array([p[1] for p in pts], dtype=float)
        out[key] = (steps, vals)
    return out


def smooth(x, k=15):
    if len(x) < k:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="valid")


def plot_line(ax, steps, vals, color, label, k=15):
    s = smooth(vals, k)
    off = len(steps) - len(s)
    ax.plot(steps[off:], s, color=color, label=label, lw=1.8)
    ax.plot(steps, vals, color=color, alpha=0.15, lw=0.8)


fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
data = {name: load(d) for name, (d, _) in RUNS.items()}

ax = axes[0][0]
for name, (_, color) in RUNS.items():
    steps, vals = data[name]["train/active_slots"]
    plot_line(ax, steps, vals, color, name)
ax.set_title("Active slots (gate > 0.5 count)")
ax.set_ylabel("slots")
ax.axhline(N_SLOTS, color="gray", ls=":", lw=0.8)
ax.legend(loc="upper left", fontsize=9)

ax = axes[0][1]
for name, (_, color) in RUNS.items():
    steps, vals = data[name]["train/gate_entropy"]
    plot_line(ax, steps, np.exp(vals), color, name)
ax.set_title("Effective slot count  exp(gate entropy)")
ax.set_ylabel("effective slots")
ax.axhline(N_SLOTS, color="gray", ls=":", lw=0.8)

ax = axes[1][0]
for name, (_, color) in RUNS.items():
    steps, vals = data[name]["train/gate_max"]
    plot_line(ax, steps, vals, color, name)
ax.set_title("Winner gate value (gate_max)")
ax.set_ylabel("gate")
ax.set_xlabel("step")

ax = axes[1][1]
for name, (_, color) in RUNS.items():
    smax, vmax = data[name]["train/gate_max"]
    stop2, vtop2 = data[name]["train/gate_top2"]
    n = min(len(vmax), len(vtop2))
    # gate_top2 is the SUM of the top-2 gates, so runner-up = top2_sum - max
    runner = np.maximum(vtop2[:n] - vmax[:n], 0.0)
    ratio = np.where(vmax[:n] > 1e-8, runner / np.maximum(vmax[:n], 1e-8), 0.0)
    plot_line(ax, smax[:n], ratio, color, name)
ax.set_title("Runner-up / winner gate ratio (1 = egalitarian)")
ax.set_ylabel("top2 / top1")
ax.set_ylim(0, 1.05)
ax.set_xlabel("step")

for row in axes:
    for a in row:
        a.grid(alpha=0.3)

fig.suptitle("Relative active-slot distribution over training (v26 / v26f / v26m)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out_path = os.path.join(ROOT, "event_analysis", "active_slot_dist_v26_family.png")
fig.savefig(out_path, dpi=130)
print("saved:", out_path)
