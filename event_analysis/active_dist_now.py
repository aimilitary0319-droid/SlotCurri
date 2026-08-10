import glob, os, math
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

N = 7
MAX_H = math.log(N)

def load(v):
    roots = glob.glob(f"/workspace/SlotCurri/logs/_ytvis_attnmass_{v}/**/events.out.tfevents*", recursive=True)
    if not roots:
        raise FileNotFoundError(v)
    ea = EventAccumulator(os.path.dirname(max(roots, key=os.path.getmtime)), size_guidance={"scalars": 0})
    ea.Reload()
    return ea

def series(ea, tag):
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(e.step, float(e.value)) for e in ea.Scalars(tag)]

def nearest(s, step, tol=150):
    best, bd = None, 1e18
    for st, v in s:
        d = abs(st - step)
        if d < bd:
            bd, best = d, (st, v)
    return best if best and bd <= tol else None

def fmt(x, w=6, p=3):
    return f"{x:{w}.{p}f}" if x is not None else f"{'-':>{w}}"

runs = ["v22", "v25", "v24s21", "v21"]
steps = [100, 300, 500, 700, 900, 1100, 1300, 2000, 3000, 4000, 5000]
tags = ["train/active_slots", "train/gate_n_half", "train/gate_max", "train/gate_top2", "train/gate_entropy", "train/gate_state_slots"]

print(f"N={N} maxH={MAX_H:.3f}  act%=active/7  2nd=top2-max  effS=exp(H)")
print()
for v in runs:
    try:
        ea = load(v)
    except Exception as e:
        print(v, e)
        continue
    data = {t: series(ea, t) for t in tags}
    last_act = data["train/active_slots"][-1][0] if data["train/active_slots"] else 0
    print(f"===== {v} (tb to step ~{last_act}) =====")
    print(f"{'step':>6} {'act':>6} {'act%':>5} {'n1/2':>6} {'gmax':>6} {'2nd':>6} {'H':>6} {'effS':>5} {'stAct':>6}")
    for st in steps:
        if st > last_act + 200:
            continue
        tol = 120 if st <= 1500 else 250
        row = {}
        for t, s in data.items():
            a = nearest(s, st, tol=tol)
            if a:
                row[t] = a[1]
        if "train/active_slots" not in row:
            continue
        act = row["train/active_slots"]
        nh = row.get("train/gate_n_half")
        gmax = row.get("train/gate_max")
        top2 = row.get("train/gate_top2")
        H = row.get("train/gate_entropy")
        stact = row.get("train/gate_state_slots")
        second = (top2 - gmax) if (top2 is not None and gmax is not None) else None
        eff = math.exp(H) if H is not None else None
        print(f"{st:6d} {fmt(act)} {fmt(100*act/N,5,1)} {fmt(nh)} {fmt(gmax)} {fmt(second)} {fmt(H)} {fmt(eff,5,2)} {fmt(stact)}")
    print()

BINS = [(0, 1.25, "~1"), (1.25, 2.0, "1.25-2"), (2.0, 3.5, "2-3.5"), (3.5, 5.0, "3.5-5"), (5.0, 7.5, "5-7")]
print("===== regime occupancy so far (step-weighted %) =====")
for v in runs:
    try:
        act = series(load(v), "train/active_slots")
    except Exception:
        continue
    if len(act) < 2:
        continue
    counts = {b[2]: 0.0 for b in BINS}
    total = 0.0
    for i in range(len(act) - 1):
        st0, v0 = act[i]
        st1, _ = act[i + 1]
        dt = max(st1 - st0, 1)
        # for finished v21, only first 1300 for fair compare with current runs
        horizon = 1300 if v == "v21" else 10**9
        if st0 > horizon:
            break
        for lo, hi, name in BINS:
            if lo <= v0 < hi:
                counts[name] += dt
                total += dt
                break
    parts = " ".join(f"{name}:{100*counts[name]/total:5.1f}%" for *_, name in BINS)
    label = f"to~{min(act[-1][0], 1300 if v=='v21' else act[-1][0])}"
    print(f"{v:8s} {label}: {parts}")
