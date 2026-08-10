"""Compare active-slot / gate concentration distributions across runs."""
import glob
import math
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

N = 7
MAX_H = math.log(N)


def load(v):
    roots = glob.glob(
        f"/workspace/SlotCurri/logs/_ytvis_attnmass_{v}/**/events.out.tfevents*",
        recursive=True,
    )
    if not roots:
        # host path fallback
        roots = glob.glob(
            f"/mnt/ssd2/hmlee/SlotCurri/logs/_ytvis_attnmass_{v}/**/events.out.tfevents*",
            recursive=True,
        )
    if not roots:
        raise FileNotFoundError(v)
    path = max(roots, key=os.path.getmtime)
    ea = EventAccumulator(os.path.dirname(path), size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def series(ea, tag):
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(e.step, float(e.value)) for e in ea.Scalars(tag)]


def nearest(s, step, tol=200):
    best, bd = None, 1e18
    for st, v in s:
        d = abs(st - step)
        if d < bd:
            bd, best = d, (st, v)
    if best and bd <= tol:
        return best
    return None


def fmt(x, w=6, p=3):
    return f"{x:{w}.{p}f}" if x is not None else f"{'—':>{w}}"


TAGS = [
    "train/active_slots",
    "train/gate_n_half",
    "train/gate_max",
    "train/gate_top2",
    "train/gate_entropy",
    "train/gate_state_slots",
    "val/ari",
    "val/image_ari",
]
STEPS = [
    100, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10000,
    15000, 20000, 30000, 40000, 50000, 60000, 65000, 70000, 100000,
]
BINS = [(0, 1.25, "~1"), (1.25, 2.0, "1.25-2"), (2.0, 3.5, "2-3.5"),
        (3.5, 5.0, "3.5-5"), (5.0, 7.5, "5-7")]


def regime_share(act, step_max):
    counts = {b[2]: 0.0 for b in BINS}
    total = 0.0
    for i in range(len(act) - 1):
        st0, v0 = act[i]
        st1, _ = act[i + 1]
        if st0 > step_max:
            break
        dt = max(st1 - st0, 1)
        for lo, hi, name in BINS:
            if lo <= v0 < hi:
                counts[name] += dt
                total += dt
                break
    return {k: (100 * v / total if total else 0.0) for k, v in counts.items()}


def main():
    print(f"N_SLOTS={N}  maxH=log(7)={MAX_H:.4f}")
    print("act% = active/7; n1/2 = #(gate>0.5); effS=exp(H)")
    print("top2 = mean sum of top-2 gate values (not normalized to 1)")
    print()

    for v in ["v21", "v23", "v24"]:
        ea = load(v)
        data = {t: series(ea, t) for t in TAGS}
        print(f"===== {v} =====")
        print(
            f"{'step':>7} {'act':>6} {'act%':>5} {'n1/2':>6} {'gmax':>6} "
            f"{'top2':>6} {'2nd':>6} {'H':>6} {'effS':>5} {'stAct':>6} "
            f"{'ARI':>6} {'iARI':>6}"
        )
        for st in STEPS:
            tol = 150 if st <= 5000 else (300 if st <= 20000 else 600)
            row = {}
            for t, s in data.items():
                a = nearest(s, st, tol=tol)
                if a:
                    row[t] = a[1]
            if "train/active_slots" not in row and "val/ari" not in row:
                continue
            act = row.get("train/active_slots")
            nh = row.get("train/gate_n_half")
            gmax = row.get("train/gate_max")
            top2 = row.get("train/gate_top2")
            H = row.get("train/gate_entropy")
            stact = row.get("train/gate_state_slots")
            ari = row.get("val/ari")
            iari = row.get("val/image_ari")
            second = (top2 - gmax) if (top2 is not None and gmax is not None) else None
            eff = math.exp(H) if H is not None else None
            print(
                f"{st:7d} {fmt(act)} {fmt(100 * act / N if act else None, 5, 1)} "
                f"{fmt(nh)} {fmt(gmax)} {fmt(top2)} {fmt(second)} {fmt(H)} "
                f"{fmt(eff, 5, 2)} {fmt(stact)} {fmt(ari)} {fmt(iari)}"
            )
        print()

    print("===== regime occupancy (step-weighted %) =====")
    for v in ["v21", "v23", "v24"]:
        act = series(load(v), "train/active_slots")
        for horizon, label in [(10000, "0-10k"), (70000, "0-70k")]:
            share = regime_share(act, horizon)
            parts = " ".join(f"{name}:{share[name]:5.1f}%" for *_, name in BINS)
            print(f"{v:4s} [{label}] {parts}")
        print()

    print("===== concentration vs residual =====")
    print("max≈1 & 2nd≈0 => pure WTA; healthy early often max~0.9 with 2nd rising")
    for v in ["v21", "v23", "v24"]:
        ea = load(v)
        gmax_s = series(ea, "train/gate_max")
        top2_s = series(ea, "train/gate_top2")
        act_s = series(ea, "train/active_slots")
        print(f"-- {v} --")
        for st in [1000, 2000, 3000, 4000, 5000, 7500, 10000, 20000, 40000, 60000]:
            tol = 250 if st <= 10000 else 500
            a = nearest(gmax_s, st, tol)
            b = nearest(top2_s, st, tol)
            c = nearest(act_s, st, tol)
            if not a or not b:
                continue
            m, t = a[1], b[1]
            act = c[1] if c else None
            # If gates were a distribution over slots summing to active,
            # residual after top2 ≈ act - top2
            resid = (act - t) if act is not None else None
            print(
                f"  {a[0]:6d}: max={m:.3f} 2nd={t-m:.3f} top2={t:.3f} "
                f"act={act:.3f} resid_after_top2={resid:.3f}" if act is not None
                else f"  {a[0]:6d}: max={m:.3f} 2nd={t-m:.3f} top2={t:.3f}"
            )


if __name__ == "__main__":
    main()
