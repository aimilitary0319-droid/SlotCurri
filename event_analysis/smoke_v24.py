"""End-to-end smoke test for the separate temporal-mix gate (v24).

Builds the real v21 and v24 models from their configs and checks that

  - v24 adds no parameters, so a v21 checkpoint loads with nothing missing
  - the decoder is handed the *annealed* gate while the predictor re-gate uses the fixed one
  - v21 is unchanged: with state_p_mult unset the two gates are the same tensor
  - at the end of the p schedule the state gate is still selective while the decoder
    gate has gone flat, which is the entire point of the split
  - the corrector output reaches `state` ungated in every version, so the gate touches the
    temporal path exactly once
  - v23 (velocity predictor) and the no-gate baseline still run through the touched code

Usage (inside the container):
  python event_analysis/smoke_v24.py
"""

import torch

from slotcurri import configuration, models

V21 = "configs/slotcurri/ytvis2021_attnmass_v21.yaml"
V23 = "configs/slotcurri/ytvis2021_attnmass_v23.yaml"
V24 = "configs/slotcurri/ytvis2021_attnmass_v24.yaml"
BASE = "configs/slotcurri/ytvis2021.yaml"

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


class _FakeTrainer:
    def __init__(self, step=0):
        self.global_step = step


def build(cfg_path, step=0):
    config = configuration.load_config(cfg_path)
    config.model.visualize = False
    model = models.build(config.model, config.optimizer, None, None)
    model._trainer = _FakeTrainer(step)
    return config, model.cuda()


def record_decoder_mask(model):
    """Capture the mask handed to the decoder, to prove which of the two gates it receives.

    Patched on the outer (time-mapping) wrapper so the recorded tensor still has the
    full (B, T, S) shape rather than a per-frame slice.
    """
    seen = {}
    orig = model.decoder.forward

    def wrapped(slots, active_mask=None, *a, **kw):
        seen.setdefault("mask", active_mask)
        return orig(slots, active_mask, *a, **kw)

    model.decoder.forward = wrapped
    return seen


def main():
    torch.manual_seed(0)
    cfg21, m21 = build(V21)
    cfg24, m24 = build(V24)

    print("\n1. no parameters added, checkpoints stay compatible")
    p21, p24 = dict(m21.named_parameters()), dict(m24.named_parameters())
    check("no parameters added", not set(p24) - set(p21), f"{sorted(set(p24) - set(p21))}")
    check("no parameters removed", not set(p21) - set(p24))
    missing, unexpected = m24.load_state_dict(m21.state_dict(), strict=False)
    check("v21 checkpoint loads into v24 cleanly",
          not missing and not unexpected, f"missing={list(missing)}")

    print("\n2. thresholds are parsed as intended")
    check("v21 has no state threshold", m21._state_gate_threshold(True) is None)
    anneal = int(cfg24.model.attn_mass_curriculum["anneal_steps"])
    p_start = cfg24.model.attn_mass_curriculum["p_start_mult"]
    p_end = cfg24.model.attn_mass_curriculum["p_end_mult"]
    sp_mult = cfg24.model.attn_mass_curriculum["state_p_mult"]
    cross = anneal * (p_start - sp_mult) / (p_start - p_end)
    # a floor, so it equals the schedule before the crossover and the floor after
    m24._trainer = _FakeTrainer(0)
    check("v24 state threshold follows the schedule early",
          abs(m24._state_gate_threshold(True) - m24._gate_threshold(True)) < 1e-12,
          f"{m24._state_gate_threshold(True):.6f} at step 0")
    m24._trainer = _FakeTrainer(anneal)
    check("v24 state threshold holds at 0.5/7 at the end of the schedule",
          abs(m24._state_gate_threshold(True) - sp_mult / 7) < 1e-12,
          f"{m24._state_gate_threshold(True):.6f}")
    check("v24 eval threshold is the floor too",
          abs(m24._state_gate_threshold(False) - sp_mult / 7) < 1e-12)
    print(f"  crossover at step {cross:,.0f} of {anneal:,}")

    HW = int(cfg24.dataset.val_pipeline.transforms.input_size)
    batch = {"video": torch.randn(2, 4, 3, HW, HW).cuda()}

    def probe(m, step):
        m._trainer = _FakeTrainer(int(step))
        m.train()
        seen = record_decoder_mask(m)
        out = m.forward(batch, train=True, cycle=False)
        total, _ = m.compute_loss(out)
        proc = out["processor"]
        am, sg = proc["active_mask"].float(), proc["state_gate"].float()
        # after max-normalization, this is the gate that actually scales the temporal mix
        n_state = sg / sg.amax(-1, keepdim=True).clamp_min(1e-8)
        n_dec = am / am.amax(-1, keepdim=True).clamp_min(1e-8)
        # gamma-sharpened mass the gates are thresholding, recomputed as build_gate does it
        att = proc["state_attn_mask"].float().pow(m.amc_mass_gamma)
        att = att / att.sum(dim=2, keepdim=True).clamp_min(1e-8)
        return {
            "p": m._gate_threshold(True), "total": total, "decoder_saw": seen["mask"],
            "am": am, "sg": sg, "n_state": n_state, "mass": att.sum(-1) / att.shape[-1],
            "state": proc["state"],
            "corrector": proc["corrector"][m.processor.module.state_key],
            "slots_dec": am.sum(-1).mean().item(), "slots_state": sg.sum(-1).mean().item(),
            "spread_dec": (n_dec.amax(-1) - n_dec.amin(-1)).mean().item(),
            "spread_state": (n_state.amax(-1) - n_state.amin(-1)).mean().item(),
            "floor_dec": n_dec.amin(-1).mean().item(),
            "floor_state": n_state.amin(-1).mean().item(),
        }

    print("\n3. the two gates over the schedule (untrained weights: masses are near-uniform,"
          "\n   so these show the mechanism, not the effect size a trained model would give)")
    print(f"  {'step':>7} {'ver':<4} {'p':>8} {'slots_dec':>10} {'slots_st':>9} "
          f"{'spread_dec':>11} {'spread_st':>10} {'weakest_st':>11}")
    probes = {}
    for step in (0, int(cross), anneal):
        for tag, m in (("v21", m21), ("v24", m24)):
            r = probe(m, step)
            probes[(tag, step)] = r
            print(f"  {step:>7} {tag:<4} {r['p']:>8.5f} {r['slots_dec']:>10.3f} "
                  f"{r['slots_state']:>9.3f} {r['spread_dec']:>11.3f} "
                  f"{r['spread_state']:>10.3f} {r['floor_state']:>11.3f}")
            check(f"{tag}@{step} loss is finite", bool(torch.isfinite(r["total"])))
            # the decoder must always be handed the annealed gate, never the state gate
            got = r["decoder_saw"]
            check(f"{tag}@{step} decoder receives active_mask",
                  got is not None and torch.equal(got.float(), r["am"]))
            # the corrector output must reach `state` untouched, so the gate enters the
            # temporal path once (at the predictor re-gate) and not twice
            check(f"{tag}@{step} corrector output is ungated",
                  torch.equal(r["state"], r["corrector"]),
                  f"max diff {(r['state'] - r['corrector']).abs().max().item():.2e}")

    print("\n4. v21 is unchanged, and v24 only diverges after the crossover")
    for step in (0, int(cross), anneal):
        check(f"v21@{step} state gate is identical to its decoder gate",
              torch.equal(probes[("v21", step)]["am"], probes[("v21", step)]["sg"]))
    # before the crossover the floor is inactive, so v24 must be v21 exactly
    for step in (0, int(cross)):
        check(f"v24@{step} state gate still equals its decoder gate (floor inactive)",
              torch.equal(probes[("v24", step)]["am"], probes[("v24", step)]["sg"]))
        check(f"v24@{step} matches v21's early capacity restriction",
              abs(probes[("v24", step)]["slots_state"]
                  - probes[("v21", step)]["slots_state"]) < 0.2,
              f"{probes[('v24', step)]['slots_state']:.3f} vs "
              f"{probes[('v21', step)]['slots_state']:.3f}")
    # after it, the decoder gate has gone flat and the state gate has not
    s24, s21 = probes[("v24", anneal)], probes[("v21", anneal)]
    check("v24 state gate differs from its decoder gate at p_end",
          not torch.equal(s24["am"], s24["sg"]))
    check("v24 keeps state selectivity at p_end",
          s24["spread_state"] > s24["spread_dec"],
          f"state spread {s24['spread_state']:.3f} vs decoder {s24['spread_dec']:.3f}")
    check("v24 throttles the weakest slot more than v21 does at p_end",
          s24["floor_state"] < s21["floor_state"] - 0.05,
          f"v24 {s24['floor_state']:.3f} vs v21 {s21['floor_state']:.3f}")
    # the decoder must not merely differ from the state gate, it must be the other tensor
    check("v24 decoder is not handed the state gate at p_end",
          not torch.equal(s24["decoder_saw"].float(), s24["sg"]))

    print(f"\n4b. the two multipliers slot by slot, v24 at step {anneal:,} (sample 0, frame 0)")
    print("    g_dec is the raw value the decoder receives; g_state is post-max-norm, i.e.")
    print("    the actual coefficient on Pred() in the temporal mix")
    order = s24["mass"][0, 0].argsort(descending=True)
    print("      " + "".join(f"{'slot ' + str(int(i)):>9}" for i in order))
    for label, key in (("mass", "mass"), ("g_dec", "am"), ("g_state", "n_state")):
        row = s24[key][0, 0][order]
        print(f"      {label:<7}" + "".join(f"{v:>9.3f}" for v in row.tolist()))
    dec_ratio = (s24["am"][0, 0].max() / s24["am"][0, 0].min()).item()
    st_ratio = (s24["n_state"][0, 0].max() / s24["n_state"][0, 0].min()).item()
    print(f"      strongest/weakest: decoder {dec_ratio:.2f}x, temporal {st_ratio:.2f}x")
    check("the decoder's gate is the flatter of the two", dec_ratio < st_ratio)

    print("\n5. logging hooks are populated")
    check("state gate mean is stashed for logging",
          getattr(m24, "_state_gate_mean", None) is not None,
          f"{m24._state_gate_mean:.4f}" if m24._state_gate_mean is not None else "None")
    check("v21 stash reflects its bool gate",
          getattr(m21, "_state_gate_mean", "missing") != "missing")

    print("\n6. cyclic inference threads the new output key")
    m24.eval()
    with torch.no_grad():
        cyc = m24.forward(batch, train=False, cycle=True)
    check("cyclic forward runs", bool(torch.isfinite(cyc["processor"]["state"]).all()))
    check("state_gate survives the backward sweep merge",
          cyc["processor"]["state_gate"].shape == cyc["processor"]["active_mask"].shape,
          f"{tuple(cyc['processor']['state_gate'].shape)}")
    with torch.no_grad():
        aux = m24.aux_forward(batch, cyc)
    check("aux_forward still produces hard masks",
          "decoder_masks_hard" in aux and torch.isfinite(aux["decoder_masks_hard"]).all())

    print("\n7. v23 and the no-gate baseline are unaffected")
    del m21, m24
    torch.cuda.empty_cache()
    _, m23 = build(V23)
    m23.train()
    out23 = m23.forward(batch, train=True, cycle=False)
    t23, l23 = m23.compute_loss(out23)
    check("v23 still runs with loss_dyn", "loss_dyn" in l23 and bool(torch.isfinite(t23)),
          ", ".join(f"{k}={v.item():.4f}" for k, v in l23.items()))
    check("v23 state gate equals its decoder gate (state_p_mult unset)",
          torch.equal(out23["processor"]["active_mask"], out23["processor"]["state_gate"]))
    del m23
    torch.cuda.empty_cache()

    cfgb, mb = build(BASE)
    HWb = int(cfgb.dataset.val_pipeline.transforms.input_size)
    bb = {"video": torch.randn(2, 4, 3, HWb, HWb).cuda()}
    mb.train()
    outb = mb.forward(bb, train=True, cycle=False)
    tb, lb = mb.compute_loss(outb)
    check("baseline still runs", bool(torch.isfinite(tb)),
          ", ".join(f"{k}={v.item():.4f}" for k, v in lb.items()))
    check("baseline state gate falls back to all-active",
          bool(outb["processor"]["state_gate"].all()))

    failed = [n for n, ok in results if not ok]
    print("\n" + "=" * 70)
    if failed:
        print(f"FAILED {len(failed)}/{len(results)}:")
        for n in failed:
            print(f"  - {n}")
        raise SystemExit(1)
    print(f"all {len(results)} checks passed")


if __name__ == "__main__":
    main()
