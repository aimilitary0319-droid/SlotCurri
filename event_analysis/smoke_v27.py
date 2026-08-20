"""End-to-end smoke test for v27 (v26 + L_pred + counterfactual slot-utility rent).

Builds the real v26 and v27 models from their configs and checks that

  - v27 adds no parameters, so a v26 checkpoint loads with nothing missing (both new
    terms are pure losses on existing modules)
  - with v26's weights loaded, v27's base losses (featrec, ss) are bit-identical to
    v26's on the same batch: the base objective is untouched
  - loss_pred / loss_util appear only in training mode, and the total decomposes exactly
    as featrec + 0.5*ss + w_eff_pred*loss_pred + w_eff_util*loss_util
  - both effective weights ride lambda_t: 0 at step 0, full at the end of the curriculum
  - the decoder is called exactly three times (main, pred, drop) with the right shapes;
    the pred pass receives a DETACHED gate slice and the drop pass runs under no_grad
    with exactly one previously-nonzero slot column zeroed per sample
  - gradient routing respects the detach discipline: loss_pred reaches the predictor
    and decoder; loss_util reaches the gate path (corrector) but NEVER the decoder
  - v26 itself gains no new loss keys (regression)

Usage (inside the container):
  python event_analysis/smoke_v27.py
"""

import torch

from slotcurri import configuration, models

V26 = "configs/slotcurri/ytvis2021_attnmass_v26.yaml"
V27 = "configs/slotcurri/ytvis2021_attnmass_v27.yaml"

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


def record_decoder_calls(model):
    """Patch the (time-mapping) decoder wrapper to log every call it receives."""
    calls = []
    orig = model.decoder.forward

    def wrapped(slots, active_mask=None, *a, **kw):
        calls.append(
            {
                "slots_shape": tuple(slots.shape),
                "mask": None if active_mask is None else active_mask.detach().clone(),
                "mask_requires_grad": bool(getattr(active_mask, "requires_grad", False)),
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        if active_mask is None:
            return orig(slots, *a, **kw)
        return orig(slots, active_mask, *a, **kw)

    model.decoder.forward = wrapped
    return calls, orig


def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.float().pow(2).sum())
    return total ** 0.5


def main():
    torch.manual_seed(0)
    cfg26, m26 = build(V26)
    cfg27, m27 = build(V27)
    anneal = int(cfg27.model.attn_mass_curriculum["anneal_steps"])

    print("\n1. no parameters added, checkpoints stay compatible")
    p26, p27 = dict(m26.named_parameters()), dict(m27.named_parameters())
    check("no parameters added", not set(p27) - set(p26), f"{sorted(set(p27) - set(p26))}")
    check("no parameters removed", not set(p26) - set(p27))
    missing, unexpected = m27.load_state_dict(m26.state_dict(), strict=False)
    check("v26 checkpoint loads into v27 cleanly",
          not missing and not unexpected, f"missing={list(missing)}")

    print("\n2. config parsing and the lambda ramp")
    check("pred weight parsed", abs(m27.pred_weight - 0.2) < 1e-12, f"{m27.pred_weight}")
    check("util weight parsed", abs(m27.util_weight - 0.05) < 1e-12, f"{m27.util_weight}")
    check("v26 has both terms off", m26.pred_weight == 0.0 and m26.util_weight == 0.0)
    m27._trainer = _FakeTrainer(0)
    check("lambda ramp is 0 at step 0", abs(m27._aux_ramp("lambda")) < 1e-9)
    m27._trainer = _FakeTrainer(anneal)
    check("lambda ramp is 1 at the end of the curriculum",
          abs(m27._aux_ramp("lambda") - 1.0) < 1e-9)
    check("'none' ramp is flat", m27._aux_ramp("none") == 1.0)

    HW = int(cfg27.dataset.val_pipeline.transforms.input_size)
    batch = {"video": torch.randn(2, 4, 3, HW, HW).cuda()}

    def probe(m, step, record=False):
        m._trainer = _FakeTrainer(int(step))
        m.train()
        torch.manual_seed(1234)  # decoded transition / dropped slot reproducible
        calls = orig = None
        if record:
            calls, orig = record_decoder_calls(m)
        out = m.forward(batch, train=True, cycle=False)
        total, losses = m.compute_loss(out)
        if record:
            m.decoder.forward = orig
        return out, total, losses, calls

    print("\n3. loss keys and exact total decomposition over the schedule")
    for step in (0, anneal // 2, anneal):
        _, total, losses, _ = probe(m27, step)
        w_pred = m27.pred_weight * m27._aux_ramp(m27.pred_ramp)
        w_util = m27.util_weight * m27._aux_ramp(m27.util_ramp)
        detail = ", ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
        print(f"    step {step}: {detail}")
        check(f"@{step} all losses finite",
              all(bool(torch.isfinite(v)) for v in losses.values()))
        check(f"@{step} loss_pred present", "loss_pred" in losses)
        if step == anneal:
            # gates are wide open at p_end (mass 1/7 >> p), so candidates must exist
            check(f"@{step} loss_util present", "loss_util" in losses)
            check(f"@{step} rel_delta stashed for logging",
                  m27._util_rel_delta is not None,
                  f"{m27._util_rel_delta}")
        recon = float(
            losses["loss_featrec"] + 0.5 * losses["loss_ss"]
            + w_pred * losses.get("loss_pred", torch.zeros(())).item()
            + w_util * losses.get("loss_util", torch.zeros(())).item()
        )
        check(f"@{step} total = featrec + 0.5*ss + w_eff*aux",
              abs(float(total) - recon) < 1e-5 * max(1.0, abs(recon)),
              f"total={float(total):.6f} recomposed={recon:.6f}")
        if "loss_util" in losses:
            check(f"@{step} loss_util in [0, 1]",
                  0.0 <= float(losses["loss_util"]) <= 1.0 + 1e-6)

    print("\n4. decoder traffic: main / pred / drop passes")
    out, total, losses, calls = probe(m27, anneal, record=True)
    n_expected = 2 + int("loss_util" in losses)
    check("decoder called main + pred (+ drop)", len(calls) == n_expected,
          f"{len(calls)} calls: {[c['slots_shape'] for c in calls]}")
    main_c, pred_c = calls[0], calls[1]
    check("main pass decodes all T frames", main_c["slots_shape"][1] == 4)
    check("pred pass decodes n_transitions frames",
          pred_c["slots_shape"][1] == int(cfg27.model.pred_recon["n_transitions"]),
          f"{pred_c['slots_shape']}")
    check("pred pass gate is detached", not pred_c["mask_requires_grad"])
    check("main pass gate is live", main_c["mask_requires_grad"])
    if len(calls) == 3:
        drop_c = calls[2]
        check("drop pass runs under no_grad", not drop_c["grad_enabled"])
        gate_full = out["processor"]["active_mask"].float()
        gate_drop = drop_c["mask"].float()
        zeroed = ((gate_full > 1e-6) & (gate_drop == 0)).all(dim=1)  # (B, S) whole column
        check("exactly one previously-open slot zeroed per sample",
              bool((zeroed.sum(-1) == 1).all()),
              f"zeroed per sample: {zeroed.sum(-1).tolist()}")
        keep = ~zeroed.unsqueeze(1).expand_as(gate_full)
        check("all other slots' gates untouched",
              float((gate_full - gate_drop)[keep].abs().max()) < 1e-6)

    print("\n5. gradient routing (the detach discipline)")
    m27.zero_grad(set_to_none=True)
    out, total, losses, _ = probe(m27, anneal)
    predictor = m27.processor.module.predictor
    corrector = m27.processor.module.corrector
    losses["loss_pred"].backward(retain_graph=True)
    gp = grad_norm(predictor.parameters())
    gd = grad_norm(m27.decoder.parameters())
    check("loss_pred trains the predictor", gp > 0, f"|grad|={gp:.3e}")
    check("loss_pred trains the decoder", gd > 0, f"|grad|={gd:.3e}")
    m27.zero_grad(set_to_none=True)
    if "loss_util" in losses:
        losses["loss_util"].backward(retain_graph=True)
        gd = grad_norm(m27.decoder.parameters())
        gc = grad_norm(corrector.parameters())
        check("loss_util never touches the decoder", gd == 0.0, f"|grad|={gd:.3e}")
        check("loss_util reaches the gate path (corrector)", gc > 0, f"|grad|={gc:.3e}")
    m27.zero_grad(set_to_none=True)
    total.backward()
    check("total backward is finite",
          all(torch.isfinite(p.grad).all() for p in m27.parameters()
              if p.grad is not None))
    m27.zero_grad(set_to_none=True)

    print("\n6. base objective untouched (v26 weights inside v27) and eval-mode hygiene")
    torch.manual_seed(1234)
    m26._trainer = _FakeTrainer(anneal)
    m26.train()
    out26 = m26.forward(batch, train=True, cycle=False)
    _, l26 = m26.compute_loss(out26)
    check("v26 gains no new loss keys",
          set(l26) == {"loss_featrec", "loss_ss"}, f"{sorted(l26)}")
    _, _, l27, _ = probe(m27, anneal)  # m27 carries v26's weights since section 1
    for key in ("loss_featrec", "loss_ss"):
        check(f"{key} identical to v26 under shared weights",
              torch.allclose(l26[key], l27[key], rtol=0, atol=1e-6),
              f"v26={float(l26[key]):.6f} v27={float(l27[key]):.6f}")
    m27.eval()
    with torch.no_grad():
        out_ev = m27.forward(batch, train=False, cycle=False)
        _, l_ev = m27.compute_loss(out_ev)
    check("aux terms are train-only",
          "loss_pred" not in l_ev and "loss_util" not in l_ev, f"{sorted(l_ev)}")

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
