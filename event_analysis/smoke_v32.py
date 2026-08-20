"""End-to-end smoke test for v32 (purity gating: gate_form=purity_weight).

Builds the real v32 model from its config and checks that

  - the config parses (gate_form=purity_weight requires a purity conf_kind; the
    entropy default and junk values are rejected)
  - v32 adds no parameters, so a v26/v29 checkpoint loads with nothing missing
  - the gate IS the detached purity_sharp statistic recomputed by hand from the
    attention (no threshold, no temperature anywhere in the path)
  - the gate is schedule-free: global_step has no effect on the forward
    (_gate_threshold is None at every step)
  - active_mask and state_gate are the same tensor (one gate, two application points)
  - losses are finite and backward works (the gate is pure forward modulation, but
    featrec still reaches the corrector through the ungated paths)

Runs on CPU or GPU. The backbone is built with pretrained=False: every check here is
about wiring, not weights, and this keeps the smoke offline.

Usage (inside the container):
  python event_analysis/smoke_v32.py
"""

import torch

from slotcurri import configuration, models

V26 = "configs/slotcurri/ytvis2021_attnmass_v26.yaml"
V32 = "configs/slotcurri/ytvis2021_attnmass_v32.yaml"

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


class _FakeTrainer:
    def __init__(self, step=0):
        self.global_step = step


def build(cfg_path, device):
    config = configuration.load_config(cfg_path)
    config.model.visualize = False
    config.model.encoder.backbone.pretrained = False  # wiring smoke: weights irrelevant
    model = models.build(config.model, config.optimizer, None, None)
    model._trainer = _FakeTrainer(0)
    return config, model.to(device)


def forward(m, batch, step):
    m._trainer = _FakeTrainer(int(step))
    m.train()
    torch.manual_seed(1234)
    out = m.forward(batch, train=True, cycle=False)
    total, losses = m.compute_loss(out)
    return out, total, losses


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg32, m32 = build(V32, device)

    print("\n1. parsing and validation")
    check("gate_form parsed", m32.amc_gate_form == "purity_weight")
    check("conf_kind parsed", m32.amc_conf_kind == "purity_sharp")
    check("thresholdless: _gate_threshold(train) is None", m32._gate_threshold(True) is None)
    check("thresholdless: _gate_threshold(eval) is None", m32._gate_threshold(False) is None)
    check("no beta under purity_weight", m32._gate_beta(True) is None)
    try:
        bad = configuration.load_config(V32)
        bad.model.visualize = False
        bad.model.encoder.backbone.pretrained = False
        bad.model.attn_mass_curriculum["conf_kind"] = "entropy"
        models.build(bad.model, bad.optimizer, None, None)
        check("purity_weight + entropy rejected", False)
    except ValueError as e:
        check("purity_weight + entropy rejected", "purity_weight" in str(e))

    print("\n2. checkpoint compatibility (no parameters added)")
    _, m26 = build(V26, device)
    p26, p32 = dict(m26.named_parameters()), dict(m32.named_parameters())
    check("same parameter set as v26", set(p32) == set(p26))
    missing, unexpected = m32.load_state_dict(m26.state_dict(), strict=False)
    check("v26 checkpoint loads into v32 cleanly", not missing and not unexpected)

    HW = int(cfg32.dataset.val_pipeline.transforms.input_size)
    batch = {"video": torch.randn(2, 2, 3, HW, HW, device=device)}

    print("\n3. the gate is the statistic")
    out, total, losses = forward(m32, batch, 0)
    g = out["processor"]["active_mask"].float()
    att = out["processor"]["state_attn_mask"].float()  # (B, T, S, F)
    att_sharp = att.pow(2.0)
    att_sharp = att_sharp / att_sharp.sum(dim=2, keepdim=True).clamp_min(1e-8)
    purity_sharp = (att_sharp * att_sharp).sum(-1) / att_sharp.sum(-1).clamp_min(1e-8)
    check("gate == purity_sharp recomputed by hand",
          torch.allclose(g, purity_sharp.clamp(0.0, 1.0), atol=1e-5),
          f"max diff {(g - purity_sharp).abs().max().item():.2e}")
    check("gate detached", not out["processor"]["active_mask"].requires_grad)
    check("gate in (0, 1]", bool((g > 0).all() and (g <= 1.0 + 1e-6).all()),
          f"range [{g.min().item():.4f}, {g.max().item():.4f}]")
    check("state_gate is the same gate",
          torch.equal(out["processor"]["state_gate"], out["processor"]["active_mask"]))
    check("gate_conf exposed and equal to the gate",
          torch.equal(out["processor"]["gate_conf"], out["processor"]["active_mask"]))
    print(f"    untrained gate stats: mean={g.mean().item():.4f} (uniform would be "
          f"{1.0 / g.shape[-1]:.4f}), max/min ratio="
          f"{(g.amax(-1) / g.amin(-1).clamp_min(1e-8)).mean().item():.2f}")

    print("\n4. schedule-free: global_step does not touch the forward")
    out_late, total_late, _ = forward(m32, batch, 75000)
    g_late = out_late["processor"]["active_mask"].float()
    check("gates identical at step 0 and step 75k", torch.equal(g, g_late),
          f"max diff {(g - g_late).abs().max().item():.2e}")
    check("totals identical too", torch.allclose(total, total_late, rtol=0, atol=1e-6))

    print("\n5. losses and backward")
    check("losses finite", all(bool(torch.isfinite(v)) for v in losses.values()),
          ", ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))
    m32.zero_grad(set_to_none=True)
    total.backward()
    check("backward finite",
          all(torch.isfinite(p.grad).all() for p in m32.parameters()
              if p.grad is not None))
    corr = sum(float(p.grad.float().pow(2).sum()) for p in
               m32.processor.module.corrector.parameters() if p.grad is not None) ** 0.5
    check("corrector still receives gradient (ungated paths)", corr > 0,
          f"|grad|={corr:.3e}")
    m32.zero_grad(set_to_none=True)

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
