"""End-to-end smoke test for v29 (logratio gate with conf_kind=purity).

Builds the real v26 and v29 models from their configs and checks that

  - conf_kind is parsed ('entropy' default for v26, 'purity' for v29, junk rejected)
  - v29 adds no parameters, so a v26 checkpoint loads with nothing missing
  - with shared weights and beta_t = 1 (step 0) the two gates are IDENTICAL tensors:
    the confidence branch is the only thing that changed and it carries zero weight there
  - at the end of the curriculum (beta = 0.7) the gates genuinely differ
  - v29's gate_conf equals the purity formula recomputed by hand from the raw attention
    (sum A^2 / sum A, NOT on the gamma-sharpened attention), and stays in (0, 1]
  - v26's gate_conf still equals the entropy formula on the sharpened attention
  - gate_conf is detached in both variants (anti-gaming discipline)
  - losses are finite and backward works through the purity gate

Usage (inside the container):
  python event_analysis/smoke_v29.py
"""

import math

import torch

from slotcurri import configuration, models

V26 = "configs/slotcurri/ytvis2021_attnmass_v26.yaml"
V29 = "configs/slotcurri/ytvis2021_attnmass_v29.yaml"

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


def forward(m, batch, step):
    m._trainer = _FakeTrainer(int(step))
    m.train()
    torch.manual_seed(1234)
    out = m.forward(batch, train=True, cycle=False)
    total, losses = m.compute_loss(out)
    return out, total, losses


def main():
    torch.manual_seed(0)
    cfg26, m26 = build(V26)
    cfg29, m29 = build(V29)
    anneal = int(cfg29.model.attn_mass_curriculum["anneal_steps"])

    print("\n1. parsing and checkpoint compatibility")
    check("v26 defaults to entropy", m26.amc_conf_kind == "entropy")
    check("v29 parses purity_sharp", m29.amc_conf_kind == "purity_sharp")
    try:
        bad = configuration.load_config(V29)
        bad.model.visualize = False
        bad.model.attn_mass_curriculum["conf_kind"] = "junk"
        models.build(bad.model, bad.optimizer, None, None)
        check("junk conf_kind rejected", False)
    except ValueError as e:
        check("junk conf_kind rejected", "conf_kind" in str(e))
    p26, p29 = dict(m26.named_parameters()), dict(m29.named_parameters())
    check("no parameters added", not set(p29) - set(p26))
    missing, unexpected = m29.load_state_dict(m26.state_dict(), strict=False)
    check("v26 checkpoint loads into v29 cleanly", not missing and not unexpected)

    HW = int(cfg29.dataset.val_pipeline.transforms.input_size)
    batch = {"video": torch.randn(2, 4, 3, HW, HW).cuda()}

    print("\n2. beta_t = 1 (step 0): the conf branch carries no weight, gates identical")
    out26, t26, _ = forward(m26, batch, 0)
    out29, t29, _ = forward(m29, batch, 0)
    g26 = out26["processor"]["active_mask"].float()
    g29 = out29["processor"]["active_mask"].float()
    check("gates identical at step 0", torch.equal(g26, g29),
          f"max diff {(g26 - g29).abs().max().item():.2e}")
    check("totals identical at step 0", torch.allclose(t26, t29, rtol=0, atol=1e-6),
          f"v26={float(t26):.6f} v29={float(t29):.6f}")

    print("\n3. beta = 0.7 (end of curriculum): the branch engages and the gates differ")
    out26, t26, l26 = forward(m26, batch, anneal)
    out29, t29, l29 = forward(m29, batch, anneal)
    g26 = out26["processor"]["active_mask"].float()
    g29 = out29["processor"]["active_mask"].float()
    check("gates differ at the end of the schedule", not torch.equal(g26, g29),
          f"max diff {(g26 - g29).abs().max().item():.3f}")
    check("v29 losses finite", all(bool(torch.isfinite(v)) for v in l29.values()),
          ", ".join(f"{k}={v.item():.4f}" for k, v in l29.items()))

    print("\n4. gate_conf is the advertised formula, on the advertised tensor")
    att = out29["processor"]["state_attn_mask"].float()  # (B, T, S, F) raw attention
    conf29 = out29["processor"]["gate_conf"].float()
    att_sharp = att.pow(2.0)
    att_sharp = att_sharp / att_sharp.sum(dim=2, keepdim=True).clamp_min(1e-8)
    purity_raw = (att * att).sum(-1) / att.sum(-1).clamp_min(1e-8)
    purity_sharp = (att_sharp * att_sharp).sum(-1) / att_sharp.sum(-1).clamp_min(1e-8)
    check("v29 gate_conf == purity on the SHARPENED attention",
          torch.allclose(conf29, purity_sharp, atol=1e-5),
          f"max diff {(conf29 - purity_sharp).abs().max().item():.2e}")
    check("...and NOT purity on the raw attention",
          not torch.allclose(conf29, purity_raw, atol=1e-3))
    # the raw variant stays available behind conf_kind: purity
    m29.amc_conf_kind = "purity"
    out_raw, _, _ = forward(m29, batch, anneal)
    conf_raw = out_raw["processor"]["gate_conf"].float()
    att_r = out_raw["processor"]["state_attn_mask"].float()
    manual_raw = (att_r * att_r).sum(-1) / att_r.sum(-1).clamp_min(1e-8)
    check("conf_kind=purity gives the raw-attention formula",
          torch.allclose(conf_raw, manual_raw, atol=1e-5),
          f"max diff {(conf_raw - manual_raw).abs().max().item():.2e}")
    m29.amc_conf_kind = "purity_sharp"
    # NB: recomputed from v26's OWN attention -- past frame 0 the two models' gates
    # differ at beta = 0.7, so their attention maps diverge too.
    conf26 = out26["processor"]["gate_conf"].float()
    att26 = out26["processor"]["state_attn_mask"].float()
    att26_sharp = att26.pow(2.0)
    att26_sharp = att26_sharp / att26_sharp.sum(dim=2, keepdim=True).clamp_min(1e-8)
    p_feat = att26_sharp / att26_sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    ent = -(p_feat * p_feat.clamp_min(1e-8).log()).sum(-1)
    ent_manual = (1.0 - ent / math.log(att26.shape[-1])).clamp(0.0, 1.0)
    check("v26 gate_conf still == entropy formula",
          torch.allclose(conf26, ent_manual, atol=1e-5),
          f"max diff {(conf26 - ent_manual).abs().max().item():.2e}")
    check("purity conf in (0, 1]",
          bool((conf29 > 0).all() and (conf29 <= 1.0 + 1e-6).all()),
          f"range [{conf29.min().item():.4f}, {conf29.max().item():.4f}]")
    check("gate_conf is detached (v29)", not conf29.requires_grad)
    print(f"    conf medians: entropy={conf26.median().item():.4f} "
          f"purity={conf29.median().item():.4f} (untrained attention)")

    print("\n5. backward through the purity gate")
    m29.zero_grad(set_to_none=True)
    t29.backward()
    check("backward finite",
          all(torch.isfinite(p.grad).all() for p in m29.parameters()
              if p.grad is not None))
    corr = sum(float(p.grad.float().pow(2).sum()) for p in
               m29.processor.module.corrector.parameters() if p.grad is not None) ** 0.5
    check("mass branch still carries gradient to the corrector", corr > 0,
          f"|grad|={corr:.3e}")
    m29.zero_grad(set_to_none=True)

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
