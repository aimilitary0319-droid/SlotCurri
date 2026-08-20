"""End-to-end smoke test for v33 (v32 purity gating + feature curriculum).

Builds the real v33 model from its config and checks that

  - the config parses (feature_curriculum block; junk schedules are rejected)
  - v33 adds no parameters, so a v32 checkpoint loads with nothing missing
    (FeatureSmoothing is parameter-free)
  - the schedule is correct at the endpoints (mix 0 at step 0, 1 at anneal_steps,
    stays 1 after) and the model initializes the encoder at mix 0
  - the encoder smooths ONLY in train mode: train-mode backbone tokens differ from
    eval-mode tokens and have lower within-frame variance; at mix = 1 train equals eval
    (the curriculum has annealed itself out of the model)
  - under smoothed features the purity gate is still exactly the recomputed statistic
    (curriculum and gate compose without touching each other)
  - losses are finite and backward reaches the corrector

Runs on CPU or GPU. The backbone is built with pretrained=False: every check here is
about wiring, not weights, and this keeps the smoke offline.

Usage (inside the container):
  python event_analysis/smoke_v33.py
"""

import torch

from slotcurri import configuration, models

V32 = "configs/slotcurri/ytvis2021_attnmass_v32.yaml"
V33 = "configs/slotcurri/ytvis2021_attnmass_v33.yaml"

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


def forward(m, batch, mix):
    m._frame_encoder().feature_smoothing_mix = float(mix)
    m.train()
    torch.manual_seed(1234)
    out = m.forward(batch, train=True, cycle=False)
    total, losses = m.compute_loss(out)
    return out, total, losses


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg33, m33 = build(V33, device)

    print("\n1. parsing and validation")
    check("feature curriculum enabled", m33.featcur_enabled)
    check("anneal_steps parsed", m33.featcur_anneal_steps == 30000)
    check("schedule parsed", m33.featcur_schedule == "cosine")
    inner = m33._frame_encoder()
    check("smoothing module attached", inner.feature_smoothing is not None)
    check("tau parsed", abs(inner.feature_smoothing.tau - 0.1) < 1e-9)
    check("window parsed", inner.feature_smoothing.window == 9)
    check("encoder starts at mix 0", inner.feature_smoothing_mix == 0.0)
    try:
        bad = configuration.load_config(V33)
        bad.model.visualize = False
        bad.model.encoder.backbone.pretrained = False
        bad.model.feature_curriculum["schedule"] = "banana"
        models.build(bad.model, bad.optimizer, None, None)
        check("junk schedule rejected", False)
    except ValueError as e:
        check("junk schedule rejected", "schedule" in str(e))

    print("\n2. checkpoint compatibility (no parameters added)")
    _, m32 = build(V32, device)
    p32, p33 = dict(m32.named_parameters()), dict(m33.named_parameters())
    check("same parameter set as v32", set(p33) == set(p32))
    missing, unexpected = m33.load_state_dict(m32.state_dict(), strict=False)
    check("v32 checkpoint loads into v33 cleanly", not missing and not unexpected)

    print("\n3. schedule endpoints")
    T = m33.featcur_anneal_steps
    check("mix(0) = 0", models.feature_curriculum_mix(0, T, "cosine") == 0.0)
    check("mix(T) = 1", models.feature_curriculum_mix(T, T, "cosine") == 1.0)
    check("mix(2T) = 1", models.feature_curriculum_mix(2 * T, T, "cosine") == 1.0)
    check("cosine midpoint = 0.5",
          abs(models.feature_curriculum_mix(T // 2, T, "cosine") - 0.5) < 1e-9)

    HW = int(cfg33.dataset.val_pipeline.transforms.input_size)
    batch = {"video": torch.randn(1, 2, 3, HW, HW, device=device)}

    print("\n4. train-only smoothing")
    inner.feature_smoothing_mix = 0.0
    m33.train()
    with torch.no_grad():
        enc_train = m33.encoder(batch["video"])["backbone_features"]
    m33.eval()
    with torch.no_grad():
        enc_eval = m33.encoder(batch["video"])["backbone_features"]
    check("train tokens differ from eval tokens (mix 0)",
          not torch.allclose(enc_train, enc_eval))
    var_train = enc_train.flatten(0, 1).var(dim=1).mean().item()
    var_eval = enc_eval.flatten(0, 1).var(dim=1).mean().item()
    check("smoothing reduces within-frame token variance", var_train < var_eval,
          f"train {var_train:.4e} vs eval {var_eval:.4e}")
    m33.train()
    inner.feature_smoothing_mix = 1.0
    with torch.no_grad():
        enc_done = m33.encoder(batch["video"])["backbone_features"]
    check("mix 1 in train mode equals eval (curriculum annealed out)",
          torch.equal(enc_done, enc_eval))

    print("\n5. gate composes with the curriculum")
    out, total, losses = forward(m33, batch, 0.0)
    g = out["processor"]["active_mask"].float()
    att = out["processor"]["state_attn_mask"].float()  # (B, T, S, F)
    att_sharp = att.pow(2.0)
    att_sharp = att_sharp / att_sharp.sum(dim=2, keepdim=True).clamp_min(1e-8)
    purity_sharp = (att_sharp * att_sharp).sum(-1) / att_sharp.sum(-1).clamp_min(1e-8)
    check("gate == purity_sharp recomputed by hand (smoothed features)",
          torch.allclose(g, purity_sharp.clamp(0.0, 1.0), atol=1e-5),
          f"max diff {(g - purity_sharp).abs().max().item():.2e}")
    check("gate detached", not out["processor"]["active_mask"].requires_grad)

    print("\n6. losses and backward")
    check("losses finite", all(bool(torch.isfinite(v)) for v in losses.values()),
          ", ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))
    m33.zero_grad(set_to_none=True)
    total.backward()
    check("backward finite",
          all(torch.isfinite(p.grad).all() for p in m33.parameters()
              if p.grad is not None))
    corr = sum(float(p.grad.float().pow(2).sum()) for p in
               m33.processor.module.corrector.parameters() if p.grad is not None) ** 0.5
    check("corrector receives gradient under the curriculum", corr > 0,
          f"|grad|={corr:.3e}")
    m33.zero_grad(set_to_none=True)

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
