"""End-to-end smoke test for the velocity-conditioned predictor (v23).

Builds the real v21 and v23 models from their configs and checks that

  - v23 adds exactly the two expected parameters and nothing else
  - forward + compute_loss run and produce a finite total with `loss_dyn` present in v23
    and absent in v21
  - backward reaches vel_proj and vel_gain through the full model, not just in isolation
  - a v21 checkpoint loads into a v23 model with only the velocity parameters missing
  - the baseline config, which shares all the touched code paths, still runs

Usage (inside the container):
  python event_analysis/smoke_v23.py
"""

import torch

from slotcurri import configuration, models

V21 = "configs/slotcurri/ytvis2021_attnmass_v21.yaml"
V23 = "configs/slotcurri/ytvis2021_attnmass_v23.yaml"
BASE = "configs/slotcurri/ytvis2021.yaml"

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


class _FakeTrainer:
    global_step = 0


def build(cfg_path):
    config = configuration.load_config(cfg_path)
    config.model.visualize = False
    model = models.build(config.model, config.optimizer, None, None)
    model._trainer = _FakeTrainer()
    return config, model.cuda()


def run_losses(model, batch):
    model.train()
    outputs = model.forward(batch, train=True, cycle=False)
    total, losses = model.compute_loss(outputs)
    return outputs, total, losses


def main():
    torch.manual_seed(0)
    cfg21, m21 = build(V21)
    cfg23, m23 = build(V23)

    print("\n1. parameter delta")
    p21 = dict(m21.named_parameters())
    p23 = dict(m23.named_parameters())
    added = sorted(set(p23) - set(p21))
    removed = sorted(set(p21) - set(p23))
    n_added = sum(p23[k].numel() for k in added)
    print(f"  added: {added}")
    check("only the two velocity parameters are added",
          added == ["processor.module.predictor.blocks.0.vel_gain",
                    "processor.module.predictor.blocks.0.vel_proj.weight"],
          f"{n_added:,} params")
    check("nothing removed", not removed, f"{removed}")
    tot21 = sum(p.numel() for p in m21.parameters())
    tot23 = sum(p.numel() for p in m23.parameters())
    print(f"  total params  v21 {tot21:,}   v23 {tot23:,}   (+{tot23 - tot21:,}, "
          f"+{100 * (tot23 - tot21) / tot21:.4f}%)")
    pred21 = sum(p.numel() for p in m21.processor.module.predictor.parameters())
    pred23 = sum(p.numel() for p in m23.processor.module.predictor.parameters())
    print(f"  predictor     v21 {pred21:,}   v23 {pred23:,}   "
          f"(+{100 * (pred23 - pred21) / pred21:.2f}% of the predictor)")

    print("\n2. velocity path is enabled on v23 only")
    check("v23 predictor takes velocity", m23.processor.module.predictor_takes_vel)
    check("v21 predictor does not", not m21.processor.module.predictor_takes_vel)

    HW = int(cfg23.dataset.val_pipeline.transforms.input_size)
    batch = {"video": torch.randn(2, 4, 3, HW, HW).cuda()}

    print("\n3. forward + compute_loss")
    out21, tot_l21, losses21 = run_losses(m21, batch)
    out23, tot_l23, losses23 = run_losses(m23, batch)
    print("  v21 losses: " + ", ".join(f"{k}={v.item():.5f}" for k, v in losses21.items()))
    print("  v23 losses: " + ", ".join(f"{k}={v.item():.5f}" for k, v in losses23.items()))
    check("loss_dyn present in v23", "loss_dyn" in losses23)
    check("loss_dyn absent in v21", "loss_dyn" not in losses21)
    check("v23 total is finite", bool(torch.isfinite(tot_l23)))
    if "loss_dyn" in losses23:
        val = losses23["loss_dyn"].item()
        check("loss_dyn lies in [0, 2]", 0.0 <= val <= 2.0, f"{val:.5f}")
    check("state_predicted_pregate is in the output tree",
          "state_predicted_pregate" in out23["processor"])

    print("\n4. backward reaches the new parameters through the whole model")
    m23.zero_grad()
    tot_l23.backward()
    blk = m23.processor.module.predictor.blocks[0]
    for name, p in (("vel_proj.weight", blk.vel_proj.weight), ("vel_gain", blk.vel_gain)):
        g = p.grad
        check(f"{name} gradient is nonzero and finite",
              g is not None and torch.isfinite(g).all() and g.abs().max().item() > 0,
              f"max |grad| {0.0 if g is None else g.abs().max().item():.3e}")
    # scale check: velocity must perturb the keys, not overwhelm them
    with torch.no_grad():
        ratio = (blk.vel_gain.abs().mean() * blk.vel_proj.weight.abs().mean() * (64 ** 0.5)).item()
    print(f"  rough |vel_gain * vel_proj| scale vs unit-RMS keys: {ratio:.4f} "
          f"(log ||vel_gain*vel_proj(v)|| / ||norm1(x)|| during training)")

    print("\n5. a v21 checkpoint loads into v23")
    missing, unexpected = m23.load_state_dict(m21.state_dict(), strict=False)
    check("only the velocity parameters are missing",
          sorted(missing) == ["processor.module.predictor.blocks.0.vel_gain",
                              "processor.module.predictor.blocks.0.vel_proj.weight"],
          f"{sorted(missing)}")
    check("nothing unexpected", not unexpected, f"{sorted(unexpected)}")

    print("\n6. the baseline config still runs (shared code paths)")
    del m21, m23
    torch.cuda.empty_cache()
    cfgb, mb = build(BASE)
    HWb = int(cfgb.dataset.val_pipeline.transforms.input_size)
    bb = {"video": torch.randn(2, 4, 3, HWb, HWb).cuda()}
    _, tot_lb, losses_b = run_losses(mb, bb)
    print("  baseline losses: " + ", ".join(f"{k}={v.item():.5f}" for k, v in losses_b.items()))
    check("baseline total is finite", bool(torch.isfinite(tot_lb)))
    check("baseline has no loss_dyn", "loss_dyn" not in losses_b)
    check("baseline predictor takes no velocity", not mb.processor.module.predictor_takes_vel)
    # the backward sweep also has to survive the prev_state threading
    mb.eval()
    with torch.no_grad():
        out_cyc = mb.forward(bb, train=False, cycle=True)
    check("cyclic inference still runs",
          bool(torch.isfinite(out_cyc["processor"]["state"]).all()))

    print("\n7. aux_forward / mask resizing tolerates the extra output key")
    with torch.no_grad():
        aux = mb.aux_forward(bb, out_cyc)
    check("aux_forward produces hard decoder masks",
          "decoder_masks_hard" in aux and torch.isfinite(aux["decoder_masks_hard"]).all(),
          f"shape {tuple(aux['decoder_masks_hard'].shape)}")

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
