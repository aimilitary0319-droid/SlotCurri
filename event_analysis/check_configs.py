"""Run a config end to end without touching the dataset, to catch errors before a launch.

For each config given (default: the three live attn-mass configs) this builds the real
model and exercises every path a training run takes, on synthetic tensors:

  1. config parses against the ModelConfig schema and the model builds
  2. the optimizer / LR schedule that trainer.max_steps drives can be constructed
  3. a training step: forward -> aux_forward -> compute_loss -> backward, checking the
     losses are finite, every trainable parameter that should move receives gradient,
     and the scalars training_step logs are all computable
  4. the p / tau schedules are monotone and land on the values the config asks for
  5. a validation step: forward(train=False) with the config's cyclic_inference, then
     aux_forward, compute_loss and a real val_metrics update/compute

Usage (inside the container, needs one GPU):
  python event_analysis/check_configs.py
  python event_analysis/check_configs.py configs/slotcurri/ytvis2021_attnmass_v20.yaml
"""

import sys
import traceback
from typing import Dict, List

import torch

from slotcurri import configuration, metrics, models

DEFAULT_CONFIGS = [
    "configs/slotcurri/ytvis2021_attnmass_v20.yaml",
    "configs/slotcurri/ytvis2021_attnmass_v21.yaml",
    "configs/slotcurri/ytvis2021_attnmass_v24.yaml",
]

B, T, N_TRUE = 2, 4, 3

failures: List[str] = []


def check(cfg: str, name: str, ok: bool, detail: str = "") -> bool:
    ok = bool(ok)
    if not ok:
        failures.append(f"{cfg}: {name}" + (f" ({detail})" if detail else ""))
    print(f"    [{'ok' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return ok


class _FakeTrainer:
    """Enough of a Trainer for the step-dependent schedules and self.log_dict."""

    def __init__(self, step: int = 0):
        self.global_step = step

    # log_dict() asks the trainer whether logging is on; keep it off.
    @property
    def logger_connector(self):
        return None


def fake_batch(hw: int, device) -> Dict[str, torch.Tensor]:
    """A batch shaped like the val pipeline's, with one-hot ground-truth masks."""
    seg = torch.zeros(B, T, N_TRUE, hw, hw, dtype=torch.bool, device=device)
    third = hw // 3
    seg[:, :, 0, :third] = True
    seg[:, :, 1, third : 2 * third] = True
    seg[:, :, 2, 2 * third :] = True
    return {"video": torch.randn(B, T, 3, hw, hw, device=device), "segmentations": seg}


def run(cfg_path: str) -> None:
    print(f"\n{'=' * 74}\n{cfg_path}\n{'=' * 74}")

    # same init and same fake batch for every config, so the loss values printed below are
    # comparable across configs instead of differing by whatever the RNG was up to
    torch.manual_seed(0)
    config = configuration.load_config(cfg_path)
    config.model.visualize = False
    # built the same way train.py builds them, so a bad metric config fails here too
    val_metrics = {name: metrics.build(c) for name, c in (config.val_metrics or {}).items()}
    model = models.build(config.model, config.optimizer, None, val_metrics or None)
    model = model.cuda()
    model._trainer = _FakeTrainer(0)
    amc = config.model.attn_mass_curriculum or {}
    anneal = int(amc.get("anneal_steps", 0) or 0)
    print(f"  p {amc.get('p_start_mult')} -> {amc.get('p_end_mult')} "
          f"{amc.get('p_anneal')} over {anneal:,} of {config.trainer.max_steps:,} steps, "
          f"cyclic_inference={config.model.cyclic_inference}")

    print("  1/5 build")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check(cfg_path, "model builds and has trainable parameters", n_params > 0, f"{n_params:,}")

    print("  2/5 optimizer")
    try:
        opt = model.configure_optimizers()
        check(cfg_path, "configure_optimizers succeeds", opt is not None, str(type(opt).__name__))
    except Exception as exc:  # noqa: BLE001 - the point is to report, not to handle
        check(cfg_path, "configure_optimizers succeeds", False, repr(exc))

    hw = int(config.dataset.val_pipeline.transforms.input_size)
    batch = fake_batch(hw, "cuda")

    print("  3/5 training step")
    model.train()
    out = model.forward(batch)
    aux = model.aux_forward(batch, out)
    total, losses = model.compute_loss(out)
    check(cfg_path, "losses are finite", bool(torch.isfinite(total)),
          ", ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))
    check(cfg_path, "aux_forward produces hard decoder masks",
          "decoder_masks_hard" in aux and bool(torch.isfinite(aux["decoder_masks_hard"]).all()))
    total.backward()
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    nonfinite = [n for n, p in model.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
    check(cfg_path, "every trainable parameter receives gradient", not no_grad,
          f"{len(no_grad)} without: {no_grad[:4]}")
    check(cfg_path, "no NaN/Inf gradients", not nonfinite, f"{nonfinite[:4]}")
    # the scalars training_step logs, computed here so a KeyError shows up now
    try:
        logged = {"active_slots": model._active_mask.float().sum(-1).mean().item(),
                  "gate_p": float(model._gate_threshold(True)),
                  "gate_tau": float(model._gate_tau(True))}
        state_p = model._state_gate_threshold(True)
        if state_p is not None:
            logged["gate_state_p"] = float(state_p)
            logged["gate_state_slots"] = model._state_gate_mean * model.n_slots
        check(cfg_path, "logged gate scalars are computable",
              all(v == v for v in logged.values()),
              ", ".join(f"{k}={v:.4f}" for k, v in logged.items()))
    except Exception as exc:  # noqa: BLE001
        check(cfg_path, "logged gate scalars are computable", False, repr(exc))
    model.zero_grad(set_to_none=True)

    print("  4/5 schedules")
    steps = sorted({0, anneal // 2, anneal, int(config.trainer.max_steps)})
    ps, taus = [], []
    for s in steps:
        model._trainer = _FakeTrainer(s)
        ps.append(float(model._gate_threshold(True)))
        taus.append(float(model._gate_tau(True)))
    print("      step " + "".join(f"{s:>10,}" for s in steps))
    print("      p    " + "".join(f"{v:>10.5f}" for v in ps))
    print("      tau  " + "".join(f"{v:>10.5f}" for v in taus))
    check(cfg_path, "p is positive and non-increasing",
          all(v > 0 for v in ps) and all(a >= b - 1e-12 for a, b in zip(ps, ps[1:])))
    check(cfg_path, "p ends at p_end_mult / n_slots",
          abs(ps[steps.index(anneal)] - float(amc["p_end_mult"]) / model.n_slots) < 1e-9,
          f"{ps[steps.index(anneal)]:.6f}")
    check(cfg_path, "p is held after anneal_steps", abs(ps[-1] - ps[steps.index(anneal)]) < 1e-12)
    check(cfg_path, "tau stays positive", all(v > 0 for v in taus))
    state_ps = []
    for s in steps:
        model._trainer = _FakeTrainer(s)
        state_ps.append(model._state_gate_threshold(True))
    if state_ps[0] is not None:
        print("      p_st " + "".join(f"{v:>10.5f}" for v in state_ps))
        check(cfg_path, "state threshold is a floor on the schedule, never below it",
              all(a >= b - 1e-12 for a, b in zip(state_ps, ps)),
              f"floor {float(amc['state_p_mult']) / model.n_slots:.5f}")

    print("  5/5 validation step")
    model._trainer = _FakeTrainer(anneal)
    model.eval()
    with torch.no_grad():
        vout = model.forward(batch, train=False, cycle=model.cyclic_inference)
        vaux = model.aux_forward(batch, vout)
        vtotal, _ = model.compute_loss(vout)
        check(cfg_path, "val forward + loss are finite", bool(torch.isfinite(vtotal)),
              f"loss={vtotal.item():.4f}")
        vals = {}
        for key, metric in (model.val_metrics or {}).items():
            metric.reset()
            metric.update(**batch, **vout, **vaux)
            vals[key] = float(metric.compute())
            metric.reset()
        check(cfg_path, "all val metrics update and compute",
              bool(vals) and all(v == v for v in vals.values()),
              ", ".join(f"{k}={v:.4f}" for k, v in vals.items()))

    del model
    torch.cuda.empty_cache()


def main() -> int:
    cfgs = sys.argv[1:] or DEFAULT_CONFIGS
    for cfg in cfgs:
        try:
            run(cfg)
        except Exception:  # noqa: BLE001 - a crash here is itself the finding
            failures.append(f"{cfg}: raised")
            traceback.print_exc()

    print("\n" + "=" * 74)
    if failures:
        print(f"FAILED {len(failures)}:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"all checks passed for {len(cfgs)} config(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
