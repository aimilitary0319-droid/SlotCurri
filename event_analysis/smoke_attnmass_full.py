"""Full-model integration smoke test for the attention-mass curriculum.

Builds the real model from configs/slotcurri/movi_e_attnmass.yaml and runs a
validation-path forward (train=False, cycle=True) + compute_loss on a synthetic
batch. No trainer / dataset needed.
"""
import torch

from slotcurri import configuration, models

CFG = "configs/slotcurri/movi_e_attnmass.yaml"

config = configuration.load_config(CFG)
print("initializer:", config.model.initializer.name,
      "n_slots:", config.model.initializer.n_slots,
      "dim:", config.model.initializer.dim)
print("attn_mass_curriculum:", dict(config.model.attn_mass_curriculum))

model = models.build(config.model, config.optimizer, None, None)
model = model.cuda().eval()
print("model.n_slots:", model.n_slots, "attn_mass_enabled:", model.attn_mass_enabled)
print("initializer type:", type(model.initializer).__name__,
      "trainable params:", sum(p.numel() for p in model.initializer.parameters()))

B, T, HW = 2, 4, 336
batch = {"video": torch.randn(B, T, 3, HW, HW).cuda()}

with torch.no_grad():
    out = model.forward(batch, train=False, cycle=True)

state = out["processor"]["state"]
active = out["processor"]["active_mask"]
dec_masks = out["decoder"]["masks"]
recon = out["decoder"]["reconstruction"]
print("\n-- shapes --")
print("processor.state:", tuple(state.shape))
print("active_mask:", tuple(active.shape), active.dtype)
print("decoder.masks:", tuple(dec_masks.shape))
print("decoder.reconstruction:", tuple(recon.shape))

print("\n-- gating stats (mode=%s, p_end=%.4f) --" % (model.amc_gate_mode, model.amc_p_end))
print("mean (effective) active slots / frame:", active.float().sum(-1).mean().item())
# default slots: bool mask -> all True; float gate -> all ~1.0
default_gate = active[:, :, model.amc_default_idx].float()
print("default slots always active:", bool(torch.allclose(default_gate, torch.ones_like(default_gate))))
if active.dtype == torch.bool:
    mass = dec_masks.sum(-1)  # (B,T,S)
    print("max decoder mask mass on non-active slots:", mass[~active].abs().max().item()
          if (~active).any() else "n/a (all active)")
else:
    print("soft gate min/max:", active.min().item(), active.max().item())

total, losses = model.compute_loss(out)
print("\n-- losses (eval mode: no gate sparsity) --")
for k, v in losses.items():
    print(f"  {k}: {v.item():.5f}")
print(f"  total: {total.item():.5f}")
assert torch.isfinite(total), "total loss must be finite"
assert "loss_gate_sparsity" not in losses, "gate sparsity must be training-only (skipped in eval)"

# --- L1 gate sparsity penalty (training-mode) ---
print("\n-- gate L1 sparsity (mode=%s, gate_l1=%s) --" % (model.amc_gate_mode, model.amc_gate_l1))
if model.amc_gate_l1 > 0.0 and model.amc_gate_mode in ("soft", "ste"):
    model.train()
    # eval-path gate (train=False -> no trainer needed) but grad-enabled this time
    out2 = model.forward(batch, train=False, cycle=True)
    total2, losses2 = model.compute_loss(out2)
    assert "loss_gate_sparsity" in losses2, "gate sparsity term must appear in training mode"
    gp = losses2["loss_gate_sparsity"]
    print("  loss_gate_sparsity (raw):", gp.item(),
          "-> weighted:", (model.amc_gate_l1 * gp).item())
    assert torch.isfinite(total2), "training total loss must be finite"
    assert 0.0 <= gp.item() <= 1.0, "mean gate must be in [0, 1]"
    # penalty alone must backprop into the processor (i.e. gate -> mass -> slot attention)
    model.zero_grad()
    (model.amc_gate_l1 * gp).backward()
    proc_grad = sum(
        p.grad.abs().sum().item() for p in model.processor.parameters() if p.grad is not None
    )
    print("  gate-sparsity grad into processor (abs sum):", proc_grad)
    assert proc_grad > 0.0, "gate sparsity should backprop into the processor (differentiable gate)"
    model.zero_grad()
    model.eval()
    print("  gate L1 sparsity OK")
else:
    print("  gate_l1 disabled or non-differentiable mode; skipped")

print("\nFULL INTEGRATION SMOKE PASSED")
