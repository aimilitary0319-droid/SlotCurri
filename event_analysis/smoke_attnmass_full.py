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

print("\n-- gating stats (p_end = %.4f) --" % model.amc_p_end)
print("mean active slots / frame:", active.float().sum(-1).mean().item())
print("default slot always active:", bool(active[:, :, model.amc_default_idx].all()))
mass = dec_masks.sum(-1)  # (B,T,S)
print("max decoder mask mass on non-active slots:", mass[~active].abs().max().item()
      if (~active).any() else "n/a (all active)")

total, losses = model.compute_loss(out)
print("\n-- losses --")
for k, v in losses.items():
    print(f"  {k}: {v.item():.5f}")
print(f"  total: {total.item():.5f}")
assert torch.isfinite(total), "total loss must be finite"
print("\nFULL INTEGRATION SMOKE PASSED")
