"""Baseline (no attention-mass) integration smoke test.

Builds the real model from a baseline config (FixedLearnedInit, no
attn_mass_curriculum) and runs a forward + compute_loss on a synthetic batch to
verify the original SlotCurri path still works after the attn-mass additions.
No dataset needed; a tiny fake trainer supplies global_step for compute_loss.
"""
import sys

import torch

from slotcurri import configuration, models

CFG = sys.argv[1] if len(sys.argv) > 1 else "configs/slotcurri/movi_e.yaml"


class _FakeTrainer:
    global_step = 0


config = configuration.load_config(CFG)
print("config:", CFG)
print("initializer:", config.model.initializer.name, "n_slots:", config.model.initializer.n_slots)
print("attn_mass_curriculum in cfg:", config.model.get("attn_mass_curriculum"))

model = models.build(config.model, config.optimizer, None, None)
model._trainer = _FakeTrainer()
model = model.cuda().eval()

print("attn_mass_enabled:", model.attn_mass_enabled, "(baseline expects False)")
print("initializer type:", type(model.initializer).__name__,
      "trainable params:", sum(p.numel() for p in model.initializer.parameters()))
print("baseline curriculum -> hier_steps:", model.hier_steps, "hier_n_slots:", model.hier_n_slots)

try:
    HW = int(config.dataset.val_pipeline.transforms.input_size)
except Exception:
    HW = int(config.dataset.train_pipeline.transforms.input_size)
B, T = 2, 4
print("input size (from config):", HW)
batch = {"video": torch.randn(B, T, 3, HW, HW).cuda()}

with torch.no_grad():
    out = model.forward(batch, train=False, cycle=True)

state = out["processor"]["state"]
dec_masks = out["decoder"]["masks"]
recon = out["decoder"]["reconstruction"]
print("\n-- shapes --")
print("processor.state:", tuple(state.shape))
print("decoder.masks:", tuple(dec_masks.shape))
print("decoder.reconstruction:", tuple(recon.shape))

total, losses = model.compute_loss(out)
print("\n-- losses --")
for k, v in losses.items():
    print(f"  {k}: {v.item():.5f}")
print(f"  total: {total.item():.5f}")
assert torch.isfinite(total), "total loss must be finite"
print("\nBASELINE INTEGRATION SMOKE PASSED")
