"""Component-level smoke test for the attention-mass curriculum (no DINO backbone)."""
import torch
from torch import nn

from slotcurri.modules.groupers import SlotAttention
from slotcurri.modules.video import LatentProcessor, ScanOverTime, MapOverTime
from slotcurri.modules.decoders import MLPDecoder
from slotcurri.losses import Slot_Slot_Contrastive_Loss

torch.manual_seed(0)
B, T, S, D, F = 2, 4, 8, 16, 36


class AddConst(nn.Module):
    """Dummy predictor: shifts every slot so we can detect frozen (non-updated) slots."""
    def forward(self, x):
        return x + 1.0


corrector = SlotAttention(inp_dim=D, slot_dim=D, n_iters=2, use_mlp=True)
proc = LatentProcessor(corrector, predictor=AddConst(),
                       first_step_corrector_args={"n_iters": 3})
scan = ScanOverTime(proc)

slots0 = torch.randn(B, S, D)
feats = torch.randn(B, T, F, D)

# Use a high threshold so several slots end up non-active (default slots 0,1 always active).
out = scan(slots0, feats, cycle=False, gate_p=0.2, default_idx=[0, 1])
state = out["state"]            # (B, T, S, D)
state_pred = out["state_predicted"]
active = out["active_mask"]      # (B, T, S) bool
print("state", tuple(state.shape), "active", tuple(active.shape), active.dtype)
print("mean active slots / frame:", active.float().sum(-1).mean().item())
assert state.shape == (B, T, S, D)
assert active.shape == (B, T, S)
assert active[:, :, 0].all() and active[:, :, 1].all(), "default slots (0,1) must always be active"

# Freeze check: at frame 0, non-active slots must equal the incoming init state exactly.
inactive0 = ~active[:, 0]  # (B, S)
if inactive0.any():
    diff = (state[:, 0][inactive0] - slots0[inactive0]).abs().max().item()
    print("frame0 non-active |state - init| max:", diff)
    assert diff < 1e-6, "non-active slot at frame 0 should keep the init state"

# Freeze check across time: a slot non-active at frame t keeps the state it had at t-1.
for t in range(1, T):
    inact = ~active[:, t]  # (B, S)
    if inact.any():
        prev = state[:, t - 1][inact]
        cur = state[:, t][inact]
        d = (cur - prev).abs().max().item()
        # incoming state at t == state_predicted at t-1; for a slot non-active at t-1 too,
        # predicted == prev state, so cur should match prev (frozen carry).
        carried = state_pred[:, t - 1][inact]
        dc = (cur - carried).abs().max().item()
        print(f"frame{t} non-active: |state_t - carried_{t-1}| max = {dc:.2e}")
        assert dc < 1e-5, "non-active slot must carry the incoming (previous) state unchanged"

# Decoder masking: non-active slots must produce ~0 mask.
dec = MapOverTime(MLPDecoder(inp_dim=D, outp_dim=D, hidden_dims=[32], n_patches=F))
dec_out = dec(state, active)
masks = dec_out["masks"]  # (B, T, S, F)
recon = dec_out["reconstruction"]
print("decoder masks", tuple(masks.shape), "recon", tuple(recon.shape))
# mask mass per slot
mass = masks.sum(-1)  # (B, T, S)
inactive_mass = mass[~active].abs().max().item()
print("max mask mass on non-active slots:", inactive_mass)
assert inactive_mass < 1e-3, "non-active slots must not contribute to reconstruction"
# masks over slots should sum to ~1 per patch (only active slots share the mass)
persum = masks.sum(2)  # (B,T,F)
print("per-patch mask sum min/max:", persum.min().item(), persum.max().item())
assert torch.allclose(persum, torch.ones_like(persum), atol=1e-4)

# Contrastive loss: active-only vs full, and equivalence when all-active.
loss_fn = Slot_Slot_Contrastive_Loss(pred_key="processor.state", target_key="processor.state",
                                     temperature=0.1, batch_contrast=True,
                                     patch_inputs=False, keep_input_dim=True)
l_full = loss_fn(state, None)
l_active = loss_fn(state, None, active_mask=active)
all_active = torch.ones_like(active)
l_allA = loss_fn(state, None, active_mask=all_active)
print(f"loss_ss full={l_full.item():.4f}  active-only={l_active.item():.4f}  all-active(masked)={l_allA.item():.4f}")
assert torch.isfinite(l_active), "active-only contrastive loss must be finite"
assert abs(l_full.item() - l_allA.item()) < 1e-4, "masked loss with all-active must match full loss"

# ------------------------------------------------------------------
# SOFT GATING: sigmoid gate in [0,1], default slots -> 1, decoder renormalized, differentiable
# ------------------------------------------------------------------
out_s = scan(slots0, feats, cycle=False, gate_p=0.2, default_idx=[0, 1],
             gate_mode="soft", gate_tau=0.02)
g = out_s["active_mask"]  # (B, T, S) float
print("\n[soft] gate", tuple(g.shape), g.dtype, "min/max:", g.min().item(), g.max().item())
assert g.dtype.is_floating_point, "soft gate must be float"
assert (g >= 0).all() and (g <= 1.0 + 1e-5).all(), "soft gate must lie in [0,1]"
assert torch.allclose(g[:, :, 0], torch.ones_like(g[:, :, 0])) and \
       torch.allclose(g[:, :, 1], torch.ones_like(g[:, :, 1])), "default slots must have gate=1"

# soft decoder: per-patch masks still sum to 1 after gate + renormalization
dec_s = dec(out_s["state"], g)
masks_s = dec_s["masks"]  # (B, T, S, F)
persum_s = masks_s.sum(2)  # (B, T, F)
print("[soft] per-patch mask sum min/max:", persum_s.min().item(), persum_s.max().item())
assert torch.allclose(persum_s, torch.ones_like(persum_s), atol=1e-4)
# limiting behavior: a gate of exactly 0 removes the slot entirely (mask mass -> 0),
# while per-patch masses still renormalize to 1 over the remaining slots (hard-consistent).
g_ctrl = torch.ones(B, T, S)
g_ctrl[:, :, 3] = 0.0  # force slot 3 fully off
dec_ctrl = dec(out_s["state"], g_ctrl)
mask_ctrl = dec_ctrl["masks"]  # (B, T, S, F)
off_mass = mask_ctrl[:, :, 3].abs().max().item()
print("[soft] mask mass on gate==0 slot:", off_mass)
assert off_mass < 1e-6, "a gate of 0 must remove the slot from reconstruction"
assert torch.allclose(mask_ctrl.sum(2), torch.ones(B, T, F), atol=1e-4), \
    "per-patch masks must still sum to 1 after removing a slot"

# soft contrastive loss is finite
l_soft = loss_fn(out_s["state"], None, active_mask=g)
assert torch.isfinite(l_soft), "soft contrastive loss must be finite"

# key advantage over hard gating: the gate itself is differentiable (gradient reaches corrector)
corrector.zero_grad()
g.sum().backward()
gate_grad = sum(p.grad.abs().sum().item() for p in corrector.parameters() if p.grad is not None)
print(f"[soft] loss_ss={l_soft.item():.4f}  gate grad-norm into corrector={gate_grad:.3e}")
assert gate_grad > 0, "soft gate must be differentiable (gradient flows to corrector)"

# ------------------------------------------------------------------
# STE GATING: forward = hard 0/1 (true dormancy), backward = sigmoid surrogate (differentiable)
# ------------------------------------------------------------------
out_e = scan(slots0, feats, cycle=False, gate_p=0.2, default_idx=[0, 1],
             gate_mode="ste", gate_tau=0.02)
ge = out_e["active_mask"]  # (B, T, S) float, but exactly 0/1 in the forward pass
uniq = torch.unique(ge.detach()).tolist()
print("\n[ste] gate", tuple(ge.shape), ge.dtype, "unique values:", uniq)
assert torch.all((ge == 0) | (ge == 1)), "STE forward gate must be exactly 0/1 (true dormancy)"
assert ge[:, :, 0].all() and ge[:, :, 1].all(), "default slots must be active"

# decoder: a gate of 0 removes the slot; per-patch masks still sum to 1
dec_e = dec(out_e["state"], ge)
mask_e = dec_e["masks"]  # (B, T, S, F)
off_e = mask_e[ge == 0].abs().max().item() if (ge == 0).any() else 0.0
print("[ste] max mask mass on gate==0 slots:", off_e)
assert off_e < 1e-6, "STE dormant (gate=0) slots must not contribute to reconstruction"
assert torch.allclose(mask_e.sum(2), torch.ones(B, T, F), atol=1e-4)

# differentiability: gradient still reaches the corrector via the surrogate
corrector.zero_grad()
ge.sum().backward()
ste_grad = sum(p.grad.abs().sum().item() for p in corrector.parameters() if p.grad is not None)
print("[ste] gate grad-norm into corrector:", f"{ste_grad:.3e}")
assert ste_grad > 0, "STE gate must be differentiable (surrogate gradient flows to corrector)"

print("\nALL SMOKE CHECKS PASSED")
