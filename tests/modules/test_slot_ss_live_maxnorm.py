"""L_ss contrastive_gate weights use π̄ = π / max_s(π)."""

import torch

from slotcurri.losses import Slot_Slot_Contrastive_Loss


def _loss(**kwargs):
    return Slot_Slot_Contrastive_Loss(
        pred_key="processor.state",
        target_key="processor.state",
        keep_input_dim=True,
        patch_inputs=False,
        **kwargs,
    )


def test_active_mask_is_max_normalized():
    torch.manual_seed(0)
    slots = torch.randn(2, 3, 4, 8)
    raw = torch.tensor(
        [
            [[0.20, 0.02, 0.02, 0.01]] * 3,
            [[0.40, 0.10, 0.00, 0.04]] * 3,
        ]
    )
    scaled = raw * 0.1
    bar = raw / raw.amax(dim=-1, keepdim=True)
    fn = _loss(batch_contrast=False)
    a = fn(slots, None, active_mask=raw)
    b = fn(slots, None, active_mask=scaled)
    c = fn(slots, None, active_mask=bar)
    assert torch.allclose(a, b)
    assert torch.allclose(a, c)


def test_max_norm_is_per_sample_before_batch_cat():
    torch.manual_seed(1)
    slots = torch.randn(2, 3, 3, 8)
    # Different per-sample maxima: batch_cat must not share one max.
    occ = torch.tensor(
        [
            [[1.00, 0.10, 0.10]] * 3,
            [[0.05, 0.05, 0.01]] * 3,
        ]
    )
    bar = occ / occ.amax(dim=-1, keepdim=True)
    fn = _loss(batch_contrast=True)
    assert torch.allclose(
        fn(slots, None, active_mask=occ),
        fn(slots, None, active_mask=bar),
    )
