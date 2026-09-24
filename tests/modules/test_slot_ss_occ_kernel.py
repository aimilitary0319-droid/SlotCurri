"""Occupancy-kernel softmax blocks on Slot_Slot_Contrastive_Loss."""

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


def test_occ_kernel_off_or_no_occupancy_matches_baseline():
    torch.manual_seed(0)
    slots = torch.randn(2, 3, 4, 8)
    occ = torch.tensor(
        [
            [[1.0, 0.1, 0.1, 0.1]] * 3,
            [[1.0, 0.2, 0.2, 0.0]] * 3,
        ]
    )
    base = _loss(occ_kernel=False, batch_contrast=False)
    on = _loss(occ_kernel=True, batch_contrast=False)
    a = base(slots, None)
    assert torch.allclose(a, on(slots, None))
    off = _loss(occ_kernel=False, batch_contrast=False)
    assert torch.allclose(a, off(slots, None, occupancy=occ))


def test_equal_occupancy_is_original_ce():
    torch.manual_seed(1)
    slots = torch.randn(2, 4, 5, 8)
    occ = torch.ones(2, 4, 5)
    base = _loss(occ_kernel=False, batch_contrast=True)
    on = _loss(occ_kernel=True, batch_contrast=True)
    a = base(slots, None)
    b = on(slots, None, occupancy=occ)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_candidate_like_live_is_cheaper_than_baseline():
    """Cross negatives drop out, so a candidate cloned from live is not a live-column competitor."""
    torch.manual_seed(3)
    t, d = 3, 16
    live = torch.nn.functional.normalize(torch.randn(d), dim=0)
    a = torch.nn.functional.normalize(torch.randn(d), dim=0)
    b = torch.nn.functional.normalize(torch.randn(d), dim=0)
    slots = torch.stack([live, live, a, b], dim=0)
    slots = slots.view(1, 1, 4, d).expand(1, t, 4, d).contiguous()
    occ = torch.tensor([[[1.0, 0.05, 0.05, 0.05]] * t])
    base = _loss(occ_kernel=False, batch_contrast=False, temperature=0.1)
    on = _loss(occ_kernel=True, batch_contrast=False, temperature=0.1)
    l0 = float(base(slots, None))
    l1 = float(on(slots, None, occupancy=occ))
    assert l1 < l0 - 1e-4


def test_duplicate_candidates_still_raise_loss():
    torch.manual_seed(2)
    t, d = 3, 16
    live = torch.nn.functional.normalize(torch.randn(d), dim=0)
    cand = torch.nn.functional.normalize(torch.randn(d), dim=0)
    other = torch.nn.functional.normalize(torch.randn(d), dim=0)
    extra = torch.nn.functional.normalize(torch.randn(d), dim=0)
    occ = torch.tensor([[[1.0, 0.05, 0.05, 0.05]] * t])
    on = _loss(occ_kernel=True, batch_contrast=False, temperature=0.1)

    slots_dup = torch.stack([live, cand, cand, extra], dim=0)
    slots_dup = slots_dup.view(1, 1, 4, d).expand(1, t, 4, d).contiguous()
    slots_sep = torch.stack([live, cand, other, extra], dim=0)
    slots_sep = slots_sep.view(1, 1, 4, d).expand(1, t, 4, d).contiguous()
    l_dup = float(on(slots_dup, None, occupancy=occ))
    l_sep = float(on(slots_sep, None, occupancy=occ))
    assert l_dup > l_sep + 1e-4
