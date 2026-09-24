"""Same-frame candidate-candidate repulsion (Slot_Candidate_Repel_Loss)."""

import torch

from slotcurri.losses import Slot_Candidate_Repel_Loss


def _loss():
    return Slot_Candidate_Repel_Loss(
        pred_key="processor.state",
        target_key="processor.state",
        keep_input_dim=True,
        patch_inputs=False,
    )


def test_all_live_is_zero():
    torch.manual_seed(0)
    slots = torch.nn.functional.normalize(torch.randn(2, 3, 4, 8), dim=-1)
    occ = torch.ones(2, 3, 4)
    val = float(_loss()(slots, None, occupancy=occ))
    assert val < 1e-6


def test_duplicate_candidates_raise_loss():
    torch.manual_seed(1)
    t, d = 3, 16
    live = torch.nn.functional.normalize(torch.randn(d), dim=0)
    cand = torch.nn.functional.normalize(torch.randn(d), dim=0)
    other = torch.nn.functional.normalize(torch.randn(d), dim=0)
    extra = torch.nn.functional.normalize(torch.randn(d), dim=0)
    occ = torch.tensor([[[1.0, 0.05, 0.05, 0.05]] * t])
    fn = _loss()
    slots_dup = torch.stack([live, cand, cand, extra], dim=0)
    slots_dup = slots_dup.view(1, 1, 4, d).expand(1, t, 4, d).contiguous()
    slots_sep = torch.stack([live, cand, other, extra], dim=0)
    slots_sep = slots_sep.view(1, 1, 4, d).expand(1, t, 4, d).contiguous()
    assert float(fn(slots_dup, None, occupancy=occ)) > float(
        fn(slots_sep, None, occupancy=occ)
    ) + 1e-4


def test_candidate_like_live_is_not_penalized():
    torch.manual_seed(2)
    d = 16
    live = torch.nn.functional.normalize(torch.randn(d), dim=0)
    occ = torch.tensor([[[1.0, 0.05]]])
    fn = _loss()
    slots = torch.stack([live, live], dim=0).view(1, 1, 2, d)
    # Only cand-live pair; w=0 so the clone is free.
    assert float(fn(slots, None, occupancy=occ)) < 1e-6


def test_more_duplicate_candidates_raise_loss():
    """n-way collapse is stricter than 2-way (not a pair-mean)."""
    torch.manual_seed(4)
    d = 16
    live = torch.nn.functional.normalize(torch.randn(d), dim=0)
    cand = torch.nn.functional.normalize(torch.randn(d), dim=0)
    extra = torch.nn.functional.normalize(torch.randn(d), dim=0)
    occ = torch.tensor([[[1.0, 0.05, 0.05, 0.05]]])
    fn = _loss()
    two = torch.stack([live, cand, cand, extra], dim=0).view(1, 1, 4, d)
    many = torch.stack([live, cand, cand, cand], dim=0).view(1, 1, 4, d)
    assert float(fn(many, None, occupancy=occ)) > float(fn(two, None, occupancy=occ)) + 1e-4


def test_tiny_max_occupancy_is_almost_off():
    """No real live → no candidates. Same relative π, small π_max, small L_cc."""
    torch.manual_seed(5)
    d = 16
    live = torch.nn.functional.normalize(torch.randn(d), dim=0)
    cand = torch.nn.functional.normalize(torch.randn(d), dim=0)
    extra = torch.nn.functional.normalize(torch.randn(d), dim=0)
    slots = torch.stack([live, cand, cand, extra], dim=0).view(1, 1, 4, d)
    fn = _loss()
    live_occ = torch.tensor([[[1.0, 0.05, 0.05, 0.05]]])
    tiny_occ = live_occ * 0.01
    live_l = float(fn(slots, None, occupancy=live_occ))
    tiny_l = float(fn(slots, None, occupancy=tiny_occ))
    assert live_l > 1e-3
    assert tiny_l < 0.05 * live_l


def test_no_occupancy_treats_all_as_candidates():
    torch.manual_seed(3)
    d = 8
    u = torch.nn.functional.normalize(torch.randn(d), dim=0)
    slots = torch.stack([u, u], dim=0).view(1, 1, 2, d)
    val = float(_loss()(slots, None))
    assert val > 1.0
