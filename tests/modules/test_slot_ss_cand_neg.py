"""Candidate-only same-frame extras on Slot_Slot_Contrastive_Loss."""

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


def test_cand_neg_off_matches_baseline():
    torch.manual_seed(0)
    slots = torch.randn(2, 3, 4, 8)
    occ = torch.tensor(
        [
            [[1.0, 0.1, 0.1, 0.1]] * 3,
            [[1.0, 0.2, 0.2, 0.0]] * 3,
        ]
    )
    base = _loss(cand_neg=False, batch_contrast=False)
    on = _loss(cand_neg=True, batch_contrast=False)
    a = base(slots, None)
    b = on(slots, None, occupancy=occ)
    # occupancy is ignored when cand_neg is False; with cand_neg the extras
    # fire, so this just checks the baseline path is unchanged by the flag
    # when occupancy is omitted.
    c = on(slots, None)
    assert torch.allclose(a, c)


def test_uniform_occupancy_is_original_ce():
    torch.manual_seed(1)
    slots = torch.randn(2, 4, 5, 8)
    occ = torch.ones(2, 4, 5)
    base = _loss(cand_neg=False, batch_contrast=True)
    on = _loss(cand_neg=True, batch_contrast=True)
    a = base(slots, None)
    b = on(slots, None, occupancy=occ)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_duplicate_candidates_raise_loss():
    torch.manual_seed(2)
    t, d = 3, 16
    live = torch.nn.functional.normalize(torch.randn(d), dim=0)
    cand = torch.nn.functional.normalize(torch.randn(d), dim=0)
    other = torch.nn.functional.normalize(torch.randn(d), dim=0)
    slots = torch.stack([live, cand, cand, other], dim=0)
    slots = slots.view(1, 1, 4, d).expand(1, t, 4, d).contiguous()
    occ = torch.tensor([[[1.0, 0.05, 0.05, 0.05]] * t])
    base = _loss(cand_neg=False, batch_contrast=False, temperature=0.1)
    on = _loss(cand_neg=True, batch_contrast=False, temperature=0.1)
    l0 = base(slots, None)
    l_dup = on(slots, None, occupancy=occ)
    assert float(l_dup) > float(l0) + 1e-4


def test_candidate_matching_live_does_not_add_extra_classes_on_live():
    """Live column extras are -inf, so a candidate cloned from live is not an extra class for that live."""
    torch.manual_seed(3)
    t, d = 3, 16
    live = torch.nn.functional.normalize(torch.randn(d), dim=0)
    a = torch.nn.functional.normalize(torch.randn(d), dim=0)
    b = torch.nn.functional.normalize(torch.randn(d), dim=0)
    slots = torch.stack([live, live, a, b], dim=0)
    slots = slots.view(1, 1, 4, d).expand(1, t, 4, d).contiguous()
    occ = torch.tensor([[[1.0, 0.05, 0.05, 0.05]] * t])
    base = _loss(cand_neg=False, batch_contrast=False, temperature=0.1)
    on = _loss(cand_neg=True, batch_contrast=False, temperature=0.1)
    # a and b are distinct candidates; live-clone is slot 1. Extra classes only among {1,2,3}.
    # Pair 2-3 is not identical so the extra denom is small; still >= baseline.
    l0 = float(base(slots, None))
    l1 = float(on(slots, None, occupancy=occ))
    assert l1 + 1e-5 >= l0
    # Duplicated *candidates* (not live clones) move the loss more.
    slots_dup = torch.stack([live, a, a, b], dim=0)
    slots_dup = slots_dup.view(1, 1, 4, d).expand(1, t, 4, d).contiguous()
    l_dup = float(on(slots_dup, None, occupancy=occ))
    l_dup_base = float(base(slots_dup, None))
    assert (l_dup - l_dup_base) > (l1 - l0) + 1e-4

