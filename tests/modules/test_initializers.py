"""Slot initialization."""

import torch
import torch.nn.functional as F

from slotcurri.modules.initializers import FixedLearnedInit


def _mean_cosine(slots: torch.Tensor) -> float:
    x = F.normalize(slots[0].float(), dim=-1)
    sim = x @ x.T
    idx = torch.triu_indices(x.shape[0], x.shape[0], offset=1)
    return float(sim[idx[0], idx[1]].mean())


def test_shared_init_is_fixed_and_cosine_near_one_half():
    cosines = []
    for seed in range(30):
        torch.manual_seed(seed)
        init = FixedLearnedInit(15, 128, shared_init=True)
        first = init(4)
        second = init(4)
        assert torch.equal(first, second)
        assert first.shape == (4, 15, 128)
        cosines.append(_mean_cosine(init.slots))
    mean = sum(cosines) / len(cosines)
    assert 0.35 < mean < 0.65


def test_identical_init_rows_are_equal():
    torch.manual_seed(0)
    init = FixedLearnedInit(15, 128, identical_init=True)
    slots = init.slots[0]
    assert torch.equal(slots, slots[:1].expand_as(slots))
    assert abs(_mean_cosine(init.slots) - 1.0) < 1e-6
    first = init(2)
    assert torch.equal(first, init(2))


def test_default_init_stays_near_orthogonal():
    torch.manual_seed(0)
    init = FixedLearnedInit(15, 128)
    assert abs(_mean_cosine(init.slots)) < 0.2
