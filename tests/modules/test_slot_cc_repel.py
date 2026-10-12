"""Same-frame candidate-candidate repulsion (Slot_Candidate_Repel_Loss)."""

import math

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


def _attn_loss():
    return Slot_Candidate_Repel_Loss(
        pred_key="processor.state",
        target_key="processor.state",
        keep_input_dim=True,
        patch_inputs=False,
        mode="attn",
    )


def test_attn_all_live_is_zero():
    att = torch.softmax(torch.randn(2, 3, 4, 6), dim=2)
    occ = torch.ones(2, 3, 4)
    val = float(_attn_loss()(None, None, occupancy=occ, attention=att))
    assert val < 1e-6


def test_attn_one_candidate_is_zero_and_split_is_not():
    # live + two candidates. Exclusive owner vs a 50/50 split.
    exclusive = torch.zeros(1, 1, 3, 2)
    exclusive[0, 0, 1, :] = 1.0
    split = torch.zeros(1, 1, 3, 2)
    split[0, 0, 1, :] = 0.5
    split[0, 0, 2, :] = 0.5
    occ = torch.tensor([[[1.0, 0.0, 0.0]]])
    fn = _attn_loss()
    assert float(fn(None, None, occupancy=occ, attention=exclusive)) < 1e-6
    # Both directions of one identical pair: log1p(2).
    assert abs(float(fn(None, None, occupancy=occ, attention=split)) - math.log1p(2.0)) < 1e-5


def test_attn_disjoint_candidates_are_zero():
    att = torch.zeros(1, 1, 3, 2)
    att[0, 0, 1, 0] = 1.0
    att[0, 0, 2, 1] = 1.0
    occ = torch.tensor([[[1.0, 0.0, 0.0]]])
    val = float(_attn_loss()(None, None, occupancy=occ, attention=att))
    assert val < 1e-6


def test_attn_tiny_pi_max_scales_down():
    split = torch.zeros(1, 1, 3, 2)
    split[0, 0, 1, :] = 0.5
    split[0, 0, 2, :] = 0.5
    fn = _attn_loss()
    full = float(fn(None, None, occupancy=torch.tensor([[[1.0, 0.0, 0.0]]]), attention=split))
    tiny = float(fn(None, None, occupancy=torch.tensor([[[0.05, 0.0, 0.0]]]), attention=split))
    assert abs(full - math.log1p(2.0)) < 1e-5
    assert abs(tiny - math.log1p(2.0 * (0.05 ** 2))) < 1e-6


def test_attn_scale_and_flat_leftover_stay_on():
    """Shrinking or flattening candidate maps does not turn overlap off."""
    split = torch.zeros(1, 1, 3, 2)
    split[0, 0, 1, :] = 0.5
    split[0, 0, 2, :] = 0.5
    occ = torch.tensor([[[1.0, 0.0, 0.0]]])
    fn = _attn_loss()
    full = float(fn(None, None, occupancy=occ, attention=split))
    shrunk = float(fn(None, None, occupancy=occ, attention=split * 1e-4))
    assert abs(full - shrunk) < 1e-5
    dust = torch.zeros(1, 1, 3, 4)
    dust[0, 0, 0, :] = 1.0
    dust[0, 0, 1, :] = 1e-4
    dust[0, 0, 2, :] = 1e-4
    flat = float(fn(None, None, occupancy=occ, attention=dust))
    assert abs(flat - math.log1p(2.0)) < 1e-4


def test_attn_all_zero_occupancy_is_off():
    att = torch.softmax(torch.randn(1, 2, 4, 5), dim=2)
    occ = torch.zeros(1, 2, 4)
    val = float(_attn_loss()(None, None, occupancy=occ, attention=att))
    assert val < 1e-6


def test_attn_grad_reaches_attention_not_occupancy():
    att = torch.softmax(torch.randn(1, 1, 3, 4), dim=2).detach().requires_grad_(True)
    occ = torch.tensor([[[1.0, 0.2, 0.2]]], requires_grad=True)
    loss = _attn_loss()(None, None, occupancy=occ, attention=att)
    loss.backward()
    assert att.grad is not None
    assert float(att.grad.abs().sum()) > 0
    assert occ.grad is None


def test_lcc_attn_configs():
    from slotcurri import configuration

    parent = configuration.load_config(
        "configs/slotcurri/movi_c_attnmass_v39lam1gu_s18_lcc.yaml"
    )
    assert parent.model.losses["loss_cc"].get("mode", "cosine") == "cosine"
    movi = configuration.load_config(
        "configs/slotcurri/movi_c_attnmass_v39lam1gu_s18_lcc_attn.yaml"
    )
    ytvis = configuration.load_config(
        "configs/slotcurri/ytvis2021_attnmass_v39lam1gu_lcc_attn.yaml"
    )
    assert movi.experiment_name == "movi_c_attnmass_v39lam1gu_s18_lcc_attn"
    assert ytvis.experiment_name == "ytvis_attnmass_v39lam1gu_lcc_attn"
    assert int(movi.globals["NUM_SLOTS"]) == 18
    assert int(ytvis.globals["NUM_SLOTS"]) == 7
    for cfg in (movi, ytvis):
        cc = cfg.model.losses["loss_cc"]
        assert cc["name"] == "Slot_Candidate_Repel_Loss"
        assert cc["mode"] == "attn"
        assert abs(float(cfg.model.loss_weights["loss_cc"]) - 0.1) < 1e-9
        assert bool(cfg.model.attn_mass_curriculum["contrastive_gate"]) is False
        assert bool(cfg.model.attn_mass_curriculum.get("predictor_input_mix", False)) is False


def test_no_occupancy_treats_all_as_candidates():
    torch.manual_seed(3)
    d = 8
    u = torch.nn.functional.normalize(torch.randn(d), dim=0)
    slots = torch.stack([u, u], dim=0).view(1, 1, 2, d)
    val = float(_loss()(slots, None))
    assert val > 1.0
