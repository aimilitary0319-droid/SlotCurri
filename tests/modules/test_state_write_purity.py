"""Temporal-mix write purity: a_s = sg(r_s) * π̄_s. Decoder stays on u^SA."""

import torch

from slotcurri.modules.video import (
    LatentProcessor,
    _attn_write_purity,
    _max_norm,
    spectral_graph_n8_slot_purity,
)


class StubCorrector(torch.nn.Module):
    def __init__(self, masks):
        super().__init__()
        self.masks_out = masks

    def forward(self, state, inputs, n_iters=None, **kwargs):
        return {"slots": state + 1.0, "masks": self.masks_out}


def _gate_kwargs(**overrides):
    kwargs = dict(
        gate_p=None,
        default_idx=[],
        gate_form="purity_weight",
        conf_kind="spectral_graph_n8",
        state_max_norm=True,
        bind_features=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_uniform_attention_purity_is_one_over_slots():
    s, f = 4, 8
    att = torch.full((2, s, f), 1.0 / s)
    r = _attn_write_purity(att)
    assert torch.allclose(r, torch.full((2, s), 1.0 / s), atol=1e-6)


def test_exclusive_attention_purity_is_one_and_empty_is_zero():
    att = torch.zeros(1, 3, 4)
    att[0, 0, :2] = 1.0
    att[0, 1, 2:] = 1.0
    r = _attn_write_purity(att)
    assert torch.allclose(r, torch.tensor([[1.0, 1.0, 0.0]]), atol=1e-6)


def test_write_purity_is_detached():
    att = torch.softmax(torch.randn(2, 3, 8), dim=1).requires_grad_(True)
    r = _attn_write_purity(att)
    assert r.requires_grad is False
    assert r.grad_fn is None


def test_write_purity_scales_temporal_mix_only():
    torch.manual_seed(0)
    att = torch.softmax(torch.zeros(2, 4, 16), dim=1)
    feat = torch.randn(2, 16, 5)
    prior = torch.randn(2, 4, 8)
    inputs = torch.randn(2, 16, 8)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    kwargs = _gate_kwargs(bind_features=feat)
    out_off = processor(prior, inputs, state_write_purity=False, **kwargs)
    out_on = processor(prior, inputs, state_write_purity=True, **kwargs)
    pi = spectral_graph_n8_slot_purity(att, feat)
    pi_bar = _max_norm(pi, True)
    r = _attn_write_purity(att)
    u = prior + 1.0
    a = (pi_bar * r).unsqueeze(-1)
    expected = a * u + (1.0 - a) * prior
    plain = pi_bar.unsqueeze(-1) * u + (1.0 - pi_bar.unsqueeze(-1)) * prior
    assert torch.allclose(out_on["state"], out_off["state"], atol=1e-6)
    assert torch.allclose(out_on["active_mask"], out_off["active_mask"], atol=1e-5)
    assert torch.allclose(out_on["active_mask"], pi, atol=1e-5)
    assert torch.allclose(out_off["state_predicted"], plain, atol=1e-5)
    assert torch.allclose(out_on["state_predicted"], expected, atol=1e-5)
    assert not torch.allclose(out_on["state_predicted"], out_off["state_predicted"])


def test_write_purity_configs():
    from slotcurri import configuration

    parent_m = configuration.load_config(
        "configs/slotcurri/movi_c_attnmass_v39lam1gu.yaml"
    )
    parent_y = configuration.load_config(
        "configs/slotcurri/ytvis2021_attnmass_v39lam1gu.yaml"
    )
    assert bool(parent_m.model.attn_mass_curriculum.get("state_write_purity", False)) is False
    assert bool(parent_y.model.attn_mass_curriculum.get("state_write_purity", False)) is False

    movi = configuration.load_config(
        "configs/slotcurri/movi_c_attnmass_v39lam1gu_s18_write_purity.yaml"
    )
    ytvis = configuration.load_config(
        "configs/slotcurri/ytvis2021_attnmass_v39lam1gu_write_purity.yaml"
    )
    assert movi.experiment_name == "movi_c_attnmass_v39lam1gu_s18_write_purity"
    assert ytvis.experiment_name == "ytvis_attnmass_v39lam1gu_write_purity"
    assert int(movi.globals["NUM_SLOTS"]) == 18
    assert int(ytvis.globals["NUM_SLOTS"]) == 7
    for cfg in (movi, ytvis):
        amc = cfg.model.attn_mass_curriculum
        assert bool(amc["state_write_purity"]) is True
        assert bool(amc["state_max_norm"]) is True
        assert bool(amc["contrastive_gate"]) is False
        assert bool(amc.get("predictor_input_mix", False)) is False
        assert "loss_cc" not in cfg.model.losses
