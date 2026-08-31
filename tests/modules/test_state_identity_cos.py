"""Unit tests for the v39i temporal identity cosine (ρ = π̄ ⊙ ReLU(cos(û, u)))."""

import torch

from slotcurri.modules.video import (
    LatentProcessor,
    _max_norm,
    _slot_identity_cos,
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


def test_slot_identity_cos_relu_and_range():
    prior = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
    post = torch.tensor([[[1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]])
    c = _slot_identity_cos(prior, post)
    assert c.shape == (1, 3)
    assert torch.allclose(c[0, 0], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(c[0, 1], torch.tensor(0.0), atol=1e-5)  # ReLU of -1
    assert torch.allclose(c[0, 2], torch.tensor(0.0), atol=1e-5)


def test_identity_cos_does_not_change_decoder_pi():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    feat = torch.randn(2, 16, 5)
    prior = torch.randn(2, 3, 8)
    inputs = torch.randn(2, 16, 8)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    kwargs = _gate_kwargs(bind_features=feat)
    out_off = processor(prior, inputs, state_identity_cos=False, **kwargs)
    out_on = processor(prior, inputs, state_identity_cos=True, **kwargs)
    pi = spectral_graph_n8_slot_purity(att, feat)
    assert torch.allclose(out_off["active_mask"], pi, atol=1e-5)
    assert torch.allclose(out_on["active_mask"], pi, atol=1e-5)
    assert torch.allclose(out_on["state"], out_off["state"], atol=1e-6)


def test_temporal_mix_is_pi_bar_times_identity_cos():
    torch.manual_seed(1)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    feat = torch.randn(2, 16, 5)
    prior = torch.randn(2, 3, 8)
    inputs = torch.randn(2, 16, 8)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    out = processor(
        prior, inputs, state_identity_cos=True, **_gate_kwargs(bind_features=feat)
    )
    pi = spectral_graph_n8_slot_purity(att, feat)
    pi_bar = _max_norm(pi, True)
    u = prior + 1.0
    c = _slot_identity_cos(prior, u)
    rho = pi_bar * c
    expected = rho.unsqueeze(-1) * u + (1.0 - rho.unsqueeze(-1)) * prior
    assert torch.allclose(out["identity_cos"], c, atol=1e-5)
    assert torch.allclose(out["state_predicted"], expected, atol=1e-5)
    assert not out["identity_cos"].requires_grad


def test_identity_cos_off_matches_v39_mix():
    torch.manual_seed(2)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    feat = torch.randn(2, 16, 5)
    prior = torch.randn(2, 3, 8)
    inputs = torch.randn(2, 16, 8)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    kwargs = _gate_kwargs(bind_features=feat)
    out_default = processor(prior, inputs, **kwargs)
    out_off = processor(prior, inputs, state_identity_cos=False, **kwargs)
    pi_bar = _max_norm(spectral_graph_n8_slot_purity(att, feat), True)
    u = prior + 1.0
    expected = pi_bar.unsqueeze(-1) * u + (1.0 - pi_bar.unsqueeze(-1)) * prior
    assert "identity_cos" not in out_default
    assert torch.allclose(out_default["state_predicted"], out_off["state_predicted"])
    assert torch.allclose(out_default["state_predicted"], expected, atol=1e-5)


def test_v39i_config_parses_identity_knob():
    from slotcurri import configuration

    v39 = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v39.yaml")
    assert bool(v39.model.attn_mass_curriculum.get("state_identity_cos", False)) is False

    v39i = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v39i.yaml")
    assert v39i.experiment_name == "ytvis_attnmass_v39i"
    amc = v39i.model.attn_mass_curriculum
    assert amc["gate_form"] == "purity_weight"
    assert amc["conf_kind"] == "spectral_graph_n8"
    assert bool(amc["state_max_norm"]) is True
    assert bool(amc["state_identity_cos"]) is True
    fc = v39i.model.feature_curriculum
    assert fc["anneal"] == "ncut"
    assert fc["apply"] == "key"
    assert bool(fc["barrier"]) is False


def test_v39si_config_parses_l1imp_and_identity():
    from slotcurri import configuration

    for path, name, n_slots in (
        ("configs/slotcurri/ytvis2021_attnmass_v39si.yaml", "ytvis_attnmass_v39si", 7),
        ("configs/slotcurri/movi_c_attnmass_v39si.yaml", "movi_c_attnmass_v39si", 11),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        amc = cfg.model.attn_mass_curriculum
        assert amc["conf_kind"] == "spectral_graph_n8_l1imp"
        assert bool(amc["state_identity_cos"]) is True
        assert bool(amc["state_max_norm"]) is True
        assert amc["gate_form"] == "purity_weight"
        assert int(cfg.globals["NUM_SLOTS"]) == n_slots


def test_l1imp_identity_mix_and_decoder_pi():
    torch.manual_seed(3)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    feat = torch.randn(2, 16, 5)
    prior = torch.randn(2, 3, 8)
    inputs = torch.randn(2, 16, 8)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    out = processor(
        prior,
        inputs,
        state_identity_cos=True,
        **_gate_kwargs(bind_features=feat, conf_kind="spectral_graph_n8_l1imp"),
    )
    pi = spectral_graph_n8_slot_purity(att, feat, l1_minus_imp=True)
    assert torch.allclose(out["active_mask"], pi, atol=1e-5)
    pi_bar = _max_norm(pi, True)
    u = prior + 1.0
    c = _slot_identity_cos(prior, u)
    rho = pi_bar * c
    expected = rho.unsqueeze(-1) * u + (1.0 - rho.unsqueeze(-1)) * prior
    assert torch.allclose(out["state_predicted"], expected, atol=1e-5)
