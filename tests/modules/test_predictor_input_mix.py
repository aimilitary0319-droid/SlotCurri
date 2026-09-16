"""v39lam1gu_umix: Pred / next prior on occupancy-mixed u_t; decoder stays u^SA."""

import torch
from torch import nn

from slotcurri.modules.video import (
    LatentProcessor,
    _max_norm,
    spectral_graph_n8_slot_purity,
)


class StubCorrector(nn.Module):
    def __init__(self, masks):
        super().__init__()
        self.masks_out = masks

    def forward(self, state, inputs, n_iters=None, **kwargs):
        return {"slots": state + 1.0, "masks": self.masks_out}


class ShiftPred(nn.Module):
    def forward(self, x, key_weights=None, **kwargs):
        return x + 0.25


class ScalePred(nn.Module):
    """Input-dependent residual so umix ≠ default mix. Pred(x)=x+c is algebraically
    the same for both: û' = π̄ u^SA + (1-π̄) û + π̄ c."""

    def forward(self, x, key_weights=None, **kwargs):
        return 0.5 * x


def _gate_kwargs(**overrides):
    kwargs = dict(
        gate_p=None,
        default_idx=[],
        gate_form="purity_weight",
        conf_kind="spectral_graph_n8_lam1",
        state_max_norm=True,
        bind_features=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_decoder_state_stays_usa():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    feat = torch.randn(2, 16, 5)
    prior = torch.randn(2, 3, 8)
    inputs = torch.randn(2, 16, 8)
    processor = LatentProcessor(StubCorrector(att), predictor=ShiftPred())
    kwargs = _gate_kwargs(bind_features=feat)
    out_off = processor(prior, inputs, **kwargs)
    out_on = processor(prior, inputs, predictor_input_mix=True, **kwargs)
    assert torch.allclose(out_on["state"], out_off["state"], atol=1e-6)
    assert torch.allclose(out_on["state"], prior + 1.0, atol=1e-6)
    pi = spectral_graph_n8_slot_purity(att, feat, use_lambda1=True)
    assert torch.allclose(out_on["active_mask"], pi, atol=1e-5)


def test_next_prior_holds_mixed_state():
    torch.manual_seed(1)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    feat = torch.randn(2, 16, 5)
    prior = torch.randn(2, 3, 8)
    inputs = torch.randn(2, 16, 8)
    processor = LatentProcessor(StubCorrector(att), predictor=ShiftPred())
    out = processor(
        prior, inputs, predictor_input_mix=True, **_gate_kwargs(bind_features=feat)
    )
    pi_bar = _max_norm(
        spectral_graph_n8_slot_purity(att, feat, use_lambda1=True), True
    ).unsqueeze(-1)
    u_sa = prior + 1.0
    mixed = pi_bar * u_sa + (1.0 - pi_bar) * prior
    pred = mixed + 0.25
    expected = pi_bar * pred + (1.0 - pi_bar) * mixed
    assert torch.allclose(out["state_predicted"], expected, atol=1e-5)
    assert torch.allclose(out["state_predicted_pregate"], pred, atol=1e-5)


def test_default_mix_is_pred_usa_hold_prior():
    torch.manual_seed(2)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    feat = torch.randn(2, 16, 5)
    prior = torch.randn(2, 3, 8)
    inputs = torch.randn(2, 16, 8)
    processor = LatentProcessor(StubCorrector(att), predictor=ShiftPred())
    out = processor(prior, inputs, **_gate_kwargs(bind_features=feat))
    pi_bar = _max_norm(
        spectral_graph_n8_slot_purity(att, feat, use_lambda1=True), True
    ).unsqueeze(-1)
    u_sa = prior + 1.0
    pred = u_sa + 0.25
    expected = pi_bar * pred + (1.0 - pi_bar) * prior
    assert torch.allclose(out["state_predicted"], expected, atol=1e-5)
    assert not torch.allclose(out["state_predicted"], pred, atol=1e-3)


def test_umix_differs_from_default_when_pred_depends_on_input():
    torch.manual_seed(3)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    feat = torch.randn(2, 16, 5)
    prior = torch.randn(2, 3, 8)
    inputs = torch.randn(2, 16, 8)
    processor = LatentProcessor(StubCorrector(att), predictor=ScalePred())
    kwargs = _gate_kwargs(bind_features=feat)
    off = processor(prior, inputs, **kwargs)
    on = processor(prior, inputs, predictor_input_mix=True, **kwargs)
    assert not torch.allclose(off["state_predicted"], on["state_predicted"], atol=1e-4)
    pi_bar = _max_norm(
        spectral_graph_n8_slot_purity(att, feat, use_lambda1=True), True
    ).unsqueeze(-1)
    mixed = pi_bar * (prior + 1.0) + (1.0 - pi_bar) * prior
    expected = pi_bar * (0.5 * mixed) + (1.0 - pi_bar) * mixed
    assert torch.allclose(on["state_predicted"], expected, atol=1e-5)
