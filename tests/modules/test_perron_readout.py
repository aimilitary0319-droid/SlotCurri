"""Eval-only decoder Perron readout: m = softmax(α) ⊙ π ⊙ max-norm(q1)."""
import torch

from slotcurri.modules.decoders import MLPDecoder
from slotcurri.modules.video import (
    LatentProcessor,
    _perron_spatial_gate,
    spectral_graph_n8_eigs,
)


class StubCorrector(torch.nn.Module):
    def __init__(self, masks):
        super().__init__()
        self.masks_out = masks

    def forward(self, state, inputs, n_iters=None, **kwargs):
        return {"slots": state + 1.0, "masks": self.masks_out}


def test_perron_spatial_gate_kills_weak_lobe_keeps_pi_scale():
    pi = torch.tensor([[0.9, 0.05]])
    q1 = torch.zeros(1, 2, 8)
    q1[0, 0, :4] = torch.tensor([4.0, 3.0, 0.2, 0.1])
    q1[0, 0, 4:] = 0.01
    q1[0, 1] = 1.0
    g = _perron_spatial_gate(pi, q1)
    assert g.shape == (1, 2, 8)
    assert torch.allclose(g[0, 0].amax(), pi[0, 0])
    assert g[0, 0, 0] > 5.0 * g[0, 0, 5]
    assert torch.allclose(g[0, 1], pi[0, 1] * torch.ones(8))


def test_perron_spatial_gate_flips_negative_mode():
    pi = torch.tensor([[0.8]])
    q1 = -torch.ones(1, 1, 4)
    q1[0, 0, 0] = -4.0
    g = _perron_spatial_gate(pi, q1)
    assert (g >= 0).all()
    assert torch.allclose(g[0, 0, 0], pi[0, 0])


def test_decoder_accepts_spatial_gate():
    dec = MLPDecoder(inp_dim=4, outp_dim=3, hidden_dims=[8], n_patches=6)
    slots = torch.randn(2, 3, 4)
    g = torch.rand(2, 3, 6)
    out = dec(slots, g)
    assert out["masks"].shape == (2, 3, 6)
    assert torch.allclose(out["masks"].sum(dim=1), torch.ones(2, 6), atol=1e-5)


def test_processor_perron_is_decoder_only():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    feat = torch.randn(2, 16, 5)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    kwargs = dict(
        gate_p=None,
        default_idx=[],
        gate_form="purity_weight",
        conf_kind="spectral_graph_n8_lam1",
        state_max_norm=True,
        bind_features=feat,
    )
    off = processor(torch.randn(2, 3, 8), torch.randn(2, 16, 8), **kwargs)
    on = processor(
        torch.randn(2, 3, 8),
        torch.randn(2, 16, 8),
        eval_perron_readout=True,
        **kwargs,
    )
    pi, _, _, q1 = spectral_graph_n8_eigs(
        att, feat, use_lambda1=True, return_q1=True
    )
    assert off["active_mask"].shape == (2, 3)
    assert "decoder_gate" not in off
    assert on["active_mask"].shape == (2, 3)
    assert on["decoder_gate"].shape == (2, 3, 16)
    assert on["state_gate"].shape == (2, 3)
    assert torch.allclose(on["active_mask"], off["active_mask"], atol=1e-5)
    assert torch.allclose(on["state_gate"], off["active_mask"], atol=1e-5)
    expected = _perron_spatial_gate(off["active_mask"], q1)
    assert torch.allclose(on["decoder_gate"], expected, atol=1e-5)
    assert torch.allclose(on["state_gate"], pi, atol=1e-4)


def test_pair_isolate_blocks_cross_and_keeps_same_group():
    from slotcurri.modules.video import _pair_isolate_weights

    g = torch.tensor([[1.0, 1.0, 0.0]])
    a = _pair_isolate_weights(g)
    assert a.shape == (1, 3, 3)
    # live-live open
    assert float(a[0, 0, 1]) == 1.0
    assert float(a[0, 1, 0]) == 1.0
    # live-ghost closed both ways
    assert float(a[0, 0, 2]) == 0.0
    assert float(a[0, 2, 0]) == 0.0
    assert float(a[0, 1, 2]) == 0.0
    assert float(a[0, 2, 1]) == 0.0
    # ghost self open
    assert float(a[0, 2, 2]) == 1.0


def test_pair_isolate_soft_mid_pi():
    from slotcurri.modules.video import _pair_isolate_weights

    g = torch.tensor([[1.0, 0.3]])
    a = _pair_isolate_weights(g)
    # max-norm leaves [1, 0.3]
    assert torch.allclose(a[0, 0, 0], torch.tensor(1.0))
    assert a[0, 0, 1] < a[0, 0, 0]
    assert a[0, 1, 0] == a[0, 0, 1]


def test_pi_gamma_identity_and_pow2():
    from slotcurri.modules.video import _pi_gamma

    g = torch.tensor([[1.0, 0.5]])
    assert _pi_gamma(g, 1.0) is g
    sharp = _pi_gamma(g, 2.0)
    assert torch.allclose(sharp, torch.tensor([[1.0, 0.25]]))
    assert torch.equal(_pi_gamma(torch.tensor([[True, False]]), 2.0), torch.tensor([[True, False]]))
