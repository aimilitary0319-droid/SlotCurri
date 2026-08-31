"""Unit tests for the π hysteresis (leaky-max occlusion memory) on the gate."""

import torch

from slotcurri.modules.video import (
    LatentProcessor,
    ScanOverTime,
    spectral_graph_n8_slot_purity,
)


class StubCorrector(torch.nn.Module):
    def __init__(self, masks):
        super().__init__()
        self.masks_out = masks

    def forward(self, state, inputs, n_iters=None, **kwargs):
        return {"slots": state + 1.0, "masks": self.masks_out}


class SeqStubCorrector(torch.nn.Module):
    """Emits a different attention mask on every call (one per frame)."""

    def __init__(self, masks_per_step):
        super().__init__()
        self.masks_per_step = list(masks_per_step)
        self.calls = 0

    def forward(self, state, inputs, n_iters=None, **kwargs):
        m = self.masks_per_step[min(self.calls, len(self.masks_per_step) - 1)]
        self.calls += 1
        return {"slots": state + 1.0, "masks": m}


def _gate_kwargs(**overrides):
    kwargs = dict(
        gate_p=None,
        default_idx=[],
        gate_form="purity_weight",
        conf_kind="spectral_graph_n8",
        state_max_norm=True,
    )
    kwargs.update(overrides)
    return kwargs


def test_processor_hysteresis_is_leaky_max():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    feat = torch.randn(2, 16, 5)
    prev = torch.rand(2, 3)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    out = processor(
        torch.randn(2, 3, 8),
        torch.randn(2, 16, 8),
        bind_features=feat,
        gate_hysteresis=0.9,
        gate_conf_prev=prev,
        **_gate_kwargs(),
    )
    pi = spectral_graph_n8_slot_purity(att, feat)
    expected = torch.maximum(pi, 0.9 * prev)
    assert torch.allclose(out["active_mask"], expected, atol=1e-5)
    # gate_conf carries the SMOOTHED statistic so the recursion is on π̃, not π
    assert torch.allclose(out["gate_conf"], expected, atol=1e-5)
    assert not out["gate_conf"].requires_grad


def test_hysteresis_off_or_no_prev_is_identity():
    torch.manual_seed(1)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    feat = torch.randn(2, 16, 5)
    prev = torch.ones(2, 3)  # would dominate if applied
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    pi = spectral_graph_n8_slot_purity(att, feat)
    out_g0 = processor(
        torch.randn(2, 3, 8),
        torch.randn(2, 16, 8),
        bind_features=feat,
        gate_hysteresis=0.0,
        gate_conf_prev=prev,
        **_gate_kwargs(),
    )
    out_none = processor(
        torch.randn(2, 3, 8),
        torch.randn(2, 16, 8),
        bind_features=feat,
        gate_hysteresis=0.9,
        gate_conf_prev=None,
        **_gate_kwargs(),
    )
    assert torch.allclose(out_g0["active_mask"], pi, atol=1e-5)
    assert torch.allclose(out_none["active_mask"], pi, atol=1e-5)


def test_scan_carries_smoothed_conf_across_frames():
    torch.manual_seed(2)
    bsz, n_slots, n_tokens, seq = 2, 3, 16, 3
    masks = [torch.softmax(torch.randn(bsz, n_slots, n_tokens), dim=1) for _ in range(seq)]
    feats = torch.randn(bsz, seq, n_tokens, 5)
    inputs = torch.randn(bsz, seq, n_tokens, 8)
    gamma = 0.8

    processor = ScanOverTime(
        LatentProcessor(SeqStubCorrector(masks), predictor=None)
    )
    out = processor(
        torch.randn(bsz, n_slots, 8),
        inputs,
        cycle=False,
        bind_inputs=feats,
        gate_hysteresis=gamma,
        **_gate_kwargs(),
    )

    smoothed = None
    for t in range(seq):
        pi_t = spectral_graph_n8_slot_purity(masks[t], feats[:, t])
        smoothed = pi_t if smoothed is None else torch.maximum(pi_t, gamma * smoothed)
        assert torch.allclose(out["active_mask"][:, t], smoothed, atol=1e-5), f"frame {t}"
        assert torch.allclose(out["gate_conf"][:, t], smoothed, atol=1e-5), f"frame {t}"

    # γ=0 through the scan reproduces the raw per-frame π (backward compat)
    processor0 = ScanOverTime(
        LatentProcessor(SeqStubCorrector(masks), predictor=None)
    )
    out0 = processor0(
        torch.randn(bsz, n_slots, 8),
        inputs,
        cycle=False,
        bind_inputs=feats,
        gate_hysteresis=0.0,
        **_gate_kwargs(),
    )
    for t in range(seq):
        pi_t = spectral_graph_n8_slot_purity(masks[t], feats[:, t])
        assert torch.allclose(out0["active_mask"][:, t], pi_t, atol=1e-5), f"frame {t}"


def test_occluded_slot_gate_survives_pi_collapse():
    """The scenario the knob exists for: π collapses for a few frames, the smoothed
    gate decays geometrically instead of dropping to the collapsed value."""
    torch.manual_seed(3)
    bsz, n_slots, n_tokens = 1, 2, 16
    feat = torch.zeros(bsz, n_tokens, 3)
    feat[..., 0] = 1.0

    # frame 0: slot 0 exclusively owns a connected 2x2 block -> high π
    a_vis = torch.zeros(n_tokens)
    for y in range(2):
        for x in range(2):
            a_vis[y * 4 + x] = 1.0
    att_vis = torch.zeros(bsz, n_slots, n_tokens)
    att_vis[0, 0] = a_vis
    att_vis[0, 1] = 1.0 - a_vis

    # frames 1-2: "occluded" -- slot 0 spread uniformly (near-zero π)
    att_occ = torch.full((bsz, n_slots, n_tokens), 1.0 / n_slots)

    gamma = 0.9
    masks = [att_vis, att_occ, att_occ]
    feats = feat.unsqueeze(1).expand(bsz, 3, n_tokens, 3)
    processor = ScanOverTime(
        LatentProcessor(SeqStubCorrector(masks), predictor=None)
    )
    out = processor(
        torch.randn(bsz, n_slots, 8),
        torch.randn(bsz, 3, n_tokens, 8),
        cycle=False,
        bind_inputs=feats,
        gate_hysteresis=gamma,
        **_gate_kwargs(),
    )
    pi0 = spectral_graph_n8_slot_purity(att_vis, feat)[0, 0]
    gate = out["active_mask"][0, :, 0]
    assert torch.allclose(gate[1], gamma * pi0, atol=1e-5)
    assert torch.allclose(gate[2], gamma * gamma * pi0, atol=1e-5)
    assert gate[2] > 0.5 * pi0  # still a live gate after 2 occluded frames


def test_v39h_config_parses_hysteresis():
    from slotcurri import configuration

    cfg = configuration.load_config("configs/slotcurri/movi_c_attnmass_v39h.yaml")
    assert cfg.experiment_name == "movi_c_attnmass_v39h"
    amc = cfg.model.attn_mass_curriculum
    assert amc["gate_form"] == "purity_weight"
    assert amc["conf_kind"] == "spectral_graph_n8"
    assert float(amc["gate_hysteresis"]) == 0.95
    # everything else stays v39
    fc = cfg.model.feature_curriculum
    assert fc["anneal"] == "ncut" and fc["apply"] == "key"
    assert bool(fc["barrier"]) is False
