"""Unit tests for v38 relation-graph spectral slot purity."""

import pytest
import torch

from slotcurri.modules.video import (
    LatentProcessor,
    spectral_graph_slot_purity,
    spectral_slot_purity,
)


class StubCorrector(torch.nn.Module):
    def __init__(self, masks):
        super().__init__()
        self.masks_out = masks

    def forward(self, state, inputs, n_iters=None, **kwargs):
        return {"slots": state + 1.0, "masks": self.masks_out}


def _hand_pi(att, features, eps=1e-6):
    """Independent copy: π = λ1 - max(λ2, 0) of G_s = diag(a) S diag(a)."""
    z = torch.nn.functional.normalize(features.float(), dim=-1)
    a = att.float()
    rel = torch.bmm(z, z.transpose(1, 2)).clamp_min(0.0)
    rel.diagonal(dim1=-2, dim2=-1).zero_()
    deg = rel.sum(dim=-1).clamp_min(eps)
    d_inv = deg.rsqrt()
    s_mat = d_inv.unsqueeze(-1) * rel * d_inv.unsqueeze(-2)
    bsz, n_slots, n_tokens = a.shape
    out = []
    for s in range(n_slots):
        aa = a[:, s]
        gram = aa.unsqueeze(-1) * s_mat * aa.unsqueeze(-2)
        gram = 0.5 * (gram + gram.transpose(-1, -2))
        evals = torch.linalg.eigvalsh(gram)
        lam1 = evals[:, -1]
        lam2 = evals[:, -2] if n_tokens >= 2 else torch.zeros_like(lam1)
        out.append((lam1 - lam2.clamp_min(0.0)).clamp_min(0.0))
    return torch.stack(out, dim=-1)


def test_matches_hand_formula_and_is_detached():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(3, 5, 16), dim=1)
    att.requires_grad_(True)
    feat = torch.randn(3, 16, 8, requires_grad=True)
    pi = spectral_graph_slot_purity(att, feat, chunk_size=2)
    assert torch.allclose(pi, _hand_pi(att.detach(), feat.detach()), atol=1e-3)
    assert not pi.requires_grad
    assert (pi >= 0).all()


def test_one_coherent_object_beats_two_communities():
    """Exclusive owner of one clique is high; same slot owning two cliques is low."""
    n_tokens, dim = 8, 4
    z = torch.zeros(1, n_tokens, dim)
    z[0, :4, 0] = 1.0
    z[0, 4:, 1] = 1.0
    att_one = torch.zeros(1, 2, n_tokens)
    att_one[0, 0, :4] = 1.0
    att_two = torch.zeros(1, 2, n_tokens)
    att_two[0, 0, :] = 1.0
    pi_one = spectral_graph_slot_purity(att_one, z)
    pi_two = spectral_graph_slot_purity(att_two, z)
    assert pi_one[0, 0].item() > 0.5
    assert pi_two[0, 0].item() < 0.15
    assert pi_one[0, 0].item() > pi_two[0, 0].item() + 0.3


def test_split_related_patches_drops_lambda1():
    """Fragments of one clique score lower than owning the whole clique."""
    n_tokens, dim = 8, 4
    z = torch.zeros(1, n_tokens, dim)
    z[..., 0] = 1.0
    att_full = torch.zeros(1, 2, n_tokens)
    att_full[0, 0, :] = 1.0
    att_split = torch.zeros(1, 2, n_tokens)
    att_split[0, 0, :4] = 1.0
    att_split[0, 1, 4:] = 1.0
    pi_full = spectral_graph_slot_purity(att_full, z)
    pi_split = spectral_graph_slot_purity(att_split, z)
    assert pi_full[0, 0].item() > pi_split[0, 0].item()
    assert pi_full[0, 0].item() > pi_split[0, 1].item()


def test_uniform_ghost_is_small():
    n_tokens, dim, n_slots = 32, 6, 4
    torch.manual_seed(1)
    z = torch.randn(2, n_tokens, dim)
    att = torch.full((2, n_slots, n_tokens), 1.0 / n_slots)
    pi = spectral_graph_slot_purity(att, z)
    assert (pi < 0.15).all()
    assert float(pi.mean()) < 0.05


def test_single_edge_negative_lambda2_does_not_inflate_gap():
    """Two identical tokens: S = [[0,1],[1,0]], evals 1 and -1, π = 1 not 2."""
    z = torch.zeros(1, 2, 2)
    z[0, :, 0] = 1.0
    att = torch.ones(1, 1, 2)
    pi = spectral_graph_slot_purity(att, z)
    hand = _hand_pi(att, z)
    assert torch.allclose(pi, hand, atol=1e-3)
    assert abs(pi[0, 0].item() - 1.0) < 1e-3


def test_v37_feature_gram_is_unchanged():
    """v38 must not change conf_kind=spectral (D×D Gram)."""
    torch.manual_seed(2)
    att = torch.softmax(torch.randn(2, 3, 10), dim=1)
    feat = torch.randn(2, 10, 6)
    pi_gram = spectral_slot_purity(att, feat)
    pi_graph = spectral_graph_slot_purity(att, feat)
    assert not torch.allclose(pi_gram, pi_graph, atol=1e-2)


def test_processor_gate_is_spectral_graph_pi():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 4, 10), dim=1)
    feat = torch.randn(2, 10, 6)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    out = processor(
        torch.randn(2, 4, 8), torch.randn(2, 10, 8),
        gate_p=None, default_idx=[], mass_gamma=4.0,
        gate_form="purity_weight", conf_kind="spectral_graph",
        bind_features=feat, state_max_norm=True,
    )
    expected = spectral_graph_slot_purity(att, feat)
    assert torch.allclose(out["active_mask"], expected, atol=1e-5)
    assert torch.equal(out["state_gate"], out["active_mask"])
    assert torch.allclose(out["gate_conf"], expected, atol=1e-5)
    out_g1 = processor(
        torch.randn(2, 4, 8), torch.randn(2, 10, 8),
        gate_p=None, default_idx=[], mass_gamma=1.0,
        gate_form="purity_weight", conf_kind="spectral_graph",
        bind_features=feat,
    )
    assert torch.allclose(out["active_mask"], out_g1["active_mask"], atol=1e-5)


def test_missing_bind_features_raises():
    att = torch.full((1, 2, 4), 0.5)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    with pytest.raises(ValueError, match="bind_features"):
        processor(
            torch.randn(1, 2, 8), torch.randn(1, 4, 8),
            gate_p=None, default_idx=[],
            gate_form="purity_weight", conf_kind="spectral_graph",
        )


def test_v38_configs_parse_spectral_graph_ncut_no_barrier():
    from slotcurri import configuration

    for path, name, n_slots in (
        ("configs/slotcurri/ytvis2021_attnmass_v38.yaml", "ytvis_attnmass_v38", 7),
        ("configs/slotcurri/movi_c_attnmass_v38.yaml", "movi_c_attnmass_v38", 11),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        fc = cfg.model.feature_curriculum
        assert bool(fc["enabled"]) is True
        assert fc["anneal"] == "ncut"
        assert fc["apply"] == "key"
        assert bool(fc["barrier"]) is False
        assert int(fc["anneal_steps"]) == 50000
        amc = cfg.model.attn_mass_curriculum
        assert amc["gate_form"] == "purity_weight"
        assert amc["conf_kind"] == "spectral_graph"
        assert bool(amc.get("purity_normalize", False)) is False
        assert int(cfg.globals["NUM_SLOTS"]) == n_slots

    v37 = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v37.yaml")
    assert v37.model.attn_mass_curriculum["conf_kind"] == "spectral"
    assert bool(v37.model.feature_curriculum["barrier"]) is False
