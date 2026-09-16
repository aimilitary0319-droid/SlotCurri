"""Unit tests for v37 spectral slot purity (no N-cut, no gamma, no 1/K)."""

import pytest
import torch

from slotcurri.modules.video import LatentProcessor, spectral_slot_purity, _weighted_grams


class StubCorrector(torch.nn.Module):
    def __init__(self, masks):
        super().__init__()
        self.masks_out = masks

    def forward(self, state, inputs, n_iters=None, **kwargs):
        return {"slots": state + 1.0, "masks": self.masks_out}


def _hand_pi(att, features, eps=1e-6, divide_by_mass=True):
    """Independent copy of the formula for the test oracle."""
    z = torch.nn.functional.normalize(features.float(), dim=-1)
    a = att.float()
    mass = a.sum(dim=-1)
    bsz, n_slots, _ = a.shape
    out = []
    for s in range(n_slots):
        w = a[:, s].unsqueeze(-1) * z
        cmat = torch.bmm(w.transpose(1, 2), w)
        cmat = 0.5 * (cmat + cmat.transpose(1, 2))
        evals = torch.linalg.eigvalsh(cmat)
        gap = (evals[:, -1] - evals[:, -2]).clamp_min(0.0)
        if divide_by_mass:
            out.append(gap / (mass[:, s] + eps))
        else:
            out.append(gap)
    stacked = torch.stack(out, dim=-1)
    return stacked.clamp(0.0, 1.0) if divide_by_mass else stacked.clamp_min(0.0)


def test_weighted_grams_matches_einsum():
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(3, 20, 8), dim=-1)
    a = torch.softmax(torch.randn(3, 5, 20), dim=1)
    got = _weighted_grams(z, a)
    a2 = a * a
    ref = torch.einsum("bnd,bsn,bne->bsde", z, a2, z)
    ref = 0.5 * (ref + ref.transpose(-1, -2))
    assert torch.allclose(got, ref, atol=1e-5)


def test_rank1_exclusive_recovers_ownership_purity():
    n_tokens, dim = 8, 4
    z = torch.zeros(2, n_tokens, dim)
    z[..., 0] = 1.0
    att = torch.zeros(2, 3, n_tokens)
    att[:, 0, :] = 1.0
    pi = spectral_slot_purity(att, z)
    # λ1 = Σ a^2 = N, λ2 = 0, mass = N -> π = 1 for the owner, 0 for empty slots
    assert torch.allclose(pi[:, 0], torch.ones(2), atol=1e-5)
    assert torch.allclose(pi[:, 1:], torch.zeros(2, 2), atol=1e-5)
    ownership = (att * att).sum(-1) / att.sum(-1).clamp_min(1e-8)
    assert torch.allclose(pi[:, 0], ownership[:, 0], atol=1e-5)


def test_two_mode_exclusive_drops_pi():
    """v32 blind spot: exclusive owner of two orthogonal modes is not π~=1."""
    n_tokens, dim = 8, 4
    z = torch.zeros(1, n_tokens, dim)
    z[0, :4, 0] = 1.0
    z[0, 4:, 1] = 1.0
    att = torch.zeros(1, 2, n_tokens)
    att[0, 0, :] = 1.0
    pi = spectral_slot_purity(att, z)
    ownership = (att * att).sum(-1) / att.sum(-1).clamp_min(1e-8)
    assert ownership[0, 0].item() > 0.99
    assert pi[0, 0].item() < 0.05


def test_matches_hand_formula_and_is_detached():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(3, 5, 16), dim=1)
    att.requires_grad_(True)
    feat = torch.randn(3, 16, 8, requires_grad=True)
    pi = spectral_slot_purity(att, feat, chunk_size=2)
    # k=2 subspace iteration vs full eigh; the gate only needs the gap.
    assert torch.allclose(pi, _hand_pi(att.detach(), feat.detach()), atol=1e-3)
    assert not pi.requires_grad
    assert (pi >= 0).all() and (pi <= 1).all()


def test_uniform_ghost_is_small():
    n_tokens, dim, n_slots = 32, 6, 4
    torch.manual_seed(1)
    z = torch.randn(2, n_tokens, dim)
    att = torch.full((2, n_slots, n_tokens), 1.0 / n_slots)
    pi = spectral_slot_purity(att, z)
    assert (pi < 0.35).all()
    assert float(pi.mean()) < 0.2


def test_missing_bind_features_raises():
    att = torch.full((1, 2, 4), 0.5)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    state = torch.randn(1, 2, 8)
    inputs = torch.randn(1, 4, 8)
    with pytest.raises(ValueError, match="bind_features"):
        processor(
            state, inputs, gate_p=None, default_idx=[],
            gate_form="purity_weight", conf_kind="spectral",
        )


def test_processor_gate_is_spectral_pi():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 4, 10), dim=1)
    feat = torch.randn(2, 10, 6)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    out = processor(
        torch.randn(2, 4, 8), torch.randn(2, 10, 8),
        gate_p=None, default_idx=[], mass_gamma=4.0,
        gate_form="purity_weight", conf_kind="spectral",
        bind_features=feat, state_max_norm=True,
    )
    expected = spectral_slot_purity(att, feat)
    assert torch.allclose(out["active_mask"], expected, atol=1e-5)
    assert torch.equal(out["state_gate"], out["active_mask"])
    assert torch.allclose(out["gate_conf"], expected, atol=1e-5)
    # mass_gamma must not change spectral π (raw A; a^2 is the within-slot moment)
    out_g1 = processor(
        torch.randn(2, 4, 8), torch.randn(2, 10, 8),
        gate_p=None, default_idx=[], mass_gamma=1.0,
        gate_form="purity_weight", conf_kind="spectral",
        bind_features=feat,
    )
    assert torch.allclose(out["active_mask"], out_g1["active_mask"], atol=1e-5)


def test_sharpened_a_is_not_v37_but_drops_always_second():
    """Gamma is not v37. If C_s is built on ã, always-second ghosts drop; two-mode winners stay ~0."""
    n_tokens, dim = 8, 4
    z = torch.zeros(1, n_tokens, dim)
    z[..., 0] = 1.0
    att = torch.zeros(1, 2, n_tokens)
    att[0, 0, :] = 0.8
    att[0, 1, :] = 0.2
    pi_raw = spectral_slot_purity(att, z)
    sharp = att.pow(2.0)
    sharp = sharp / sharp.sum(dim=1, keepdim=True).clamp_min(1e-8)
    pi_g2 = spectral_slot_purity(sharp, z)
    assert abs(pi_raw[0, 1].item() - 0.2) < 1e-4
    assert pi_g2[0, 1].item() < 0.08
    assert pi_g2[0, 0].item() > 0.9

    z2 = torch.zeros(1, n_tokens, dim)
    z2[0, :4, 0] = 1.0
    z2[0, 4:, 1] = 1.0
    att2 = torch.zeros(1, 2, n_tokens)
    att2[0, 0, :] = 1.0
    assert spectral_slot_purity(att2, z2)[0, 0].item() < 0.05


def test_proj_dim_keeps_rank1_and_two_mode():
    """64-d (here 8-d) JL map: one appearance stays π~1; two orthogonal modes drop."""
    n_tokens, dim = 8, 32
    z = torch.zeros(1, n_tokens, dim)
    z[..., 0] = 1.0
    att = torch.zeros(1, 2, n_tokens)
    att[0, 0, :] = 1.0
    pi_one = spectral_slot_purity(att, z, proj_dim=8)
    assert pi_one[0, 0].item() > 0.99
    z2 = torch.zeros(1, n_tokens, dim)
    z2[0, :4, 0] = 1.0
    z2[0, 4:, 1] = 1.0
    att2 = torch.zeros(1, 2, n_tokens)
    att2[0, 0, :] = 1.0
    pi_two = spectral_slot_purity(att2, z2, proj_dim=8)
    assert pi_two[0, 0].item() < 0.25


def test_processor_passes_spectral_proj_dim():
    n_tokens, dim = 8, 32
    z = torch.zeros(1, n_tokens, dim)
    z[..., 0] = 1.0
    att = torch.zeros(1, 2, n_tokens)
    att[0, 0, :] = 1.0
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    out = processor(
        torch.randn(1, 2, 8), torch.randn(1, n_tokens, 8),
        gate_p=None, default_idx=[],
        gate_form="purity_weight", conf_kind="spectral",
        bind_features=z, spectral_proj_dim=8,
    )
    expected = spectral_slot_purity(att, z, proj_dim=8)
    assert torch.allclose(out["gate_conf"], expected, atol=1e-5)


def test_v37_configs_parse_spectral_ncut_no_barrier():
    from slotcurri import configuration

    for path, name, n_slots in (
        ("configs/slotcurri/ytvis2021_attnmass_v37.yaml", "ytvis_attnmass_v37", 7),
        ("configs/slotcurri/movi_c_attnmass_v37.yaml", "movi_c_attnmass_v37", 11),
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
        assert amc["conf_kind"] == "spectral"
        assert int(amc.get("spectral_proj_dim", 0)) == 64
        assert bool(amc.get("purity_normalize", False)) is False
        assert abs(float(amc["mass_gamma"]) - 1.0) < 1e-9
        assert int(cfg.globals["NUM_SLOTS"]) == n_slots
        assert bool(cfg.model.cyclic_inference) is False

    v36 = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v36.yaml")
    assert v36.model.feature_curriculum["anneal"] == "ncut"
    assert v36.model.feature_curriculum.get("barrier", True) is not False
    assert v36.model.attn_mass_curriculum["conf_kind"] == "purity_sharp"


def test_gap_rank1_recovers_sum_a2_not_ownership():
    n_tokens, dim = 8, 4
    z = torch.zeros(2, n_tokens, dim)
    z[..., 0] = 1.0
    att = torch.zeros(2, 3, n_tokens)
    att[:, 0, :] = 1.0
    pi = spectral_slot_purity(att, z, divide_by_mass=False)
    # λ1 = Σ a^2 = N, λ2 = 0 -> π = N for the owner, 0 for empty slots
    assert torch.allclose(pi[:, 0], torch.full((2,), float(n_tokens)), atol=1e-4)
    assert torch.allclose(pi[:, 1:], torch.zeros(2, 2), atol=1e-4)
    v37 = spectral_slot_purity(att, z, divide_by_mass=True)
    assert torch.allclose(v37[:, 0], torch.ones(2), atol=1e-5)
    assert (pi[:, 0] > v37[:, 0] + 1.0).all()


def test_gap_two_mode_drops_and_leftover_is_small():
    n_tokens, dim = 8, 4
    z = torch.zeros(1, n_tokens, dim)
    z[0, :4, 0] = 1.0
    z[0, 4:, 1] = 1.0
    att_two = torch.zeros(1, 2, n_tokens)
    att_two[0, 0, :] = 1.0
    att_one = torch.zeros(1, 2, n_tokens)
    att_one[0, 0, :4] = 1.0
    att_left = torch.zeros(1, 2, n_tokens)
    att_left[0, 0, 0] = 1.0
    pi_two = spectral_slot_purity(att_two, z, divide_by_mass=False)
    pi_one = spectral_slot_purity(att_one, z, divide_by_mass=False)
    pi_left = spectral_slot_purity(att_left, z, divide_by_mass=False)
    assert pi_two[0, 0].item() < 0.5
    assert pi_one[0, 0].item() > 3.0
    assert pi_left[0, 0].item() < pi_one[0, 0].item()
    v37_left = spectral_slot_purity(att_left, z, divide_by_mass=True)
    assert v37_left[0, 0].item() > 0.9


def test_gap_matches_hand_formula_and_is_detached():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(3, 5, 16), dim=1)
    att.requires_grad_(True)
    feat = torch.randn(3, 16, 8, requires_grad=True)
    pi = spectral_slot_purity(att, feat, chunk_size=2, divide_by_mass=False)
    assert torch.allclose(
        pi, _hand_pi(att.detach(), feat.detach(), divide_by_mass=False), atol=5e-3
    )
    assert not pi.requires_grad
    assert (pi >= 0).all()


def test_processor_gate_is_spectral_gap():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 4, 10), dim=1)
    feat = torch.randn(2, 10, 6)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    out = processor(
        torch.randn(2, 4, 8), torch.randn(2, 10, 8),
        gate_p=None, default_idx=[], mass_gamma=4.0,
        gate_form="purity_weight", conf_kind="spectral_gap",
        bind_features=feat, state_max_norm=True,
    )
    expected = spectral_slot_purity(att, feat, divide_by_mass=False)
    assert torch.allclose(out["active_mask"], expected, atol=1e-5)
    assert torch.allclose(out["gate_conf"], expected, atol=1e-5)
    out_v37 = processor(
        torch.randn(2, 4, 8), torch.randn(2, 10, 8),
        gate_p=None, default_idx=[],
        gate_form="purity_weight", conf_kind="spectral",
        bind_features=feat,
    )
    assert not torch.allclose(out["gate_conf"], out_v37["gate_conf"], atol=1e-3)


def test_ratio_rank1_is_one_two_mode_is_zero_empty_is_zero():
    n_tokens, dim = 8, 4
    z = torch.zeros(1, n_tokens, dim)
    z[0, :4, 0] = 1.0
    z[0, 4:, 1] = 1.0
    att_one = torch.zeros(1, 3, n_tokens)
    att_one[0, 0, :4] = 1.0
    att_two = torch.zeros(1, 3, n_tokens)
    att_two[0, 0, :] = 1.0
    r_one = spectral_slot_purity(att_one, z, ratio=True)
    r_two = spectral_slot_purity(att_two, z, ratio=True)
    assert r_one[0, 0].item() > 0.95
    assert r_two[0, 0].item() < 0.05
    assert torch.allclose(r_one[0, 1:], torch.zeros(2), atol=1e-4)
    # (λ1-λ2)/(λ1+λ2) = (r-1)/(r+1) for r=λ1/λ2; two equal modes r=1 → 0
    gap_two = spectral_slot_purity(att_two, z, divide_by_mass=False)
    assert gap_two[0, 0].item() < 0.5


def test_processor_gate_is_spectral_ratio():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 4, 10), dim=1)
    feat = torch.randn(2, 10, 6)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    out = processor(
        torch.randn(2, 4, 8), torch.randn(2, 10, 8),
        gate_p=None, default_idx=[],
        gate_form="purity_weight", conf_kind="spectral_ratio",
        bind_features=feat, state_max_norm=True,
    )
    expected = spectral_slot_purity(att, feat, ratio=True)
    assert torch.allclose(out["active_mask"], expected, atol=1e-5)
    assert torch.allclose(out["gate_conf"], expected, atol=1e-5)
    assert (out["gate_conf"] >= 0).all() and (out["gate_conf"] <= 1).all()
    out_gap = processor(
        torch.randn(2, 4, 8), torch.randn(2, 10, 8),
        gate_p=None, default_idx=[],
        gate_form="purity_weight", conf_kind="spectral_gap",
        bind_features=feat,
    )
    assert not torch.allclose(out["gate_conf"], out_gap["gate_conf"], atol=1e-3)


def test_v37_purity_normalize_maps_pi_like_ownership():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 4, 10), dim=1)
    feat = torch.randn(2, 10, 6)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    out = processor(
        torch.randn(2, 4, 8), torch.randn(2, 10, 8),
        gate_p=None, default_idx=[],
        gate_form="purity_weight", conf_kind="spectral",
        bind_features=feat, purity_normalize=True,
    )
    pi = spectral_slot_purity(att, feat)
    n_slots = pi.shape[-1]
    mapped = ((n_slots * pi - 1.0) / (n_slots - 1.0)).clamp(0.0, 1.0)
    assert torch.allclose(out["gate_conf"], pi, atol=1e-5)
    assert torch.allclose(out["active_mask"], mapped, atol=1e-5)


def test_v37g_configs_parse_spectral_gap():
    from slotcurri import configuration

    for path, name, n_slots in (
        ("configs/slotcurri/ytvis2021_attnmass_v37g.yaml", "ytvis_attnmass_v37g", 7),
        ("configs/slotcurri/movi_c_attnmass_v37g.yaml", "movi_c_attnmass_v37g", 11),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        fc = cfg.model.feature_curriculum
        assert bool(fc["enabled"]) is True
        assert fc["anneal"] == "ncut"
        assert fc["apply"] == "key"
        assert bool(fc["barrier"]) is False
        amc = cfg.model.attn_mass_curriculum
        assert amc["gate_form"] == "purity_weight"
        assert amc["conf_kind"] == "spectral_gap"
        assert int(amc.get("spectral_proj_dim", 0)) == 64
        assert bool(amc.get("purity_normalize", False)) is False
        assert int(cfg.globals["NUM_SLOTS"]) == n_slots

    v37 = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v37.yaml")
    assert v37.model.attn_mass_curriculum["conf_kind"] == "spectral"


def test_v37r_configs_parse_spectral_ratio():
    from slotcurri import configuration

    for path, name, n_slots in (
        ("configs/slotcurri/ytvis2021_attnmass_v37r.yaml", "ytvis_attnmass_v37r", 7),
        ("configs/slotcurri/movi_c_attnmass_v37r.yaml", "movi_c_attnmass_v37r", 11),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        amc = cfg.model.attn_mass_curriculum
        assert amc["gate_form"] == "purity_weight"
        assert amc["conf_kind"] == "spectral_ratio"
        assert int(amc.get("spectral_proj_dim", 0)) == 64
        assert bool(amc.get("purity_normalize", False)) is False
        assert int(cfg.globals["NUM_SLOTS"]) == n_slots


def test_frame_encoder_smoothing_key_only_leaves_target_raw():
    from slotcurri.modules.encoders import FeatureSmoothing, FrameEncoder

    class _TokenBackbone(torch.nn.Module):
        def __init__(self, tokens):
            super().__init__()
            self.tokens = tokens

        def forward(self, images):
            return self.tokens.clone()

    torch.manual_seed(0)
    tokens = torch.randn(2, 16, 8)
    enc = FrameEncoder(backbone=_TokenBackbone(tokens), output_transform=torch.nn.Identity())
    enc.feature_smoothing = FeatureSmoothing(tau=0.1, window=None, chunk_size=4)
    enc.feature_smoothing_mix = 0.0
    enc.feature_curriculum_apply = "key"
    images = torch.zeros(2, 3, 4, 4)

    enc.train()
    out = enc(images)
    assert torch.equal(out["backbone_features"], tokens)
    assert "features_key" in out and "backbone_key" in out
    assert not torch.allclose(out["backbone_key"], tokens)
    assert torch.allclose(out["features"], tokens)

    enc.eval()
    out_eval = enc(images)
    assert "features_key" not in out_eval
    assert "backbone_key" not in out_eval
    assert torch.equal(out_eval["backbone_features"], tokens)
