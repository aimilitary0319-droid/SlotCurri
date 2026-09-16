"""Unit tests for v39 8-neighbor relation-graph spectral slot purity."""

import math

import pytest
import torch

from slotcurri.modules.video import (
    LatentProcessor,
    _N8_OFFSETS,
    _apply_g_n8,
    _apply_r_n8,
    _apply_s_n8,
    _n8_affinity,
    _n8_gather,
    _pad1_hw,
    _shift_hw,
    _top2_algebraic_n8,
    spectral_graph_n8_eigs,
    spectral_graph_n8_slot_purity,
    spectral_graph_slot_purity,
)


class StubCorrector(torch.nn.Module):
    def __init__(self, masks):
        super().__init__()
        self.masks_out = masks

    def forward(self, state, inputs, n_iters=None, **kwargs):
        return {"slots": state + 1.0, "masks": self.masks_out}


def _n8_S_dense(z, eps=1e-6):
    """Dense 8-neighbor S for the hand formula. z: (B, N, D) already L2-normed."""
    bsz, n_tokens, _ = z.shape
    grid = int(math.sqrt(n_tokens))
    rel = z.new_zeros(bsz, n_tokens, n_tokens)
    for y in range(grid):
        for x in range(grid):
            i = y * grid + x
            for dy, dx in _N8_OFFSETS:
                yy, xx = y + dy, x + dx
                if 0 <= yy < grid and 0 <= xx < grid:
                    j = yy * grid + xx
                    rel[:, i, j] = (z[:, i] * z[:, j]).sum(dim=-1).clamp_min(0.0)
    deg = rel.sum(dim=-1).clamp_min(eps)
    d_inv = deg.rsqrt()
    return d_inv.unsqueeze(-1) * rel * d_inv.unsqueeze(-2)


def _hand_pi_n8(
    att, features, eps=1e-6, divide_by_lambda1=False, l1_minus_imp=False,
    use_lambda1=False,
):
    z = torch.nn.functional.normalize(features.float(), dim=-1)
    a = att.float()
    s_mat = _n8_S_dense(z, eps)
    bsz, n_slots, n_tokens = a.shape
    out = []
    for s in range(n_slots):
        aa = a[:, s]
        gram = aa.unsqueeze(-1) * s_mat * aa.unsqueeze(-2)
        gram = 0.5 * (gram + gram.transpose(-1, -2))
        evals = torch.linalg.eigvalsh(gram)
        lam1 = evals[:, -1]
        lam2 = evals[:, -2] if n_tokens >= 2 else torch.zeros_like(lam1)
        lam2p = lam2.clamp_min(0.0)
        if use_lambda1:
            out.append(lam1.clamp_min(0.0))
        elif l1_minus_imp:
            out.append((lam1 - lam2p / (lam1 + eps)).clamp_min(0.0))
        elif divide_by_lambda1:
            out.append(((lam1 - lam2p).clamp_min(0.0) / lam1.clamp_min(eps)).clamp(0.0, 1.0))
        else:
            out.append((lam1 - lam2p).clamp_min(0.0))
    return torch.stack(out, dim=-1)


def _mask_xy(grid, ys, ye, xs, xe):
    a = torch.zeros(grid * grid)
    for y in range(ys, ye):
        for x in range(xs, xe):
            a[y * grid + x] = 1.0
    return a


def _n8_affinity_shift_hw(z, eps=1e-6):
    """Original eight-pad affinity; used to lock the pad-once rewrite."""
    bsz, n_tokens, _ = z.shape
    grid = int(math.sqrt(n_tokens))
    z_hw = z.view(bsz, grid, grid, -1)
    weights = []
    deg = z.new_zeros(bsz, grid, grid)
    for dy, dx in _N8_OFFSETS:
        r = (z_hw * _shift_hw(z_hw, dy, dx)).sum(dim=-1).clamp_min(0.0)
        weights.append(r)
        deg = deg + r
    d_inv = deg.clamp_min(eps).rsqrt()
    return d_inv, torch.stack(weights, dim=1)


def _apply_s_n8_shift_hw(d_inv, weights, vec):
    bsz, n_tokens, n_c = vec.shape
    grid = d_inv.shape[1]
    v_hw = vec.view(bsz, grid, grid, n_c)
    w_hw = d_inv.unsqueeze(-1) * v_hw
    acc = vec.new_zeros(bsz, grid, grid, n_c)
    for i, (dy, dx) in enumerate(_N8_OFFSETS):
        acc = acc + weights[:, i].unsqueeze(-1) * _shift_hw(w_hw, dy, dx)
    return (d_inv.unsqueeze(-1) * acc).view(bsz, n_tokens, n_c)


def _apply_r_n8_shift_hw(weights, vec):
    bsz, n_tokens, n_c = vec.shape
    grid = weights.shape[2]
    v_hw = vec.view(bsz, grid, grid, n_c)
    acc = vec.new_zeros(bsz, grid, grid, n_c)
    for i, (dy, dx) in enumerate(_N8_OFFSETS):
        acc = acc + weights[:, i].unsqueeze(-1) * _shift_hw(v_hw, dy, dx)
    return acc.view(bsz, n_tokens, n_c)


def test_pad1_gather_matches_shift_hw():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 5, 7)
    xp = _pad1_hw(x)
    for dy, dx in _N8_OFFSETS:
        got = _n8_gather(xp, dy, dx, 5, 5)
        assert torch.equal(got, _shift_hw(x, dy, dx))


def test_n8_kernels_match_shift_hw_reference():
    torch.manual_seed(1)
    z = torch.nn.functional.normalize(torch.randn(2, 36, 11), dim=-1)
    vec = torch.randn(2, 36, 14)
    d_inv, weights = _n8_affinity(z, 1e-6)
    d_ref, w_ref = _n8_affinity_shift_hw(z, 1e-6)
    assert torch.equal(d_inv, d_ref)
    assert torch.equal(weights, w_ref)
    assert torch.equal(_apply_s_n8(d_inv, weights, vec), _apply_s_n8_shift_hw(d_inv, weights, vec))
    assert torch.equal(_apply_r_n8(weights, vec), _apply_r_n8_shift_hw(weights, vec))


def test_matches_hand_formula_and_is_detached():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 4, 16), dim=1)
    att.requires_grad_(True)
    feat = torch.randn(2, 16, 6, requires_grad=True)
    pi = spectral_graph_n8_slot_purity(att, feat, chunk_size=1)
    assert torch.allclose(pi, _hand_pi_n8(att.detach(), feat.detach()), atol=2e-3)
    assert not pi.requires_grad
    assert (pi >= 0).all()


def test_disconnected_merge_scores_below_exclusive():
    """Same features, two far 2x2 blobs: n8 merge < exclusive; dense merge > exclusive."""
    grid = 6
    n_tokens = grid * grid
    z = torch.zeros(1, n_tokens, 3)
    z[..., 0] = 1.0
    a_ex = _mask_xy(grid, 0, 2, 0, 2)
    a_mer = a_ex + _mask_xy(grid, 4, 6, 4, 6)
    att_ex = torch.zeros(1, 2, n_tokens)
    att_ex[0, 0] = a_ex
    att_mer = torch.zeros(1, 2, n_tokens)
    att_mer[0, 0] = a_mer
    pi_ex = spectral_graph_n8_slot_purity(att_ex, z)
    pi_mer = spectral_graph_n8_slot_purity(att_mer, z)
    assert pi_mer[0, 0].item() < pi_ex[0, 0].item()
    dense_ex = spectral_graph_slot_purity(att_ex, z)
    dense_mer = spectral_graph_slot_purity(att_mer, z)
    assert dense_mer[0, 0].item() > dense_ex[0, 0].item()


def test_split_does_not_beat_exclusive():
    grid = 6
    n_tokens = grid * grid
    z = torch.zeros(1, n_tokens, 3)
    z[..., 0] = 1.0
    a_full = _mask_xy(grid, 1, 5, 1, 5)
    a_left = _mask_xy(grid, 1, 5, 1, 3)
    att_full = torch.zeros(1, 2, n_tokens)
    att_full[0, 0] = a_full
    att_split = torch.zeros(1, 2, n_tokens)
    att_split[0, 0] = a_left
    pi_full = spectral_graph_n8_slot_purity(att_full, z)
    pi_split = spectral_graph_n8_slot_purity(att_split, z)
    assert pi_split[0, 0].item() <= pi_full[0, 0].item() + 1e-4


def test_uniform_1k_slots_are_similar():
    torch.manual_seed(3)
    n_tokens, n_slots = 16, 4
    z = torch.randn(2, n_tokens, 5)
    att = torch.full((2, n_slots, n_tokens), 1.0 / n_slots)
    pi = spectral_graph_n8_slot_purity(att, z)
    span = (pi.max(dim=-1).values - pi.min(dim=-1).values).max().item()
    assert span < 1e-5


def test_lambda1_at_most_u():
    torch.manual_seed(4)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    feat = torch.randn(2, 16, 4)
    z = torch.nn.functional.normalize(feat.float(), dim=-1)
    from slotcurri.modules.video import _top2_algebraic_n8

    lam1, _lam2 = _top2_algebraic_n8(z, att.float(), n_iter=16, eps=1e-6)
    u = att.float().amax(dim=-1).square()
    assert (lam1 <= u + 1e-4).all()


def test_non_square_n_raises():
    att = torch.softmax(torch.randn(1, 2, 12), dim=1)
    feat = torch.randn(1, 12, 4)
    with pytest.raises(ValueError, match="square patch grid"):
        spectral_graph_n8_slot_purity(att, feat)


def test_processor_gate_is_n8_pi():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    feat = torch.randn(2, 16, 5)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    out = processor(
        torch.randn(2, 3, 8),
        torch.randn(2, 16, 8),
        gate_p=None,
        default_idx=[],
        mass_gamma=4.0,
        gate_form="purity_weight",
        conf_kind="spectral_graph_n8",
        bind_features=feat,
        state_max_norm=True,
    )
    expected = spectral_graph_n8_slot_purity(att, feat)
    assert torch.allclose(out["active_mask"], expected, atol=1e-5)
    assert torch.equal(out["state_gate"], out["active_mask"])
    v38 = processor(
        torch.randn(2, 3, 8),
        torch.randn(2, 16, 8),
        gate_p=None,
        default_idx=[],
        gate_form="purity_weight",
        conf_kind="spectral_graph",
        bind_features=feat,
    )
    assert not torch.allclose(out["active_mask"], v38["active_mask"], atol=1e-3)


def test_missing_bind_features_raises():
    att = torch.full((1, 2, 16), 0.5)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    with pytest.raises(ValueError, match="bind_features"):
        processor(
            torch.randn(1, 2, 8),
            torch.randn(1, 16, 8),
            gate_p=None,
            default_idx=[],
            gate_form="purity_weight",
            conf_kind="spectral_graph_n8",
        )


def test_v39_configs_parse_n8_ncut_no_barrier():
    from slotcurri import configuration

    for path, name, n_slots in (
        ("configs/slotcurri/ytvis2021_attnmass_v39.yaml", "ytvis_attnmass_v39", 7),
        ("configs/slotcurri/movi_c_attnmass_v39.yaml", "movi_c_attnmass_v39", 11),
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
        assert amc["conf_kind"] == "spectral_graph_n8"
        assert bool(amc.get("purity_normalize", False)) is False
        assert int(cfg.globals["NUM_SLOTS"]) == n_slots

    v38 = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v38.yaml")
    assert v38.model.attn_mass_curriculum["conf_kind"] == "spectral_graph"
    assert bool(v38.model.feature_curriculum["barrier"]) is False


def test_v39ema_configs_parse_temporal_ema():
    from slotcurri import configuration

    for path, name, n_slots in (
        ("configs/slotcurri/ytvis2021_attnmass_v39ema.yaml", "ytvis_attnmass_v39ema", 7),
        ("configs/slotcurri/movi_c_attnmass_v39ema.yaml", "movi_c_attnmass_v39ema", 11),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        amc = cfg.model.attn_mass_curriculum
        assert amc["conf_kind"] == "spectral_graph_n8"
        assert amc["gate_form"] == "purity_weight"
        assert float(amc["state_gate_ema"]) == 0.2
        assert float(amc.get("state_gate_hold", 0.0)) == 0.0
        assert bool(amc["state_max_norm"]) is True
        assert int(cfg.globals["NUM_SLOTS"]) == n_slots
        v39 = configuration.load_config(path.replace("v39ema", "v39"))
        assert v39.model.attn_mass_curriculum.get("state_gate_ema", 1.0) == 1.0
        assert bool(cfg.model.cyclic_inference) is False


def test_rel_is_gap_over_lambda1_and_detached():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 4, 16), dim=1)
    att.requires_grad_(True)
    feat = torch.randn(2, 16, 6, requires_grad=True)
    pi = spectral_graph_n8_slot_purity(
        att, feat, chunk_size=1, divide_by_lambda1=True
    )
    expected = _hand_pi_n8(att.detach(), feat.detach(), divide_by_lambda1=True)
    assert torch.allclose(pi, expected, atol=2e-3)
    assert not pi.requires_grad
    assert (pi >= 0).all() and (pi <= 1).all()
    gap = spectral_graph_n8_slot_purity(att.detach(), feat.detach(), chunk_size=1)
    from slotcurri.modules.video import _top2_algebraic_n8

    z = torch.nn.functional.normalize(feat.detach().float(), dim=-1)
    lam1, _lam2 = _top2_algebraic_n8(z, att.detach().float(), n_iter=16, eps=1e-6)
    rel_from_gap = (gap / lam1.clamp_min(1e-6)).clamp(0.0, 1.0)
    assert torch.allclose(pi, rel_from_gap, atol=2e-3)


def test_rel_merge_below_exclusive():
    grid = 6
    n_tokens = grid * grid
    z = torch.zeros(1, n_tokens, 3)
    z[..., 0] = 1.0
    a_ex = _mask_xy(grid, 0, 2, 0, 2)
    a_mer = a_ex + _mask_xy(grid, 4, 6, 4, 6)
    att_ex = torch.zeros(1, 2, n_tokens)
    att_ex[0, 0] = a_ex
    att_mer = torch.zeros(1, 2, n_tokens)
    att_mer[0, 0] = a_mer
    pi_ex = spectral_graph_n8_slot_purity(att_ex, z, divide_by_lambda1=True)
    pi_mer = spectral_graph_n8_slot_purity(att_mer, z, divide_by_lambda1=True)
    assert pi_mer[0, 0].item() < pi_ex[0, 0].item()


def test_processor_gate_is_n8_rel_pi():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    feat = torch.randn(2, 16, 5)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    out = processor(
        torch.randn(2, 3, 8),
        torch.randn(2, 16, 8),
        gate_p=None,
        default_idx=[],
        gate_form="purity_weight",
        conf_kind="spectral_graph_n8_rel",
        bind_features=feat,
        state_max_norm=True,
    )
    expected = spectral_graph_n8_slot_purity(att, feat, divide_by_lambda1=True)
    assert torch.allclose(out["active_mask"], expected, atol=1e-5)
    v39 = processor(
        torch.randn(2, 3, 8),
        torch.randn(2, 16, 8),
        gate_p=None,
        default_idx=[],
        gate_form="purity_weight",
        conf_kind="spectral_graph_n8",
        bind_features=feat,
    )
    assert not torch.allclose(out["active_mask"], v39["active_mask"], atol=1e-3)


def test_temporal_ema_hold_leaves_decoder_gate():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    feat = torch.randn(2, 16, 5)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    slots = torch.randn(2, 3, 8)
    inp = torch.randn(2, 16, 8)
    kwargs = dict(
        gate_p=None,
        default_idx=[],
        gate_form="purity_weight",
        conf_kind="spectral_graph_n8",
        bind_features=feat,
        state_max_norm=True,
    )
    base = processor(slots, inp, **kwargs)
    prev = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.1, 0.3]], dtype=base["active_mask"].dtype)
    ema = processor(slots, inp, state_gate_ema=0.5, state_gate_prev=prev, **kwargs)
    assert torch.allclose(ema["active_mask"], base["active_mask"], atol=1e-6)
    expected = 0.5 * base["active_mask"] + 0.5 * prev
    assert torch.allclose(ema["state_gate"], expected, atol=1e-5)
    hold = processor(slots, inp, state_gate_hold=0.9, state_gate_prev=prev, **kwargs)
    assert torch.allclose(hold["active_mask"], base["active_mask"], atol=1e-6)
    assert torch.allclose(
        hold["state_gate"], torch.maximum(base["active_mask"], 0.9 * prev), atol=1e-5
    )


def test_v39n_configs_parse_rel_n8():
    from slotcurri import configuration

    for path, name, n_slots in (
        ("configs/slotcurri/ytvis2021_attnmass_v39n.yaml", "ytvis_attnmass_v39n", 7),
        ("configs/slotcurri/movi_c_attnmass_v39n.yaml", "movi_c_attnmass_v39n", 11),
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
        assert amc["conf_kind"] == "spectral_graph_n8_rel"
        assert bool(amc.get("purity_normalize", False)) is False
        assert int(cfg.globals["NUM_SLOTS"]) == n_slots


def test_l1imp_matches_hand_formula_and_is_detached():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 4, 16), dim=1)
    att.requires_grad_(True)
    feat = torch.randn(2, 16, 6, requires_grad=True)
    pi = spectral_graph_n8_slot_purity(
        att, feat, chunk_size=1, l1_minus_imp=True
    )
    expected = _hand_pi_n8(att.detach(), feat.detach(), l1_minus_imp=True)
    assert torch.allclose(pi, expected, atol=2e-3)
    assert not pi.requires_grad
    assert (pi >= 0).all()


def test_l1imp_merge_below_exclusive():
    grid = 6
    n_tokens = grid * grid
    z = torch.zeros(1, n_tokens, 3)
    z[..., 0] = 1.0
    a_ex = _mask_xy(grid, 0, 2, 0, 2)
    a_mer = a_ex + _mask_xy(grid, 4, 6, 4, 6)
    att_ex = torch.zeros(1, 2, n_tokens)
    att_ex[0, 0] = a_ex
    att_mer = torch.zeros(1, 2, n_tokens)
    att_mer[0, 0] = a_mer
    pi_ex = spectral_graph_n8_slot_purity(att_ex, z, l1_minus_imp=True)
    pi_mer = spectral_graph_n8_slot_purity(att_mer, z, l1_minus_imp=True)
    assert pi_mer[0, 0].item() < pi_ex[0, 0].item()


def test_processor_gate_is_n8_l1imp_pi():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    feat = torch.randn(2, 16, 5)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    out = processor(
        torch.randn(2, 3, 8),
        torch.randn(2, 16, 8),
        gate_p=None,
        default_idx=[],
        gate_form="purity_weight",
        conf_kind="spectral_graph_n8_l1imp",
        bind_features=feat,
        state_max_norm=True,
    )
    expected = spectral_graph_n8_slot_purity(att, feat, l1_minus_imp=True)
    assert torch.allclose(out["active_mask"], expected, atol=1e-5)
    v39 = processor(
        torch.randn(2, 3, 8),
        torch.randn(2, 16, 8),
        gate_p=None,
        default_idx=[],
        gate_form="purity_weight",
        conf_kind="spectral_graph_n8",
        bind_features=feat,
    )
    assert not torch.allclose(out["active_mask"], v39["active_mask"], atol=1e-3)


def test_lam1_is_perron_power_pair():
    """v39lam1 uses q1^T G q1 from +ones power iteration, not k=8 Ritz λ1."""
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 4, 36), dim=1)
    feat = torch.randn(2, 36, 8)
    z = torch.nn.functional.normalize(feat.float(), dim=-1)
    a = att.float()
    lam1, lam2, q1 = _top2_algebraic_n8(
        z, a, n_iter=16, eps=1e-6, return_q1=True, lam1_only=True
    )
    assert torch.equal(lam2, torch.zeros_like(lam2))
    d_inv, weights = _n8_affinity(z, 1e-6)
    gq = _apply_g_n8(d_inv, weights, a, q1)
    rayleigh = (q1 * gq).sum(dim=2).squeeze(-1)
    assert torch.allclose(lam1, rayleigh, atol=1e-6)
    pi, lam1_out, lam2_out, q1_out = spectral_graph_n8_eigs(
        att, feat, use_lambda1=True, return_q1=True, chunk_size=1
    )
    assert torch.allclose(pi, lam1.clamp_min(0.0), atol=1e-6)
    assert torch.allclose(lam1_out, lam1, atol=1e-6)
    assert torch.equal(lam2_out, torch.zeros_like(lam2_out))
    assert torch.allclose(q1_out, q1.squeeze(-1), atol=1e-6)
    lam1_ritz, _ = _top2_algebraic_n8(z, a, n_iter=16, eps=1e-6, lam1_only=False)
    assert not torch.allclose(lam1, lam1_ritz, atol=0.0, rtol=0.0)


def test_lam1_matches_hand_formula_and_is_detached():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 4, 16), dim=1)
    att.requires_grad_(True)
    feat = torch.randn(2, 16, 6, requires_grad=True)
    pi = spectral_graph_n8_slot_purity(
        att, feat, chunk_size=1, use_lambda1=True
    )
    z = torch.nn.functional.normalize(feat.detach().float(), dim=-1)
    lam1, _ = _top2_algebraic_n8(
        z, att.detach().float(), n_iter=16, eps=1e-6, lam1_only=True
    )
    assert torch.allclose(pi, lam1.clamp_min(0.0).to(dtype=pi.dtype), atol=1e-6)
    # Power Rayleigh tracks dense λ1 closely enough on N=16 that occupancy
    # ranking is unchanged; the gate is not the k=8 Ritz value.
    expected_dense = _hand_pi_n8(att.detach(), feat.detach(), use_lambda1=True)
    assert torch.allclose(pi, expected_dense, atol=2e-2)
    assert not pi.requires_grad
    assert (pi >= 0).all()
    gap = spectral_graph_n8_slot_purity(att.detach(), feat.detach(), chunk_size=1)
    assert not torch.allclose(pi, gap, atol=1e-3)


def test_processor_gate_is_n8_lam1():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    feat = torch.randn(2, 16, 5)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    out = processor(
        torch.randn(2, 3, 8),
        torch.randn(2, 16, 8),
        gate_p=None,
        default_idx=[],
        gate_form="purity_weight",
        conf_kind="spectral_graph_n8_lam1",
        bind_features=feat,
        state_max_norm=True,
    )
    expected = spectral_graph_n8_slot_purity(att, feat, use_lambda1=True)
    assert torch.allclose(out["active_mask"], expected, atol=1e-5)
    v39 = processor(
        torch.randn(2, 3, 8),
        torch.randn(2, 16, 8),
        gate_p=None,
        default_idx=[],
        gate_form="purity_weight",
        conf_kind="spectral_graph_n8",
        bind_features=feat,
        state_max_norm=True,
    )
    assert not torch.allclose(out["active_mask"], v39["active_mask"], atol=1e-3)


def test_v39lam1_configs_parse():
    from slotcurri import configuration

    v39 = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v39.yaml")
    assert v39.model.attn_mass_curriculum["conf_kind"] == "spectral_graph_n8"
    assert str(v39.model.attn_mass_curriculum.get("state_conf_kind", "") or "") in (
        "",
        "pi",
    )

    for path, name, n_slots in (
        (
            "configs/slotcurri/ytvis2021_attnmass_v39lam1.yaml",
            "ytvis_attnmass_v39lam1",
            7,
        ),
        (
            "configs/slotcurri/movi_c_attnmass_v39lam1.yaml",
            "movi_c_attnmass_v39lam1",
            11,
        ),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        amc = cfg.model.attn_mass_curriculum
        assert amc["gate_form"] == "purity_weight"
        assert amc["conf_kind"] == "spectral_graph_n8_lam1"
        assert str(amc.get("state_conf_kind", "") or "") in ("", "pi")
        assert bool(amc.get("purity_normalize", False)) is False
        assert bool(amc["state_max_norm"]) is True
        assert int(cfg.globals["NUM_SLOTS"]) == n_slots
        fc = cfg.model.feature_curriculum
        assert bool(fc["enabled"]) is True
        assert fc["anneal"] == "ncut"
        assert fc["apply"] == "key"
        assert bool(fc["barrier"]) is False


def test_v39lam1g_configs_parse_predictor_src_gate():
    from slotcurri import configuration

    for path, name, n_slots in (
        (
            "configs/slotcurri/ytvis2021_attnmass_v39lam1g.yaml",
            "ytvis_attnmass_v39lam1g",
            7,
        ),
        (
            "configs/slotcurri/movi_c_attnmass_v39lam1g.yaml",
            "movi_c_attnmass_v39lam1g",
            11,
        ),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        amc = cfg.model.attn_mass_curriculum
        assert amc["gate_form"] == "purity_weight"
        assert amc["conf_kind"] == "spectral_graph_n8_lam1"
        assert bool(amc["predictor_src_gate"]) is True
        assert bool(amc.get("predictor_ungated", False)) is False
        assert bool(amc["state_max_norm"]) is True
        assert int(cfg.globals["NUM_SLOTS"]) == n_slots
        lam1 = configuration.load_config(path.replace("lam1g", "lam1"))
        assert bool(lam1.model.attn_mass_curriculum.get("predictor_src_gate", False)) is False


def test_v39lam1gu_configs_split_featrec():
    from slotcurri import configuration

    for path, name in (
        (
            "configs/slotcurri/ytvis2021_attnmass_v39lam1gu.yaml",
            "ytvis_attnmass_v39lam1gu",
        ),
        (
            "configs/slotcurri/movi_c_attnmass_v39lam1gu.yaml",
            "movi_c_attnmass_v39lam1gu",
        ),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        amc = cfg.model.attn_mass_curriculum
        assert bool(amc["predictor_src_gate"]) is True
        assert amc["conf_kind"] == "spectral_graph_n8_lam1"
        losses = cfg.model.losses
        assert "loss_featrec" in losses
        assert losses["loss_featrec_ungated"]["pred_key"] == "decoder.reconstruction_ungated"
        weights = cfg.model.loss_weights
        assert abs(float(weights["loss_featrec"]) - 1.0) < 1e-9
        assert abs(float(weights["loss_featrec_ungated"]) - 0.5) < 1e-9
        assert abs(float(weights["loss_ss"]) - 0.5) < 1e-9
        parent = configuration.load_config(path.replace("lam1gu", "lam1g"))
        assert "loss_featrec_ungated" not in parent.model.losses


def test_v39lam1gu_umix_configs_predictor_input_mix():
    from slotcurri import configuration

    for path, name in (
        (
            "configs/slotcurri/ytvis2021_attnmass_v39lam1gu_umix.yaml",
            "ytvis_attnmass_v39lam1gu_umix",
        ),
        (
            "configs/slotcurri/movi_c_attnmass_v39lam1gu_umix.yaml",
            "movi_c_attnmass_v39lam1gu_umix",
        ),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        amc = cfg.model.attn_mass_curriculum
        assert amc["conf_kind"] == "spectral_graph_n8_lam1"
        assert bool(amc["predictor_src_gate"]) is True
        assert bool(amc["predictor_input_mix"]) is True
        assert bool(amc["state_max_norm"]) is True
        assert "loss_featrec_ungated" in cfg.model.losses
        parent = configuration.load_config(path.replace("lam1gu_umix", "lam1gu"))
        assert bool(parent.model.attn_mass_curriculum.get("predictor_input_mix", False)) is False
        assert parent.model.attn_mass_curriculum["conf_kind"] == "spectral_graph_n8_lam1"


def test_v39lam1gu_umix_perron_configs():
    from slotcurri import configuration

    for path, name in (
        (
            "configs/slotcurri/ytvis2021_attnmass_v39lam1gu_umix_perron.yaml",
            "ytvis_attnmass_v39lam1gu_umix_perron",
        ),
        (
            "configs/slotcurri/movi_c_attnmass_v39lam1gu_umix_perron.yaml",
            "movi_c_attnmass_v39lam1gu_umix_perron",
        ),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        amc = cfg.model.attn_mass_curriculum
        assert amc["conf_kind"] == "spectral_graph_n8_lam1"
        assert bool(amc["predictor_input_mix"]) is True
        assert bool(amc["perron_readout"]) is True
        assert bool(amc.get("eval_perron_readout", False)) is False
        parent = configuration.load_config(path.replace("_perron", ""))
        assert bool(parent.model.attn_mass_curriculum.get("perron_readout", False)) is False
        assert bool(parent.model.attn_mass_curriculum["predictor_input_mix"]) is True


def test_v39lam1u_configs_split_featrec_no_src_gate():
    from slotcurri import configuration

    for path, name in (
        (
            "configs/slotcurri/ytvis2021_attnmass_v39lam1u.yaml",
            "ytvis_attnmass_v39lam1u",
        ),
        (
            "configs/slotcurri/movi_c_attnmass_v39lam1u.yaml",
            "movi_c_attnmass_v39lam1u",
        ),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        amc = cfg.model.attn_mass_curriculum
        assert amc["gate_form"] == "purity_weight"
        assert amc["conf_kind"] == "spectral_graph_n8_lam1"
        assert bool(amc["predictor_src_gate"]) is False
        assert bool(amc.get("predictor_ungated", False)) is False
        assert bool(amc["state_max_norm"]) is True
        losses = cfg.model.losses
        assert "loss_featrec" in losses
        assert losses["loss_featrec_ungated"]["pred_key"] == "decoder.reconstruction_ungated"
        weights = cfg.model.loss_weights
        assert abs(float(weights["loss_featrec"]) - 1.0) < 1e-9
        assert abs(float(weights["loss_featrec_ungated"]) - 0.5) < 1e-9
        assert abs(float(weights["loss_ss"]) - 0.5) < 1e-9
        lam1 = configuration.load_config(path.replace("lam1u", "lam1"))
        assert bool(lam1.model.attn_mass_curriculum.get("predictor_src_gate", False)) is False
        assert "loss_featrec_ungated" not in lam1.model.losses
        lam1gu = configuration.load_config(path.replace("lam1u", "lam1gu"))
        assert bool(lam1gu.model.attn_mass_curriculum["predictor_src_gate"]) is True
        assert "loss_featrec_ungated" in lam1gu.model.losses


def test_v39gu_configs_gap_plus_src_gate_split_featrec():
    from slotcurri import configuration

    for path, name in (
        (
            "configs/slotcurri/ytvis2021_attnmass_v39gu.yaml",
            "ytvis_attnmass_v39gu",
        ),
        (
            "configs/slotcurri/movi_c_attnmass_v39gu.yaml",
            "movi_c_attnmass_v39gu",
        ),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        amc = cfg.model.attn_mass_curriculum
        assert bool(amc["predictor_src_gate"]) is True
        assert amc["conf_kind"] == "spectral_graph_n8"
        losses = cfg.model.losses
        assert "loss_featrec" in losses
        assert losses["loss_featrec_ungated"]["pred_key"] == "decoder.reconstruction_ungated"
        weights = cfg.model.loss_weights
        assert abs(float(weights["loss_featrec"]) - 1.0) < 1e-9
        assert abs(float(weights["loss_featrec_ungated"]) - 0.5) < 1e-9
        assert abs(float(weights["loss_ss"]) - 0.5) < 1e-9
        parent = configuration.load_config(path.replace("v39gu", "v39"))
        assert parent.model.attn_mass_curriculum["conf_kind"] == "spectral_graph_n8"
        assert bool(parent.model.attn_mass_curriculum.get("predictor_src_gate", False)) is False
        assert "loss_featrec_ungated" not in parent.model.losses
        lam1gu = configuration.load_config(path.replace("v39gu", "v39lam1gu"))
        assert lam1gu.model.attn_mass_curriculum["conf_kind"] == "spectral_graph_n8_lam1"
        assert bool(lam1gu.model.attn_mass_curriculum["predictor_src_gate"]) is True
        assert "loss_featrec_ungated" in lam1gu.model.losses
        eabis = configuration.load_config(path.replace("v39gu", "v39lam1gu_eabis"))
        assert eabis.model.cyclic_inference == "evidence_sum"
        assert eabis.model.attn_mass_curriculum["conf_kind"] == "spectral_graph_n8_lam1"


def test_v39s_config_parses_l1imp():
    from slotcurri import configuration

    v39 = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v39.yaml")
    assert v39.model.attn_mass_curriculum["conf_kind"] == "spectral_graph_n8"

    v39s = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v39s.yaml")
    assert v39s.experiment_name == "ytvis_attnmass_v39s"
    amc = v39s.model.attn_mass_curriculum
    assert amc["gate_form"] == "purity_weight"
    assert amc["conf_kind"] == "spectral_graph_n8_l1imp"
    assert bool(amc.get("purity_normalize", False)) is False
    fc = v39s.model.feature_curriculum
    assert fc["anneal"] == "ncut"
    assert fc["apply"] == "key"
    assert bool(fc["barrier"]) is False
    assert float(amc.get("state_gate_ema", 1.0)) == 1.0


def test_v39sema_configs_parse_l1imp_and_temporal_ema():
    from slotcurri import configuration

    for path, name, n_slots in (
        ("configs/slotcurri/ytvis2021_attnmass_v39sema.yaml", "ytvis_attnmass_v39sema", 7),
        ("configs/slotcurri/movi_c_attnmass_v39sema.yaml", "movi_c_attnmass_v39sema", 11),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        amc = cfg.model.attn_mass_curriculum
        assert amc["gate_form"] == "purity_weight"
        assert amc["conf_kind"] == "spectral_graph_n8_l1imp"
        assert float(amc["state_gate_ema"]) == 0.2
        assert float(amc.get("state_gate_hold", 0.0)) == 0.0
        assert bool(amc["state_max_norm"]) is True
        assert bool(amc.get("state_identity_cos", False)) is False
        assert int(cfg.globals["NUM_SLOTS"]) == n_slots
        assert bool(cfg.model.cyclic_inference) is False
        fc = cfg.model.feature_curriculum
        assert fc["anneal"] == "ncut"
        assert fc["apply"] == "key"
        assert bool(fc["barrier"]) is False
