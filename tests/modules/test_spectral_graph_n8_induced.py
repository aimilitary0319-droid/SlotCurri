"""Induced n8 S_ind: two components ⇒ π≈0, one blob ⇒ π=μ2>0."""

import math

import pytest
import torch

from slotcurri.modules.video import (
    LatentProcessor,
    _N8_OFFSETS,
    spectral_graph_n8_induced_impurity,
    spectral_graph_n8_induced_purity,
    spectral_graph_n8_slot_purity,
)


class StubCorrector(torch.nn.Module):
    def __init__(self, masks):
        super().__init__()
        self.masks_out = masks

    def forward(self, state, inputs, n_iters=None, **kwargs):
        return {"slots": state + 1.0, "masks": self.masks_out}


def _mask_xy(grid, ys, ye, xs, xe):
    a = torch.zeros(grid * grid)
    for y in range(ys, ye):
        for x in range(xs, xe):
            a[y * grid + x] = 1.0
    return a


def _n8_R_dense(z, eps=1e-6):
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
    return rel


def _hand_induced_pi(att, features, support_rel=0.0, eps=1e-6):
    z = torch.nn.functional.normalize(features.float(), dim=-1)
    a = att.float()
    if support_rel > 0.0:
        thr = support_rel * a.amax(dim=-1, keepdim=True).clamp_min(eps)
        a = torch.where(a >= thr, a, torch.zeros_like(a))
    rel = _n8_R_dense(z, eps)
    out = []
    for s in range(a.shape[1]):
        aa = a[:, s]
        w = aa.unsqueeze(-1) * rel * aa.unsqueeze(-2)
        deg = w.sum(dim=-1)
        d_inv = torch.where(deg > eps, deg.rsqrt(), torch.zeros_like(deg))
        s_mat = d_inv.unsqueeze(-1) * w * d_inv.unsqueeze(-2)
        s_mat = 0.5 * (s_mat + s_mat.transpose(-1, -2))
        evals = torch.linalg.eigvalsh(s_mat)
        lam1 = evals[:, -1]
        lam2 = evals[:, -2] if a.shape[-1] >= 2 else torch.zeros_like(lam1)
        gap = (lam1 - lam2.clamp_min(0.0)).clamp_min(0.0)
        n_sup = (aa > eps).sum(dim=-1)
        pi = torch.where(n_sup <= 0, torch.zeros_like(gap), gap)
        pi = torch.where(n_sup == 1, torch.ones_like(pi), pi)
        out.append(pi)
    return torch.stack(out, dim=-1)


def test_matches_hand_formula_and_is_detached():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 4, 16), dim=1)
    att.requires_grad_(True)
    feat = torch.randn(2, 16, 6, requires_grad=True)
    pi = spectral_graph_n8_induced_purity(att, feat, chunk_size=1)
    assert torch.allclose(pi, _hand_induced_pi(att.detach(), feat.detach()), atol=3e-3)
    assert not pi.requires_grad
    assert (pi >= 0).all()


def test_disconnected_merge_pi_near_zero_exclusive_not():
    """Same features, two far 2x2 blobs: induced π≈0; exclusive blob π>0.

    v39 G_s gap stays large on both, so it cannot mark the merge.
    """
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
    pi_ex = spectral_graph_n8_induced_purity(att_ex, z)
    pi_mer = spectral_graph_n8_induced_purity(att_mer, z)
    assert pi_mer[0, 0].item() < 0.05
    assert pi_ex[0, 0].item() > 0.15
    v39_ex = spectral_graph_n8_slot_purity(att_ex, z)
    v39_mer = spectral_graph_n8_slot_purity(att_mer, z)
    assert (pi_ex[0, 0] - pi_mer[0, 0]).item() > (
        v39_ex[0, 0] - v39_mer[0, 0]
    ).item()


def test_touching_merge_stays_one_component():
    grid = 6
    n_tokens = grid * grid
    z = torch.zeros(1, n_tokens, 3)
    z[..., 0] = 1.0
    a_ex = _mask_xy(grid, 1, 3, 1, 3)
    a_touch = a_ex + _mask_xy(grid, 1, 3, 3, 5)
    att_ex = torch.zeros(1, 2, n_tokens)
    att_ex[0, 0] = a_ex
    att_t = torch.zeros(1, 2, n_tokens)
    att_t[0, 0] = a_touch
    pi_ex = spectral_graph_n8_induced_purity(att_ex, z)
    pi_t = spectral_graph_n8_induced_purity(att_t, z)
    assert pi_t[0, 0].item() > 0.1
    assert pi_ex[0, 0].item() > 0.1


def test_support_rel_cuts_leaky_background_bridge():
    grid = 6
    n_tokens = grid * grid
    z = torch.zeros(1, n_tokens, 3)
    z[..., 0] = 1.0
    a = 0.05 * torch.ones(n_tokens)
    a = a + 0.95 * _mask_xy(grid, 0, 2, 0, 2)
    a = a + 0.95 * _mask_xy(grid, 4, 6, 4, 6)
    att = torch.zeros(1, 2, n_tokens)
    att[0, 0] = a
    pi_full = spectral_graph_n8_induced_purity(att, z, support_rel=0.0)
    pi_cut = spectral_graph_n8_induced_purity(att, z, support_rel=0.25)
    assert pi_cut[0, 0].item() < 0.05
    assert pi_full[0, 0].item() > pi_cut[0, 0].item()


def test_singleton_is_pure_empty_is_zero():
    grid = 4
    n_tokens = grid * grid
    z = torch.zeros(1, n_tokens, 2)
    z[..., 0] = 1.0
    att = torch.zeros(1, 2, n_tokens)
    att[0, 0, 5] = 1.0
    pi = spectral_graph_n8_induced_purity(att, z)
    assert abs(pi[0, 0].item() - 1.0) < 1e-5
    assert abs(pi[0, 1].item()) < 1e-5


def test_induced_impurity_marks_disconnected_merge():
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
    rho_ex = spectral_graph_n8_induced_impurity(
        att_ex, z, support_rel=0.0, fiedler_tau=0.05
    )
    rho_mer = spectral_graph_n8_induced_impurity(
        att_mer, z, support_rel=0.0, fiedler_tau=0.05
    )
    assert rho_mer[0, 0].item() > 0.8
    assert rho_mer[0, 0].item() > rho_ex[0, 0].item() + 0.3


def test_impurity_grads_a_not_z():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    att.requires_grad_(True)
    feat = torch.randn(2, 16, 6, requires_grad=True)
    rho = spectral_graph_n8_induced_impurity(
        att, feat, chunk_size=1, support_rel=0.25
    )
    assert torch.isfinite(rho).all()
    rho.mean().backward()
    assert att.grad is not None
    assert torch.isfinite(att.grad).all()
    assert att.grad.abs().sum() > 0
    assert feat.grad is None


def test_processor_gate_is_induced_pi():
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
        conf_kind="spectral_graph_n8_ind",
        bind_features=feat,
        n8_support_rel=0.0,
        state_max_norm=True,
    )
    expected = spectral_graph_n8_induced_purity(att, feat, support_rel=0.0)
    assert torch.allclose(out["active_mask"], expected, atol=1e-5)


def test_v39ind_configs_parse():
    from slotcurri import configuration

    for path, name, n_slots in (
        ("configs/slotcurri/ytvis2021_attnmass_v39ind.yaml", "ytvis_attnmass_v39ind", 7),
        ("configs/slotcurri/movi_c_attnmass_v39ind.yaml", "movi_c_attnmass_v39ind", 11),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        fc = cfg.model.feature_curriculum
        assert bool(fc["enabled"]) is True
        assert bool(fc["barrier"]) is False
        amc = cfg.model.attn_mass_curriculum
        assert amc["gate_form"] == "purity_weight"
        assert amc["conf_kind"] == "purity"
        assert bool(amc["purity_normalize"]) is True
        sei = cfg.model.slot_ent_impurity
        assert bool(sei["enabled"]) is True
        assert sei["impurity_kind"] == "induced_n8"
        assert abs(float(sei["support_rel"]) - 0.25) < 1e-9
        assert int(cfg.globals["NUM_SLOTS"]) == n_slots


def test_non_square_n_raises():
    att = torch.softmax(torch.randn(1, 2, 12), dim=1)
    feat = torch.randn(1, 12, 4)
    with pytest.raises(ValueError, match="square patch grid"):
        spectral_graph_n8_induced_purity(att, feat)
