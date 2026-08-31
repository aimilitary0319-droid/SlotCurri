"""Unit tests for v40 slot-confidence entropy and 8-neighbor impurity losses."""

import math

import torch

from slotcurri.modules.video import (
    _N8_OFFSETS,
    LatentProcessor,
    ownership_confidence,
    slot_confidence_entropy,
    spectral_graph_n8_impurity,
    spectral_graph_n8_slot_purity,
)


BATCH, SLOTS, FEATS, DIM = 2, 4, 16, 8


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


def _n8_S_dense(z, eps=1e-6):
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


def _hand_rho(att, features, eps=1e-6):
    z = torch.nn.functional.normalize(features.float(), dim=-1)
    a = att.float()
    s_mat = _n8_S_dense(z, eps)
    out = []
    for s in range(a.shape[1]):
        aa = a[:, s]
        gram = aa.unsqueeze(-1) * s_mat * aa.unsqueeze(-2)
        gram = 0.5 * (gram + gram.transpose(-1, -2))
        evals = torch.linalg.eigvalsh(gram)
        lam1 = evals[:, -1].clamp_min(0.0)
        lam2 = evals[:, -2].clamp_min(0.0) if a.shape[-1] >= 2 else torch.zeros_like(lam1)
        out.append(lam2 / (lam1 + eps))
    return torch.stack(out, dim=-1)


def test_ownership_uniform_is_zero_exclusive_is_one():
    att = torch.full((BATCH, SLOTS, FEATS), 1.0 / SLOTS)
    c = ownership_confidence(att)
    assert torch.allclose(c, torch.zeros_like(c), atol=1e-6)

    onehot = torch.zeros(BATCH, SLOTS, FEATS)
    onehot[:, 0, :] = 1.0
    c_ex = ownership_confidence(onehot)
    assert torch.allclose(c_ex[:, 0], torch.ones(BATCH), atol=1e-5)
    assert torch.allclose(c_ex[:, 1:], torch.zeros(BATCH, SLOTS - 1), atol=1e-5)


def test_ownership_matches_purity_normalize_gate():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(BATCH, SLOTS, FEATS), dim=1)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    out = processor(
        torch.randn(BATCH, SLOTS, DIM),
        torch.randn(BATCH, FEATS, DIM),
        gate_p=None,
        default_idx=[],
        mass_gamma=1.0,
        gate_form="purity_weight",
        conf_kind="purity",
        purity_normalize=True,
    )
    expected = ownership_confidence(att)
    assert torch.allclose(out["active_mask"], expected, atol=1e-6)
    assert not out["active_mask"].requires_grad


def test_slot_entropy_onehot_zero_uniform_one():
    onehot = torch.zeros(BATCH, SLOTS)
    onehot[:, 0] = 1.0
    assert torch.allclose(slot_confidence_entropy(onehot), torch.zeros(BATCH), atol=1e-4)

    zeros = torch.zeros(BATCH, SLOTS)
    assert torch.allclose(slot_confidence_entropy(zeros), torch.ones(BATCH), atol=1e-5)

    uniform = torch.ones(BATCH, SLOTS)
    assert torch.allclose(slot_confidence_entropy(uniform), torch.ones(BATCH), atol=1e-5)


def test_ent_has_grad_through_attention_not_if_detached():
    torch.manual_seed(1)
    logits = torch.randn(BATCH, SLOTS, FEATS, requires_grad=True)
    att = torch.softmax(logits, dim=1)
    loss = slot_confidence_entropy(ownership_confidence(att)).mean()
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0

    logits2 = torch.randn(BATCH, SLOTS, FEATS, requires_grad=True)
    att2 = torch.softmax(logits2, dim=1).detach()
    loss2 = slot_confidence_entropy(ownership_confidence(att2)).mean()
    assert not loss2.requires_grad


def test_impurity_matches_hand_formula_and_grads_a_not_z():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    att.requires_grad_(True)
    feat = torch.randn(2, 16, 6, requires_grad=True)
    rho = spectral_graph_n8_impurity(att, feat, chunk_size=1)
    assert torch.allclose(rho, _hand_rho(att.detach(), feat.detach()), atol=3e-3)
    assert (rho >= 0).all()
    rho.mean().backward()
    assert att.grad is not None and att.grad.abs().sum() > 0
    assert feat.grad is None


def test_disconnected_merge_is_more_impure_than_exclusive():
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
    rho_ex = spectral_graph_n8_impurity(att_ex, z)
    rho_mer = spectral_graph_n8_impurity(att_mer, z)
    assert rho_mer[0, 0].item() > rho_ex[0, 0].item()
    pi_ex = spectral_graph_n8_slot_purity(att_ex, z)
    pi_mer = spectral_graph_n8_slot_purity(att_mer, z)
    assert pi_mer[0, 0].item() < pi_ex[0, 0].item()


def test_c_weighted_impurity_ignores_empty_slots():
    grid = 4
    n_tokens = grid * grid
    z = torch.zeros(1, n_tokens, 2)
    z[..., 0] = 1.0
    att = torch.zeros(1, 3, n_tokens)
    att[0, 0] = _mask_xy(grid, 0, 2, 0, 2)
    att[0, 0] = att[0, 0] + _mask_xy(grid, 2, 4, 2, 4)
    c = ownership_confidence(att)
    rho = spectral_graph_n8_impurity(att, z)
    weighted = (c.detach() * rho).sum(-1) / (c.detach().sum(-1) + 1e-6)
    assert torch.allclose(weighted, rho[:, 0], atol=1e-5)


def test_sei_weights_couple_at_endpoints_and_mid():
    from slotcurri.models import ObjectCentricModel

    class Duck:
        _sei_lambda = ObjectCentricModel._sei_lambda
        _sei_weights = ObjectCentricModel._sei_weights

    duck = Duck()
    duck.sei_anneal_steps = 50000
    duck.sei_weight = 0.2

    class Trainer:
        def __init__(self, step):
            self.global_step = step

    duck.trainer = Trainer(0)
    w_ent, w_imp, lam = duck._sei_weights(True)
    assert abs(lam) < 1e-9
    assert abs(w_ent - 0.2) < 1e-9
    assert abs(w_imp) < 1e-9

    duck.trainer = Trainer(50000)
    w_ent, w_imp, lam = duck._sei_weights(True)
    assert abs(lam - 1.0) < 1e-9
    assert abs(w_ent) < 1e-9
    assert abs(w_imp - 0.2) < 1e-9

    duck.trainer = Trainer(25000)
    w_ent, w_imp, lam = duck._sei_weights(True)
    assert abs(lam - 0.5) < 1e-9
    assert abs(w_ent + w_imp - 0.2) < 1e-9
    assert abs(w_ent - 0.1) < 1e-9

    w_ent_e, w_imp_e, lam_e = duck._sei_weights(False)
    assert abs(lam_e - 1.0) < 1e-9
    assert abs(w_imp_e - 0.2) < 1e-9


def test_v40_config_parses():
    from slotcurri import configuration

    cfg = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v40.yaml")
    assert cfg.experiment_name == "ytvis_attnmass_v40"
    assert bool(cfg.model.slot_expansion) is False
    assert bool(cfg.model.cyclic_inference) is False
    amc = cfg.model.attn_mass_curriculum
    assert amc["gate_form"] == "purity_weight"
    assert amc["conf_kind"] == "purity"
    assert bool(amc["purity_normalize"]) is True
    assert abs(float(amc["mass_gamma"]) - 1.0) < 1e-9
    sei = cfg.model.slot_ent_impurity
    assert bool(sei["enabled"]) is True
    assert abs(float(sei["weight"]) - 0.2) < 1e-9
    assert int(sei["anneal_steps"]) == 50000
    assert "feature_curriculum" not in cfg.model or cfg.model.feature_curriculum is None
