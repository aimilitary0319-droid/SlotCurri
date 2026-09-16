"""Unit tests for v40/v41 slot-entropy and impurity (n8 G_s and v37 C_s) losses."""

import math

import torch

from slotcurri.modules.video import (
    _N8_OFFSETS,
    LatentProcessor,
    SlotUsageHead,
    ownership_confidence,
    slot_confidence_entropy,
    spectral_cs_impurity,
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


def test_slot_entropy_raw_is_nats():
    onehot = torch.zeros(BATCH, SLOTS)
    onehot[:, 0] = 1.0
    assert torch.allclose(
        slot_confidence_entropy(onehot, normalize=False),
        torch.zeros(BATCH),
        atol=1e-4,
    )
    uniform = torch.ones(BATCH, SLOTS)
    h = slot_confidence_entropy(uniform, normalize=False)
    assert torch.allclose(h, torch.full((BATCH,), math.log(SLOTS)), atol=1e-5)
    h_norm = slot_confidence_entropy(uniform, normalize=True)
    assert torch.allclose(h / math.log(SLOTS), h_norm, atol=1e-6)


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


def _hand_cs_rho(att, features, eps=1e-6):
    z = torch.nn.functional.normalize(features.float(), dim=-1)
    a2 = att.float() * att.float()
    cmat = torch.einsum("bnd,bsn,bne->bsde", z, a2, z)
    cmat = 0.5 * (cmat + cmat.transpose(-1, -2))
    evals = torch.linalg.eigvalsh(cmat)
    lam1 = evals[..., -1].clamp_min(0.0)
    lam2 = evals[..., -2].clamp_min(0.0)
    return lam2 / (lam1 + eps)


def test_cs_impurity_matches_hand_formula_and_grads_a_not_z():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    att.requires_grad_(True)
    feat = torch.randn(2, 16, 6, requires_grad=True)
    rho = spectral_cs_impurity(att, feat)
    # k=2 subspace / Ritz vs full eigh of C_s; the loss only needs λ2/λ1.
    assert torch.allclose(rho, _hand_cs_rho(att.detach(), feat.detach()), atol=1e-3)
    assert (rho >= 0).all()
    assert (rho <= 1.0 + 1e-5).all()
    rho.mean().backward()
    assert att.grad is not None and att.grad.abs().sum() > 0
    assert feat.grad is None


def test_cs_rank1_is_pure_two_modes_are_impure():
    n_tokens = 8
    z = torch.zeros(1, n_tokens, 4)
    z[0, :4, 0] = 1.0
    z[0, 4:, 1] = 1.0
    att = torch.zeros(1, 2, n_tokens)
    att[0, 0] = 1.0
    att[0, 1, :4] = 1.0
    rho = spectral_cs_impurity(att, z)
    assert rho[0, 1].item() < 0.05
    assert rho[0, 0].item() > 0.9


def test_cs_empty_slot_rho_is_zero():
    att = torch.zeros(1, 2, 8)
    att[0, 0] = 1.0
    z = torch.zeros(1, 8, 4)
    z[..., 0] = 1.0
    rho = spectral_cs_impurity(att, z)
    assert rho[0, 1].item() < 1e-5
    assert rho[0, 0].item() < 0.05


def test_cs_proj_dim_keeps_live_a():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(2, 3, 16), dim=1)
    att.requires_grad_(True)
    feat = torch.randn(2, 16, 12)
    rho = spectral_cs_impurity(att, feat, proj_dim=8)
    assert rho.shape == (2, 3)
    rho.mean().backward()
    assert att.grad is not None and att.grad.abs().sum() > 0


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
    duck.sei_w_ent_start = 0.2
    duck.sei_w_ent_end = 0.0
    duck.sei_w_imp_start = 0.0
    duck.sei_w_imp_end = 0.2

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


def test_v41_sei_weights_floored_swap():
    from slotcurri.models import ObjectCentricModel

    class Duck:
        _sei_lambda = ObjectCentricModel._sei_lambda
        _sei_weights = ObjectCentricModel._sei_weights

    duck = Duck()
    duck.sei_anneal_steps = 50000
    duck.sei_w_ent_start = 0.3
    duck.sei_w_ent_end = 0.1
    duck.sei_w_imp_start = 0.1
    duck.sei_w_imp_end = 0.3

    class Trainer:
        def __init__(self, step):
            self.global_step = step

    duck.trainer = Trainer(0)
    w_ent, w_imp, lam = duck._sei_weights(True)
    assert abs(lam) < 1e-9
    assert abs(w_ent - 0.3) < 1e-9
    assert abs(w_imp - 0.1) < 1e-9

    duck.trainer = Trainer(50000)
    w_ent, w_imp, lam = duck._sei_weights(True)
    assert abs(lam - 1.0) < 1e-9
    assert abs(w_ent - 0.1) < 1e-9
    assert abs(w_imp - 0.3) < 1e-9

    duck.trainer = Trainer(25000)
    w_ent, w_imp, lam = duck._sei_weights(True)
    assert abs(lam - 0.5) < 1e-9
    assert abs(w_ent - 0.2) < 1e-9
    assert abs(w_imp - 0.2) < 1e-9
    assert abs(w_ent + w_imp - 0.4) < 1e-9


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


def test_usage_head_stopgrad_a_live_z():
    torch.manual_seed(0)
    head = SlotUsageHead(n_patches=FEATS, mlp_hidden=16, d_model=8, n_blocks=1, n_heads=2)
    att = torch.softmax(torch.randn(BATCH, SLOTS, FEATS), dim=1)
    att.requires_grad_(True)
    z = head(att)
    assert z.shape == (BATCH, SLOTS)
    assert (z > 0).all() and (z < 1).all()
    z.mean().backward()
    assert att.grad is None
    assert head.out.weight.grad is not None
    assert head.out.weight.grad.abs().sum() > 0


def test_usage_gate_is_live_and_matches_head():
    torch.manual_seed(0)
    att = torch.softmax(torch.randn(BATCH, SLOTS, FEATS), dim=1)
    head = SlotUsageHead(n_patches=FEATS, mlp_hidden=16, d_model=8, n_blocks=1, n_heads=2)
    processor = LatentProcessor(StubCorrector(att), predictor=None, usage_head=head)
    out = processor(
        torch.randn(BATCH, SLOTS, DIM),
        torch.randn(BATCH, FEATS, DIM),
        gate_p=None,
        default_idx=[],
        mass_gamma=1.0,
        gate_form="purity_weight",
        conf_kind="usage",
        purity_normalize=False,
        gate_detach=False,
    )
    z = head(att)
    assert torch.allclose(out["active_mask"], z, atol=1e-6)
    assert out["active_mask"].requires_grad
    assert out["gate_conf"].requires_grad


def test_usage_head_softmax_is_simplex():
    torch.manual_seed(0)
    head = SlotUsageHead(
        n_patches=FEATS, mlp_hidden=16, d_model=8, n_blocks=1, n_heads=2,
        normalize="softmax",
    )
    att = torch.softmax(torch.randn(BATCH, SLOTS, FEATS), dim=1)
    z = head(att)
    assert z.shape == (BATCH, SLOTS)
    assert torch.allclose(z.sum(-1), torch.ones(BATCH), atol=1e-6)
    assert (z > 0).all()
    z.sum().backward()
    assert head.out.weight.grad is not None
    assert head.out.weight.grad.abs().sum() > 0


def test_usage_head_sparsemax_can_be_sparse():
    torch.manual_seed(0)
    head = SlotUsageHead(
        n_patches=FEATS, mlp_hidden=16, d_model=8, n_blocks=1, n_heads=2,
        normalize="sparsemax",
    )
    att = torch.softmax(torch.randn(BATCH, SLOTS, FEATS), dim=1)
    logits, z = head.logits_and_usage(att)
    assert z.shape == (BATCH, SLOTS)
    assert torch.allclose(z.sum(-1), torch.ones(BATCH), atol=1e-5)
    assert logits.shape == z.shape
    z.sum().backward()
    assert head.out.weight.grad is not None


def test_v41_config_parses():
    from slotcurri import configuration

    cfg = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v41.yaml")
    assert cfg.experiment_name == "ytvis_attnmass_v41"
    assert bool(cfg.model.slot_expansion) is False
    assert bool(cfg.model.cyclic_inference) is False
    uh = cfg.model.usage_head
    assert bool(uh["enabled"]) is True
    assert int(uh["n_patches"]) == 1369
    assert int(uh["n_blocks"]) == 1
    assert bool(uh["stopgrad_attn"]) is True
    assert bool(uh["live_gate"]) is True
    amc = cfg.model.attn_mass_curriculum
    assert amc["gate_form"] == "purity_weight"
    assert amc["conf_kind"] == "usage"
    assert bool(amc["purity_normalize"]) is False
    assert bool(amc["gate_detach"]) is False
    assert bool(amc["state_max_norm"]) is True
    assert abs(float(amc["mass_gamma"]) - 1.0) < 1e-9
    sei = cfg.model.slot_ent_impurity
    assert bool(sei["enabled"]) is True
    assert sei["target"] == "z"
    assert sei["impurity_kind"] == "c_s"
    assert int(sei["proj_dim"]) == 64
    assert abs(float(sei["w_ent_start"]) - 0.3) < 1e-9
    assert abs(float(sei["w_ent_end"]) - 0.1) < 1e-9
    assert abs(float(sei["w_imp_start"]) - 0.1) < 1e-9
    assert abs(float(sei["w_imp_end"]) - 0.3) < 1e-9
    assert int(sei["anneal_steps"]) == 50000
    assert "feature_curriculum" not in cfg.model or cfg.model.feature_curriculum is None
    assert bool(amc["contrastive_gate"]) is False


def test_v42_config_parses():
    from slotcurri import configuration

    cfg = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v42.yaml")
    assert cfg.experiment_name == "ytvis_attnmass_v42"
    assert cfg.model.usage_head["normalize"] == "softmax"
    sei = cfg.model.slot_ent_impurity
    assert bool(sei["enabled"]) is True
    assert bool(sei["entropy_normalize"]) is False
    assert abs(float(sei["w_ent_start"]) - 0.05) < 1e-9
    assert abs(float(sei["w_ent_end"]) - 0.05) < 1e-9
    su = cfg.model.slot_utility
    assert abs(float(su["weight"]) - 0.2) < 1e-9
    assert su["mode"] == "algebraic"
    assert su["ramp"] == "none"
    lw = cfg.model.loss_weights
    assert abs(float(lw["loss_featrec"]) - 1.0) < 1e-9
    assert abs(float(lw["loss_ss"]) - 0.5) < 1e-9
    assert getattr(cfg.model, "slot_k_eff", None) is None


def test_v43_config_parses():
    from slotcurri import configuration

    cfg = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v43.yaml")
    assert cfg.experiment_name == "ytvis_attnmass_v43"
    assert cfg.model.usage_head["normalize"] == "sparsemax"
    assert bool(cfg.model.usage_head["live_gate"]) is False
    amc = cfg.model.attn_mass_curriculum
    assert bool(amc["gate_detach"]) is True
    assert bool(amc["predictor_src_gate"]) is True
    assert bool(amc["predictor_src_max_norm"]) is True
    ur = cfg.model.slot_usage_redistribute
    assert bool(ur["enabled"]) is True
    assert abs(float(ur["lambda"]) - 0.3) < 1e-9
    assert cfg.model.slot_ent_impurity is None
    assert cfg.model.slot_utility is None
    lw = cfg.model.loss_weights
    assert abs(float(lw["loss_featrec"]) - 0.5) < 1e-9
    assert abs(float(lw["loss_featrec_ungated"]) - 0.5) < 1e-9
    assert abs(float(lw["loss_ss"]) - 0.5) < 1e-9


def test_movi_c_v43_config_parses():
    from slotcurri import configuration

    cfg = configuration.load_config("configs/slotcurri/movi_c_attnmass_v43.yaml")
    assert cfg.experiment_name == "movi_c_attnmass_v43"
    assert cfg.model.usage_head["normalize"] == "sparsemax"
    assert abs(float(cfg.model.slot_usage_redistribute["lambda"]) - 0.3) < 1e-9
    assert int(cfg.globals.NUM_SLOTS) == 11
    assert int(cfg.model.usage_head["n_patches"]) == 576


def test_v42b_config_parses():
    from slotcurri import configuration

    cfg = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v42b.yaml")
    assert cfg.experiment_name == "ytvis_attnmass_v42b"
    assert cfg.model.usage_head["normalize"] == "softmax"
    assert cfg.model.slot_ent_impurity is None
    assert cfg.model.feature_curriculum is None
    su = cfg.model.slot_utility
    assert abs(float(su["weight"]) - 0.2) < 1e-9
    assert su["mode"] == "algebraic"


def test_movi_c_v42b_config_parses():
    from slotcurri import configuration

    cfg = configuration.load_config("configs/slotcurri/movi_c_attnmass_v42b.yaml")
    assert cfg.experiment_name == "movi_c_attnmass_v42b"
    assert int(cfg.globals.NUM_SLOTS) == 11
    assert int(cfg.globals.NUM_PATCHES) == 576
    assert cfg.model.usage_head["normalize"] == "softmax"
    assert int(cfg.model.usage_head["n_patches"]) == 576
    assert cfg.model.slot_ent_impurity is None
    assert cfg.model.feature_curriculum is None
    assert cfg.model.attn_mass_curriculum["conf_kind"] == "usage"
    assert cfg.model.slot_utility["mode"] == "algebraic"


def test_v41b_config_parses():
    from slotcurri import configuration

    cfg = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v41b.yaml")
    assert cfg.experiment_name == "ytvis_attnmass_v41b"
    amc = cfg.model.attn_mass_curriculum
    assert amc["conf_kind"] == "usage"
    assert bool(amc["contrastive_gate"]) is True
    assert bool(cfg.model.losses["loss_ss"]["gate_negatives"]) is False
    sei = cfg.model.slot_ent_impurity
    assert sei["target"] == "z"
    assert sei["impurity_kind"] == "c_s"


def test_v44_config_parses():
    from slotcurri import configuration

    cfg = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v44.yaml")
    assert cfg.experiment_name == "ytvis_attnmass_v44"
    assert cfg.model.usage_head["normalize"] == "softmax"
    assert bool(cfg.model.usage_head["live_gate"]) is False
    amc = cfg.model.attn_mass_curriculum
    assert bool(amc["gate_detach"]) is True
    assert bool(amc["predictor_src_gate"]) is True
    assert bool(amc["predictor_src_max_norm"]) is True
    assert cfg.model.slot_usage_redistribute is None
    assert cfg.model.slot_ent_impurity is None
    su = cfg.model.slot_utility
    assert abs(float(su["weight"]) - 0.2) < 1e-9
    assert su["mode"] == "algebraic"
    assert su["ramp"] == "none"
    assert str(su["add"]) == "insert"
    assert abs(float(su["psi_weight"]) - 0.05) < 1e-9
    lw = cfg.model.loss_weights
    assert abs(float(lw["loss_featrec"]) - 1.0) < 1e-9
    assert abs(float(lw["loss_featrec_ungated"]) - 0.5) < 1e-9
    assert abs(float(lw["loss_ss"]) - 0.5) < 1e-9


def test_movi_c_v44_config_parses():
    from slotcurri import configuration

    cfg = configuration.load_config("configs/slotcurri/movi_c_attnmass_v44.yaml")
    assert cfg.experiment_name == "movi_c_attnmass_v44"
    assert cfg.model.usage_head["normalize"] == "softmax"
    assert int(cfg.globals.NUM_SLOTS) == 11
    assert str(cfg.model.slot_utility["add"]) == "insert"
    assert abs(float(cfg.model.slot_utility["psi_weight"]) - 0.05) < 1e-9
    assert abs(float(cfg.model.loss_weights["loss_featrec"]) - 1.0) < 1e-9
    assert abs(float(cfg.model.loss_weights["loss_featrec_ungated"]) - 0.5) < 1e-9


def test_v45_config_parses():
    from slotcurri import configuration

    cfg = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v45.yaml")
    assert cfg.experiment_name == "ytvis_attnmass_v45"
    assert cfg.model.usage_head["normalize"] == "softmax"
    assert bool(cfg.model.usage_head["live_gate"]) is False
    amc = cfg.model.attn_mass_curriculum
    assert bool(amc["gate_detach"]) is True
    assert bool(amc["state_max_norm"]) is True
    assert bool(amc["predictor_src_gate"]) is True
    assert bool(amc["predictor_src_max_norm"]) is True
    assert cfg.model.slot_usage_redistribute is None
    assert cfg.model.slot_ent_impurity is None
    su = cfg.model.slot_utility
    assert abs(float(su["weight"]) - 0.2) < 1e-9
    assert su["mode"] == "algebraic"
    assert str(su["add"]) == "insert"
    assert str(su["teacher"]) == "ce"
    assert abs(float(su["ce_tau"]) - 0.5) < 1e-9
    assert abs(float(su["psi_weight"])) < 1e-9


def test_movi_c_v45_config_parses():
    from slotcurri import configuration

    cfg = configuration.load_config("configs/slotcurri/movi_c_attnmass_v45.yaml")
    assert cfg.experiment_name == "movi_c_attnmass_v45"
    assert int(cfg.globals.NUM_SLOTS) == 11
    su = cfg.model.slot_utility
    assert str(su["teacher"]) == "ce"
    assert abs(float(su["ce_tau"]) - 0.5) < 1e-9
    assert abs(float(su["psi_weight"])) < 1e-9
