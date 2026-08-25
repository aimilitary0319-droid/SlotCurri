"""Tests for the evidence-aware (coverage x confidence) log-ratio gate and its
coupled activity curriculum (gate_form="logratio", p_anneal="cosine", beta_final).
"""
import math

import torch

from slotcurri.modules.video import LatentProcessor


BATCH, SLOTS, FEATS, DIM = 2, 4, 100, 8
EPS = 1e-6


class StubCorrector(torch.nn.Module):
    """Corrector returning fixed attention masks so the gate math is controllable."""

    def __init__(self, masks):
        super().__init__()
        self.masks_out = masks

    def forward(self, state, inputs, n_iters=None, **kwargs):
        return {"slots": state + 1.0, "masks": self.masks_out}


def _make_attention(requires_grad=False):
    """Softmax-over-slots attention (B, S, F) with distinct slot roles.

    slot 0: large object (owns feats 0..49)
    slot 1: small object, cleanly owned (feats 50..51)
    slot 2: diffuse ghost (uniform low logits, never wins a patch)
    slot 3: background (owns feats 52..99)
    """
    logits = torch.zeros(BATCH, SLOTS, FEATS)
    logits[:, 0, :50] = 6.0
    logits[:, 1, 50:52] = 6.0
    logits[:, 2, :] = 1.0
    logits[:, 3, 52:] = 6.0
    logits = logits + 0.01 * torch.randn(BATCH, SLOTS, FEATS)
    if requires_grad:
        logits.requires_grad_(True)
    return logits, torch.softmax(logits, dim=1)


def _sharpen(att, gamma):
    att_s = att.pow(gamma)
    return att_s / att_s.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _expected_conf(att_sharp):
    p_feat = att_sharp / att_sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    ent = -(p_feat * p_feat.clamp_min(1e-8).log()).sum(dim=-1)
    return (1.0 - ent / math.log(att_sharp.shape[-1])).clamp(min=0.0, max=1.0)


def _run_processor(att, **gate_kwargs):
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    state = torch.randn(BATCH, SLOTS, DIM)
    inputs = torch.randn(BATCH, FEATS, DIM)
    return processor(state, inputs, **gate_kwargs)


def test_linear_gate_unchanged():
    """Regression: gate_form='linear' reproduces sigmoid((m_sharp - p) / tau)."""
    _, att = _make_attention()
    gamma, p, tau = 2.0, 0.05, 0.01
    out = _run_processor(
        att, gate_p=p, default_idx=[], gate_mode="soft", gate_tau=tau, mass_gamma=gamma
    )
    mass = _sharpen(att, gamma).sum(dim=-1) / FEATS
    expected = torch.sigmoid((mass - p) / tau)
    assert torch.allclose(out["active_mask"], expected, atol=1e-6)
    assert "gate_conf" not in out  # confidence only computed for the logratio form


def test_logratio_gate_matches_formula():
    gamma, p, tau_g, beta = 2.0, 0.05, 0.5, 0.7
    _, att = _make_attention()
    out = _run_processor(
        att, gate_p=p, default_idx=[], gate_mode="soft", mass_gamma=gamma,
        gate_form="logratio", gate_beta=beta, gate_tau_log=tau_g,
    )
    att_sharp = _sharpen(att, gamma)
    mass = att_sharp.sum(dim=-1) / FEATS
    conf = _expected_conf(att_sharp)
    log_r = beta * (mass + EPS).log() + (1.0 - beta) * (conf + EPS).log()
    expected = torch.sigmoid((log_r - math.log(p + EPS)) / tau_g)
    assert torch.allclose(out["active_mask"], expected, atol=1e-6)
    assert torch.allclose(out["gate_conf"], conf, atol=1e-6)
    assert not out["gate_conf"].requires_grad  # sg(c)
    # gate is scale-invariant in (r, p): only the ratio to the threshold matters,
    # so r = p sits exactly at g = 0.5
    mid = torch.sigmoid(torch.zeros(())).item()
    assert abs(mid - 0.5) < 1e-6


def test_confidence_separates_small_object_from_ghost():
    """Small-but-cleanly-owned slot must out-gate the diffuse ghost under beta < 1."""
    gamma, p, tau_g, beta = 2.0, 1.5 / SLOTS, 0.5, 0.7
    _, att = _make_attention()
    out = _run_processor(
        att, gate_p=p, default_idx=[], gate_mode="soft", mass_gamma=gamma,
        gate_form="logratio", gate_beta=beta, gate_tau_log=tau_g,
    )
    conf = out["gate_conf"]
    gate = out["active_mask"]
    # slot 1 = small object (peaked), slot 2 = ghost (diffuse)
    assert (conf[:, 1] > conf[:, 2]).all()
    assert (gate[:, 1] > gate[:, 2]).all()


def test_logratio_mass_gradient_flows_conf_detached():
    """Gradient must reach the attention through the mass branch only."""
    logits, att = _make_attention(requires_grad=True)
    out = _run_processor(
        att, gate_p=0.05, default_idx=[], gate_mode="soft", mass_gamma=2.0,
        gate_form="logratio", gate_beta=0.7, gate_tau_log=0.5,
    )
    out["active_mask"].sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0.0


def test_logratio_state_mix_and_shared_gate():
    """Temporal mix uses the same log-ratio gate; corrector output stays ungated."""
    _, att = _make_attention()
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    state = torch.randn(BATCH, SLOTS, DIM)
    inputs = torch.randn(BATCH, FEATS, DIM)
    out = processor(
        state, inputs, gate_p=0.05, default_idx=[], gate_mode="soft", mass_gamma=2.0,
        gate_form="logratio", gate_beta=0.7, gate_tau_log=0.5, state_max_norm=True,
    )
    g = out["active_mask"]
    assert torch.equal(out["state_gate"], g)  # no gate_p_state -> shared gate
    # corrector output ungated
    assert torch.allclose(out["state"], state + 1.0)
    # mix: alpha * pred + (1 - alpha) * prior with alpha = g / max(g) (predictor=None ->
    # pred == corrected state)
    alpha = (g / g.amax(dim=-1, keepdim=True).clamp_min(1e-8)).unsqueeze(-1)
    expected = alpha * (state + 1.0) + (1.0 - alpha) * state
    assert torch.allclose(out["state_predicted"], expected, atol=1e-6)


def test_logratio_hard_and_ste_modes():
    _, att = _make_attention()
    p = 0.05
    hard = _run_processor(
        att, gate_p=p, default_idx=[], gate_mode="hard", mass_gamma=2.0,
        gate_form="logratio", gate_beta=1.0, gate_tau_log=0.5,
    )["active_mask"]
    assert hard.dtype == torch.bool
    # beta = 1 -> pure coverage: r >= p iff m >= p (log is monotone)
    mass = _sharpen(att, 2.0).sum(dim=-1) / FEATS
    assert torch.equal(hard, mass + EPS >= p + EPS)

    ste = _run_processor(
        att, gate_p=p, default_idx=[0], gate_mode="ste", mass_gamma=2.0,
        gate_form="logratio", gate_beta=0.7, gate_tau_log=0.5,
    )["active_mask"]
    assert ((ste == 0.0) | (ste == 1.0)).all()  # forward is binary
    assert (ste[:, 0] == 1.0).all()  # default slot forced on


def _expected_purity(att):
    return ((att * att).sum(-1) / att.sum(-1).clamp_min(1e-8)).clamp(0.0, 1.0)


def test_purity_weight_gate_is_the_statistic():
    """gate_form='purity_weight': the detached ownership purity IS the gate (v32).

    No threshold, no temperature: active_mask == sg(purity_sharp), shared with the
    temporal path.
    """
    _, att = _make_attention()
    out = _run_processor(
        att, gate_p=None, default_idx=[], mass_gamma=2.0,
        gate_form="purity_weight", conf_kind="purity_sharp",
    )
    g = out["active_mask"]
    expected = _expected_purity(_sharpen(att, 2.0))
    assert g.dtype != torch.bool
    assert torch.allclose(g, expected, atol=1e-6)
    assert torch.equal(out["state_gate"], g)
    assert not g.requires_grad  # sg(c): pure forward modulation, no gate gradient
    # ownership ordering: every real claimant out-gates the diffuse ghost, and the
    # small object is not penalized for its size (the entropy form's defect)
    for obj in (0, 1, 3):
        assert (g[:, obj] > g[:, 2]).all()
    assert (g[:, 1] > 0.9).all() and (g[:, 0] > 0.9).all()


def test_purity_weight_uniform_attention_is_noop():
    """Untrained-like uniform attention: near-uniform c cancels in both application
    points (decoder renorm / temporal max-norm), so early training is the baseline."""
    att = torch.full((BATCH, SLOTS, FEATS), 1.0 / SLOTS)
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    state = torch.randn(BATCH, SLOTS, DIM)
    inputs = torch.randn(BATCH, FEATS, DIM)
    out = processor(
        state, inputs, gate_p=None, default_idx=[], mass_gamma=2.0,
        gate_form="purity_weight", conf_kind="purity_sharp", state_max_norm=True,
    )
    g = out["active_mask"]
    assert torch.allclose(g, torch.full_like(g, 1.0 / SLOTS), atol=1e-6)
    # temporal mix: alpha = c / max(c) = 1 everywhere -> prediction passes through
    assert torch.allclose(out["state_predicted"], out["state"], atol=1e-6)


def test_purity_weight_state_mix_max_norm():
    """Temporal mix advances by alpha = c / max(c); corrector output stays ungated."""
    _, att = _make_attention()
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    state = torch.randn(BATCH, SLOTS, DIM)
    inputs = torch.randn(BATCH, FEATS, DIM)
    out = processor(
        state, inputs, gate_p=None, default_idx=[], mass_gamma=2.0,
        gate_form="purity_weight", conf_kind="purity_sharp", state_max_norm=True,
    )
    g = out["active_mask"]
    assert torch.allclose(out["state"], state + 1.0)  # corrector ungated
    alpha = (g / g.amax(dim=-1, keepdim=True).clamp_min(1e-8)).unsqueeze(-1)
    expected = alpha * (state + 1.0) + (1.0 - alpha) * state
    assert torch.allclose(out["state_predicted"], expected, atol=1e-6)


def test_purity_weight_raw_variant_and_default_slot():
    """conf_kind='purity' reads the RAW attention; default slots are forced fully on."""
    _, att = _make_attention()
    out = _run_processor(
        att, gate_p=None, default_idx=[2], mass_gamma=2.0,
        gate_form="purity_weight", conf_kind="purity",
    )
    g = out["active_mask"]
    expected = torch.maximum(
        _expected_purity(att), torch.tensor([0.0, 0.0, 1.0, 0.0]).expand(1, SLOTS)
    )
    assert torch.allclose(g, expected, atol=1e-6)
    assert (g[:, 2] == 1.0).all()  # ghost slot forced on as the default slot


def test_purity_weight_thresholdless_schedules():
    """The model exposes no threshold and no beta under purity_weight."""
    from slotcurri.models import ObjectCentricModel

    class Duck:
        _gate_threshold = ObjectCentricModel._gate_threshold
        _gate_beta = ObjectCentricModel._gate_beta
        _curriculum_lambda = ObjectCentricModel._curriculum_lambda

    duck = Duck()
    duck.attn_mass_enabled = True
    duck.amc_gate_form = "purity_weight"
    assert duck._gate_threshold(True) is None
    assert duck._gate_threshold(False) is None
    assert duck._gate_beta(True) is None


def test_purity_normalize_uniform_is_zero_and_exclusive_is_one():
    """v36: p = clip((K c - 1)/(K - 1), 0, 1) on detached purity_sharp."""
    att = torch.full((BATCH, SLOTS, FEATS), 1.0 / SLOTS)
    out = _run_processor(
        att, gate_p=None, default_idx=[], mass_gamma=2.0,
        gate_form="purity_weight", conf_kind="purity_sharp",
        purity_normalize=True,
    )
    assert torch.allclose(out["active_mask"], torch.zeros(BATCH, SLOTS), atol=1e-6)

    # one-hot exclusive owner on every patch
    onehot = torch.zeros(BATCH, SLOTS, FEATS)
    onehot[:, 0, :] = 1.0
    out_ex = _run_processor(
        onehot, gate_p=None, default_idx=[], mass_gamma=2.0,
        gate_form="purity_weight", conf_kind="purity_sharp",
        purity_normalize=True,
    )
    assert torch.allclose(out_ex["active_mask"][:, 0], torch.ones(BATCH), atol=1e-5)
    assert torch.allclose(out_ex["active_mask"][:, 1:], torch.zeros(BATCH, SLOTS - 1), atol=1e-5)


def test_purity_normalize_matches_formula_on_sharp_c():
    _, att = _make_attention()
    out = _run_processor(
        att, gate_p=None, default_idx=[], mass_gamma=2.0,
        gate_form="purity_weight", conf_kind="purity_sharp",
        purity_normalize=True,
    )
    c = _expected_purity(_sharpen(att, 2.0))
    expected = ((SLOTS * c - 1.0) / (SLOTS - 1.0)).clamp(0.0, 1.0)
    assert torch.allclose(out["active_mask"], expected, atol=1e-6)
    assert not out["active_mask"].requires_grad


def test_coupled_curriculum_schedules():
    """lambda drives p (cosine) and beta jointly; eval pins the curriculum end."""
    from slotcurri.models import ObjectCentricModel

    class Duck:
        # borrow the schedule methods; they only touch amc_* attrs and trainer.global_step
        _curriculum_lambda = ObjectCentricModel._curriculum_lambda
        _gate_beta = ObjectCentricModel._gate_beta
        _annealed_p_mult = ObjectCentricModel._annealed_p_mult

    duck = Duck()
    duck.attn_mass_enabled = True
    duck.amc_gate_form = "logratio"
    duck.amc_beta_start = 1.0
    duck.amc_beta_final = 0.7
    duck.amc_anneal_steps = 1000
    duck.amc_p_anneal = "cosine"
    duck.amc_p_start_mult = 1.5
    duck.amc_p_end_mult = 0.1
    duck.trainer = Duck()

    for step, lam_want in ((0, 0.0), (500, 0.5), (1000, 1.0), (5000, 1.0)):
        duck.trainer.global_step = step
        lam = duck._curriculum_lambda(True)
        assert abs(lam - lam_want) < 1e-9
        beta = duck._gate_beta(True)
        assert abs(beta - (1.0 - lam_want * 0.3)) < 1e-9
        p_mult = duck._annealed_p_mult(True)
        assert abs(p_mult - (1.5 + (0.1 - 1.5) * lam_want)) < 1e-9

    # eval convention: end of curriculum
    assert duck._curriculum_lambda(False) == 1.0
    assert abs(duck._gate_beta(False) - 0.7) < 1e-9
    assert abs(duck._annealed_p_mult(False) - 0.1) < 1e-9

    # uncoupled ablation (v26f): beta_start == beta_final holds beta fixed from step 0
    duck.amc_beta_start = 0.7
    for step in (0, 500, 1000):
        duck.trainer.global_step = step
        assert abs(duck._gate_beta(True) - 0.7) < 1e-9
    duck.amc_beta_start = 1.0

    # legacy linear form has no beta
    duck.amc_gate_form = "linear"
    assert duck._gate_beta(True) is None
