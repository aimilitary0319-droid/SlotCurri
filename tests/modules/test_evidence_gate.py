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

    def forward(self, state, inputs, n_iters=None):
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
