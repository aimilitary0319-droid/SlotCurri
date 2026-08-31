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


def test_entropy_max_matches_product_of_entropy_and_peak():
    """conf_kind=entropy_max is sg(c_ent * max_f A) on raw softmax attention."""
    gamma, p, tau_g, beta = 2.0, 0.05, 0.5, 0.7
    _, att = _make_attention()
    out = _run_processor(
        att, gate_p=p, default_idx=[], gate_mode="soft", mass_gamma=gamma,
        gate_form="logratio", gate_beta=beta, gate_tau_log=tau_g,
        conf_kind="entropy_max",
    )
    att_sharp = _sharpen(att, gamma)
    c_ent = _expected_conf(att_sharp)
    peak = att.max(dim=-1).values.clamp(min=0.0, max=1.0)
    expected_c = (c_ent * peak).clamp(min=0.0, max=1.0)
    mass = att_sharp.sum(dim=-1) / FEATS
    log_r = beta * (mass + EPS).log() + (1.0 - beta) * (expected_c + EPS).log()
    expected_g = torch.sigmoid((log_r - math.log(p + EPS)) / tau_g)
    assert torch.allclose(out["gate_conf"], expected_c, atol=1e-6)
    assert torch.allclose(out["active_mask"], expected_g, atol=1e-6)
    assert not out["gate_conf"].requires_grad


def test_entropy_max_separates_always_second_when_entropy_ties():
    """Same spatial support, different height: entropy ties, entropy_max does not."""
    att = torch.zeros(2, 2, 10)
    att[:, 0, :] = 0.75
    att[:, 1, :] = 0.25
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    state = torch.randn(2, 2, DIM)
    inputs = torch.randn(2, 10, DIM)
    kwargs = dict(
        gate_p=0.05, default_idx=[], gate_mode="soft", mass_gamma=2.0,
        gate_form="logratio", gate_beta=0.7, gate_tau_log=0.5,
    )
    out_ent = processor(state, inputs, conf_kind="entropy", **kwargs)
    out_max = processor(state, inputs, conf_kind="entropy_max", **kwargs)
    # identical P after per-slot renorm -> identical entropy
    assert torch.allclose(out_ent["gate_conf"][:, 0], out_ent["gate_conf"][:, 1], atol=1e-5)
    # peak 0.75 vs 0.25 restores the height
    assert (out_max["gate_conf"][:, 0] > out_max["gate_conf"][:, 1]).all()
    ratio = out_max["gate_conf"][:, 0] / out_max["gate_conf"][:, 1].clamp_min(1e-8)
    assert torch.allclose(ratio, torch.full_like(ratio, 0.75 / 0.25), atol=1e-4)
    assert (out_max["active_mask"][:, 0] > out_max["active_mask"][:, 1]).all()


def test_entropy_max_p_end_zero_opens_mass_gate():
    """p<=0 is identity even with entropy_max (v26fmax leftover floor removed)."""
    _, att = _make_attention()
    out = _run_processor(
        att, gate_p=0.0, default_idx=[], gate_mode="soft", mass_gamma=2.0,
        gate_form="logratio", gate_beta=0.7, gate_tau_log=0.5,
        conf_kind="entropy_max",
    )
    assert torch.allclose(out["active_mask"], torch.ones_like(out["active_mask"]))
    # confidence is still computed (logged) even though the mass gate is open
    assert "gate_conf" in out
    assert out["gate_conf"].shape[1] == SLOTS


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


def test_p_end_mult_zero_cosine_opens_at_eval():
    """v26p: p_end_mult=0 reaches 0 at anneal_steps and stays 0 at eval."""
    from slotcurri.models import ObjectCentricModel

    class Duck:
        _curriculum_lambda = ObjectCentricModel._curriculum_lambda
        _annealed_p_mult = ObjectCentricModel._annealed_p_mult
        _gate_threshold = ObjectCentricModel._gate_threshold

    duck = Duck()
    duck.attn_mass_enabled = True
    duck.amc_gate_form = "logratio"
    duck.amc_p_anneal = "cosine"
    duck.amc_p_start_mult = 1.5
    duck.amc_p_end_mult = 0.0
    duck.amc_anneal_steps = 1000
    duck.n_slots = 7
    duck.amc_p_mode = "absolute"
    duck.trainer = Duck()

    duck.trainer.global_step = 0
    assert abs(duck._annealed_p_mult(True) - 1.5) < 1e-9
    duck.trainer.global_step = 1000
    assert abs(duck._annealed_p_mult(True) - 0.0) < 1e-9
    assert duck._gate_threshold(True) == 0.0
    assert duck._annealed_p_mult(False) == 0.0
    assert duck._gate_threshold(False) == 0.0


def test_state_gate_form_purity_splits_from_decoder_mass():
    """v26p: decoder is mass log-ratio; temporal mix is v36 normalized purity."""
    gamma, p, tau_g = 2.0, 1.5 / SLOTS, 0.5
    _, att = _make_attention()
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    state = torch.randn(BATCH, SLOTS, DIM)
    inputs = torch.randn(BATCH, FEATS, DIM)
    out = processor(
        state, inputs, gate_p=p, default_idx=[], gate_mode="soft", mass_gamma=gamma,
        gate_form="logratio", gate_beta=1.0, gate_tau_log=tau_g,
        state_gate_form="purity_weight", conf_kind="purity_sharp",
        purity_normalize=True, state_max_norm=True,
    )
    att_sharp = _sharpen(att, gamma)
    mass = att_sharp.sum(dim=-1) / FEATS
    expected_dec = torch.sigmoid(((mass + EPS).log() - math.log(p + EPS)) / tau_g)
    expected_state = ((SLOTS * _expected_purity(att_sharp) - 1.0) / (SLOTS - 1.0)).clamp(
        0.0, 1.0
    )

    assert torch.allclose(out["active_mask"], expected_dec, atol=1e-6)
    assert torch.allclose(out["state_gate"], expected_state, atol=1e-6)
    assert not torch.equal(out["active_mask"], out["state_gate"])
    assert not out["state_gate"].requires_grad
    assert torch.allclose(out["state"], state + 1.0)  # corrector ungated
    g = out["state_gate"]
    alpha = (g / g.amax(dim=-1, keepdim=True).clamp_min(1e-8)).unsqueeze(-1)
    expected_pred = alpha * (state + 1.0) + (1.0 - alpha) * state
    assert torch.allclose(out["state_predicted"], expected_pred, atol=1e-6)


def test_p_zero_opens_decoder_keeps_temporal_purity():
    """p_end_mult=0: decoder gate is identity; temporal purity stays selective."""
    _, att = _make_attention()
    out = _run_processor(
        att, gate_p=0.0, default_idx=[], gate_mode="soft", mass_gamma=2.0,
        gate_form="logratio", gate_beta=1.0, gate_tau_log=0.5,
        state_gate_form="purity_weight", conf_kind="purity_sharp",
        purity_normalize=True,
    )
    assert torch.allclose(out["active_mask"], torch.ones(BATCH, SLOTS), atol=1e-6)
    c = _expected_purity(_sharpen(att, 2.0))
    expected_state = ((SLOTS * c - 1.0) / (SLOTS - 1.0)).clamp(0.0, 1.0)
    assert torch.allclose(out["state_gate"], expected_state, atol=1e-6)
    assert (out["state_gate"].amax(dim=-1) - out["state_gate"].amin(dim=-1) > 0.1).all()


def test_p_zero_shared_gate_opens_both():
    """Without state_gate_form, p=0 opens decoder and temporal together."""
    _, att = _make_attention()
    out = _run_processor(
        att, gate_p=0.0, default_idx=[], gate_mode="soft", mass_gamma=2.0,
        gate_form="logratio", gate_beta=1.0, gate_tau_log=0.5,
    )
    ones = torch.ones(BATCH, SLOTS)
    assert torch.allclose(out["active_mask"], ones, atol=1e-6)
    assert torch.equal(out["state_gate"], out["active_mask"])


def test_state_mul_decoder_products_mass_and_purity():
    """v26pg: temporal mix is π ⊙ g_dec, then max-norm. Decoder stays mass-only."""
    gamma, p, tau_g = 2.0, 1.5 / SLOTS, 0.5
    _, att = _make_attention()
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    state = torch.randn(BATCH, SLOTS, DIM)
    inputs = torch.randn(BATCH, FEATS, DIM)
    out = processor(
        state, inputs, gate_p=p, default_idx=[], gate_mode="soft", mass_gamma=gamma,
        gate_form="logratio", gate_beta=1.0, gate_tau_log=tau_g,
        state_gate_form="purity_weight", conf_kind="purity_sharp",
        purity_normalize=True, state_max_norm=True, state_mul_decoder=True,
    )
    att_sharp = _sharpen(att, gamma)
    mass = att_sharp.sum(dim=-1) / FEATS
    g_dec = torch.sigmoid(((mass + EPS).log() - math.log(p + EPS)) / tau_g)
    pi = ((SLOTS * _expected_purity(att_sharp) - 1.0) / (SLOTS - 1.0)).clamp(0.0, 1.0)
    expected_state = g_dec * pi

    assert torch.allclose(out["active_mask"], g_dec, atol=1e-6)
    assert torch.allclose(out["state_gate"], expected_state, atol=1e-6)
    g = out["state_gate"]
    alpha = (g / g.amax(dim=-1, keepdim=True).clamp_min(1e-8)).unsqueeze(-1)
    expected_pred = alpha * (state + 1.0) + (1.0 - alpha) * state
    assert torch.allclose(out["state_predicted"], expected_pred, atol=1e-6)


def test_state_mul_decoder_p_zero_is_purity_only():
    """p=0 opens g_dec; product reduces to v26p temporal purity."""
    _, att = _make_attention()
    out = _run_processor(
        att, gate_p=0.0, default_idx=[], gate_mode="soft", mass_gamma=2.0,
        gate_form="logratio", gate_beta=1.0, gate_tau_log=0.5,
        state_gate_form="purity_weight", conf_kind="purity_sharp",
        purity_normalize=True, state_mul_decoder=True,
    )
    assert torch.allclose(out["active_mask"], torch.ones(BATCH, SLOTS), atol=1e-6)
    c = _expected_purity(_sharpen(att, 2.0))
    expected_state = ((SLOTS * c - 1.0) / (SLOTS - 1.0)).clamp(0.0, 1.0)
    assert torch.allclose(out["state_gate"], expected_state, atol=1e-6)


def test_decoder_mul_state_products_decoder_and_keeps_temporal_purity():
    """decoder_mul_state alone: decoder is g ⊙ π; temporal stays π."""
    gamma, p, tau_g = 2.0, 1.5 / SLOTS, 0.5
    _, att = _make_attention()
    out = _run_processor(
        att, gate_p=p, default_idx=[], gate_mode="soft", mass_gamma=gamma,
        gate_form="logratio", gate_beta=1.0, gate_tau_log=tau_g,
        state_gate_form="purity_weight", conf_kind="purity_sharp",
        purity_normalize=True, decoder_mul_state=True,
    )
    att_sharp = _sharpen(att, gamma)
    mass = att_sharp.sum(dim=-1) / FEATS
    g_dec = torch.sigmoid(((mass + EPS).log() - math.log(p + EPS)) / tau_g)
    pi = ((SLOTS * _expected_purity(att_sharp) - 1.0) / (SLOTS - 1.0)).clamp(0.0, 1.0)
    assert torch.allclose(out["active_mask"], g_dec * pi, atol=1e-6)
    assert torch.allclose(out["state_gate"], pi, atol=1e-6)


def test_decoder_mul_state_with_temporal_product_shares_gate():
    """v26pgd: decoder and temporal both see g ⊙ π (shared tensor)."""
    gamma, p, tau_g = 2.0, 1.5 / SLOTS, 0.5
    _, att = _make_attention()
    processor = LatentProcessor(StubCorrector(att), predictor=None)
    state = torch.randn(BATCH, SLOTS, DIM)
    inputs = torch.randn(BATCH, FEATS, DIM)
    out = processor(
        state, inputs, gate_p=p, default_idx=[], gate_mode="soft", mass_gamma=gamma,
        gate_form="logratio", gate_beta=1.0, gate_tau_log=tau_g,
        state_gate_form="purity_weight", conf_kind="purity_sharp",
        purity_normalize=True, state_max_norm=True,
        state_mul_decoder=True, decoder_mul_state=True,
    )
    att_sharp = _sharpen(att, gamma)
    mass = att_sharp.sum(dim=-1) / FEATS
    g_dec = torch.sigmoid(((mass + EPS).log() - math.log(p + EPS)) / tau_g)
    pi = ((SLOTS * _expected_purity(att_sharp) - 1.0) / (SLOTS - 1.0)).clamp(0.0, 1.0)
    expected = g_dec * pi
    assert torch.allclose(out["active_mask"], expected, atol=1e-6)
    assert torch.allclose(out["state_gate"], expected, atol=1e-6)
    g = out["state_gate"]
    alpha = (g / g.amax(dim=-1, keepdim=True).clamp_min(1e-8)).unsqueeze(-1)
    expected_pred = alpha * (state + 1.0) + (1.0 - alpha) * state
    assert torch.allclose(out["state_predicted"], expected_pred, atol=1e-6)


def test_decoder_mul_state_p_zero_is_purity_on_decoder():
    """p=0 opens g_dec; v26pgd decoder becomes π (not identity)."""
    _, att = _make_attention()
    out = _run_processor(
        att, gate_p=0.0, default_idx=[], gate_mode="soft", mass_gamma=2.0,
        gate_form="logratio", gate_beta=1.0, gate_tau_log=0.5,
        state_gate_form="purity_weight", conf_kind="purity_sharp",
        purity_normalize=True, state_mul_decoder=True, decoder_mul_state=True,
    )
    c = _expected_purity(_sharpen(att, 2.0))
    expected = ((SLOTS * c - 1.0) / (SLOTS - 1.0)).clamp(0.0, 1.0)
    assert torch.allclose(out["active_mask"], expected, atol=1e-6)
    assert torch.allclose(out["state_gate"], expected, atol=1e-6)


def test_v26pgd_config_multiplies_decoder_and_temporal_by_product():
    from slotcurri import configuration

    for path, name, n_slots in (
        ("configs/slotcurri/ytvis2021_attnmass_v26pgd.yaml", "ytvis_attnmass_v26pgd", 7),
        ("configs/slotcurri/movi_c_attnmass_v26pgd.yaml", "movi_c_attnmass_v26pgd", 11),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        amc = cfg.model.attn_mass_curriculum
        assert amc["gate_form"] == "logratio"
        assert amc["state_gate_form"] == "purity_weight"
        assert bool(amc["state_mul_decoder"]) is True
        assert bool(amc["decoder_mul_state"]) is True
        assert float(amc["p_end_mult"]) == 0.0
        assert float(amc["beta_final"]) == 1.0
        assert bool(amc["purity_normalize"]) is True
        assert int(cfg.globals.NUM_SLOTS) == n_slots
        assert cfg.model.get("feature_curriculum") is None
        assert bool(cfg.model.cyclic_inference) is False


def test_v26p_config_splits_decoder_mass_and_temporal_purity():
    from slotcurri import configuration

    for path, name, n_slots in (
        ("configs/slotcurri/ytvis2021_attnmass_v26p.yaml", "ytvis_attnmass_v26p", 7),
        ("configs/slotcurri/movi_c_attnmass_v26p.yaml", "movi_c_attnmass_v26p", 11),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        amc = cfg.model.attn_mass_curriculum
        assert amc["gate_form"] == "logratio"
        assert amc["state_gate_form"] == "purity_weight"
        assert amc["conf_kind"] == "purity_sharp"
        assert bool(amc["purity_normalize"]) is True
        assert float(amc["p_end_mult"]) == 0.0
        assert float(amc["beta_final"]) == 1.0
        assert float(amc["p_start_mult"]) == 1.5
        assert not bool(amc.get("state_mul_decoder", False))
        assert int(cfg.globals.NUM_SLOTS) == n_slots
        assert cfg.model.get("feature_curriculum") is None
        assert bool(cfg.model.cyclic_inference) is False


def test_v26pg_config_multiplies_temporal_by_decoder_gate():
    from slotcurri import configuration

    for path, name, n_slots in (
        ("configs/slotcurri/ytvis2021_attnmass_v26pg.yaml", "ytvis_attnmass_v26pg", 7),
        ("configs/slotcurri/movi_c_attnmass_v26pg.yaml", "movi_c_attnmass_v26pg", 11),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        amc = cfg.model.attn_mass_curriculum
        assert amc["state_gate_form"] == "purity_weight"
        assert bool(amc["state_mul_decoder"]) is True
        assert not bool(amc.get("decoder_mul_state", False))
        assert float(amc["p_end_mult"]) == 0.0
        assert int(cfg.globals.NUM_SLOTS) == n_slots


def test_v26fmax_config_entropy_max_and_p_end_zero():
    from slotcurri import configuration

    for path, name, n_slots in (
        ("configs/slotcurri/ytvis2021_attnmass_v26fmax.yaml", "ytvis_attnmass_v26fmax", 7),
        ("configs/slotcurri/movi_c_attnmass_v26fmax.yaml", "movi_c_attnmass_v26fmax", 11),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        amc = cfg.model.attn_mass_curriculum
        assert amc["gate_form"] == "logratio"
        assert amc["conf_kind"] == "entropy_max"
        assert float(amc["p_end_mult"]) == 0.0
        assert float(amc["p_start_mult"]) == 1.5
        assert float(amc["beta_start"]) == 1.0
        assert float(amc["beta_final"]) == 0.7
        assert not bool(amc.get("state_mul_decoder", False))
        assert not bool(amc.get("decoder_mul_state", False))
        assert int(cfg.globals.NUM_SLOTS) == n_slots
        assert cfg.model.get("feature_curriculum") is None
        assert bool(cfg.model.cyclic_inference) is False

