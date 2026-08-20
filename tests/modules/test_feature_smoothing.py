"""Unit tests for the v33 feature curriculum.

Covers the FeatureSmoothing module (affinity math, blend, window mask, chunking),
the schedule function, and the FrameEncoder hook (train-only application).
"""

import torch
import torch.nn.functional as F
from torch import nn

from slotcurri.models import feature_curriculum_mix
from slotcurri.modules.encoders import FeatureSmoothing, FrameEncoder


def _two_cluster_tokens(n_per=8, dim=16, noise=0.1, seed=0):
    """Tokens in two orthogonal feature clusters: A ~ e0, B ~ e1 (plus noise)."""
    g = torch.Generator().manual_seed(seed)
    e0 = torch.zeros(dim)
    e0[0] = 1.0
    e1 = torch.zeros(dim)
    e1[1] = 1.0
    a = e0 + noise * torch.randn(n_per, dim, generator=g)
    b = e1 + noise * torch.randn(n_per, dim, generator=g)
    return torch.cat([a, b], dim=0).unsqueeze(0)  # (1, 2*n_per, dim)


def test_mix_schedule_endpoints_and_monotonicity():
    for schedule in ("cosine", "linear"):
        assert feature_curriculum_mix(0, 30000, schedule) == 0.0
        assert feature_curriculum_mix(30000, 30000, schedule) == 1.0
        assert feature_curriculum_mix(60000, 30000, schedule) == 1.0
        vals = [feature_curriculum_mix(s, 30000, schedule) for s in range(0, 40000, 500)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))
    # cosine passes through 1/2 at the midpoint; linear is the identity ramp
    assert abs(feature_curriculum_mix(15000, 30000, "cosine") - 0.5) < 1e-9
    assert abs(feature_curriculum_mix(7500, 30000, "linear") - 0.25) < 1e-9


def test_smoothing_homogenizes_within_clusters_and_keeps_them_apart():
    x = _two_cluster_tokens()
    out = FeatureSmoothing(tau=0.1)(x, 0.0)

    # within-cluster variance collapses toward the cluster mean
    for sl in (slice(0, 8), slice(8, 16)):
        var_before = x[0, sl].var(dim=0).mean()
        var_after = out[0, sl].var(dim=0).mean()
        assert var_after < 0.2 * var_before

    # cluster identities are preserved (means stay on their basis directions)
    e0 = torch.zeros(x.shape[-1])
    e0[0] = 1.0
    e1 = torch.zeros(x.shape[-1])
    e1[1] = 1.0
    cos_a = F.cosine_similarity(out[0, :8].mean(0), e0, dim=0)
    cos_b = F.cosine_similarity(out[0, 8:].mean(0), e1, dim=0)
    assert cos_a > 0.95 and cos_b > 0.95

    # smoothing must not import the other cluster's direction: the component along the
    # foreign basis stays at the raw cluster mean's level (the noise average that was
    # already there), i.e. cross-cluster affinity contributes ~nothing
    for sl, foreign in ((slice(0, 8), 1), (slice(8, 16), 0)):
        raw_component = x[0, sl, foreign].mean().abs()
        out_component = out[0, sl, foreign].mean().abs()
        assert out_component <= raw_component + 0.02


def test_mix_one_is_exact_noop():
    x = _two_cluster_tokens()
    sm = FeatureSmoothing(tau=0.1)
    assert sm(x, 1.0) is x


def test_blend_interpolates_between_smoothed_and_raw():
    x = _two_cluster_tokens()
    sm = FeatureSmoothing(tau=0.1)
    full = sm(x, 0.0)
    half = sm(x, 0.5)
    assert torch.allclose(half, 0.5 * full + 0.5 * x, atol=1e-6)


def test_chunking_does_not_change_the_result():
    x = _two_cluster_tokens().repeat(7, 1, 1) + 0.01 * torch.randn(7, 16, 16)
    a = FeatureSmoothing(tau=0.1, chunk_size=3)(x, 0.0)
    b = FeatureSmoothing(tau=0.1, chunk_size=100)(x, 0.0)
    assert torch.allclose(a, b, atol=1e-6)


def test_window_mask_structure():
    sm = FeatureSmoothing(tau=0.1, window=1)
    mask = sm._window_mask(16, torch.device("cpu"))  # 4x4 grid
    assert mask is not None
    assert not mask[0, 0]   # self
    assert not mask[0, 1]   # (0,1), Chebyshev 1
    assert not mask[0, 5]   # (1,1), Chebyshev 1
    assert mask[0, 2]       # (0,2), Chebyshev 2
    assert mask[0, 15]      # (3,3), Chebyshev 3
    # non-square token count: window falls back to global affinity
    assert sm._window_mask(15, torch.device("cpu")) is None


def test_window_blocks_distant_similar_tokens():
    # 4x4 grid: token 0 holds u, the far corner (3,3) holds u' (similar to u), everything
    # else is orthogonal filler. Global affinity pulls token 0 toward u'; a window of 1
    # must block that (the "two distant cows" case).
    dim = 8
    u = torch.zeros(dim)
    u[0] = 1.0
    z = torch.zeros(dim)
    z[1] = 1.0
    u_far = F.normalize(u + 0.3 * z, dim=0)
    filler = torch.zeros(dim)
    filler[2] = 1.0

    x = filler.repeat(16, 1)
    x[0] = u
    x[15] = u_far
    x = x.unsqueeze(0)

    global_out = FeatureSmoothing(tau=0.1)(x, 0.0)
    local_out = FeatureSmoothing(tau=0.1, window=1)(x, 0.0)

    cos_local = F.cosine_similarity(local_out[0, 0], u, dim=0)
    cos_global = F.cosine_similarity(global_out[0, 0], u, dim=0)
    assert cos_local > 0.999
    assert cos_global < 0.999
    # and the contamination is specifically along u_far's extra direction
    assert global_out[0, 0, 1] > 10 * local_out[0, 0, 1].abs()


class _TokenBackbone(nn.Module):
    """Backbone stub returning fixed (B, F, D) tokens regardless of the images."""

    def __init__(self, tokens):
        super().__init__()
        self.tokens = tokens

    def forward(self, images):
        return self.tokens.clone()


def test_frame_encoder_applies_smoothing_only_in_train_mode():
    tokens = _two_cluster_tokens().repeat(2, 1, 1)  # (2, 16, 16)
    enc = FrameEncoder(backbone=_TokenBackbone(tokens))
    enc.feature_smoothing = FeatureSmoothing(tau=0.1)
    enc.feature_smoothing_mix = 0.0
    images = torch.zeros(2, 3, 4, 4)

    enc.train()
    out_train = enc(images)
    assert not torch.allclose(out_train["backbone_features"], tokens)
    # grouper input and recon target derive from the same smoothed tensor
    assert torch.allclose(out_train["features"], out_train["backbone_features"])

    enc.eval()
    out_eval = enc(images)
    assert torch.equal(out_eval["backbone_features"], tokens)
    assert torch.equal(out_eval["features"], tokens)

    # mix = 1 is a no-op even in train mode
    enc.train()
    enc.feature_smoothing_mix = 1.0
    out_done = enc(images)
    assert torch.equal(out_done["backbone_features"], tokens)
