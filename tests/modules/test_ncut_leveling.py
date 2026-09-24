"""Unit tests for v36 Ncut relational leveling (Key-only curriculum)."""

import torch
import torch.nn.functional as F
from torch import nn

from slotcurri.modules.encoders import FrameEncoder, NcutRelationalLeveling
from slotcurri.modules.groupers import SlotAttention


def _two_cluster_tokens(n_per=8, dim=16, noise=0.02, seed=0):
    g = torch.Generator().manual_seed(seed)
    e0 = torch.zeros(dim)
    e0[0] = 1.0
    e1 = torch.zeros(dim)
    e1[1] = 1.0
    a = e0 + noise * torch.randn(n_per, dim, generator=g)
    b = e1 + noise * torch.randn(n_per, dim, generator=g)
    return torch.cat([a, b], dim=0).unsqueeze(0)  # (1, 2*n_per, dim)


def test_affinity_relu_and_zero_diag():
    x = _two_cluster_tokens()
    ncut = NcutRelationalLeveling(chunk_size=2)
    z = F.normalize(x.float(), dim=-1)
    w = torch.bmm(z, z.transpose(1, 2)).clamp_min(0)
    w.diagonal(dim1=-2, dim2=-1).zero_()
    # self-loops removed; cross-cluster cosine is ~0 so ReLU keeps the graph block-diag
    assert torch.allclose(w.diagonal(dim1=-2, dim2=-1), torch.zeros(1, 16))
    assert (w[0, :8, 8:].abs().mean() < 0.05)


def test_explain_matches_mix0():
    x = _two_cluster_tokens(noise=0.0)
    ncut = NcutRelationalLeveling(chunk_size=1)
    exp = ncut.explain(x)
    assert torch.allclose(exp["rel"], ncut(x, 0.0), atol=1e-5)
    assert exp["region"].shape == (1, 16)
    assert exp["v2"].shape == (1, 16)


def test_ncut_splits_two_clusters_and_levels_inside():
    # Zero within-cluster noise: W is block-diagonal, so the Fiedler/median cut
    # recovers the two components and each token is replaced by its partner mean.
    x = _two_cluster_tokens(noise=0.0)
    out = NcutRelationalLeveling(chunk_size=1)(x, 0.0)

    assert torch.allclose(out[0, :8], out[0, :8].mean(0), atol=1e-5)
    assert torch.allclose(out[0, 8:], out[0, 8:].mean(0), atol=1e-5)

    e0 = torch.zeros(x.shape[-1])
    e0[0] = 1.0
    e1 = torch.zeros(x.shape[-1])
    e1[1] = 1.0
    assert F.cosine_similarity(out[0, :8].mean(0), e0, dim=0) > 0.99
    assert F.cosine_similarity(out[0, 8:].mean(0), e1, dim=0) > 0.99
    assert F.cosine_similarity(out[0, 0], out[0, 8], dim=0) < 0.2


def test_ncut_under_cuda_amp():
    if not torch.cuda.is_available():
        return
    x = _two_cluster_tokens(noise=0.0).cuda()
    ncut = NcutRelationalLeveling(chunk_size=2).cuda()
    with torch.cuda.amp.autocast():
        out = ncut(x, 0.0)
    assert torch.isfinite(out.float()).all()
    assert out.shape == x.shape
    x = _two_cluster_tokens()
    ncut = NcutRelationalLeveling()
    assert ncut(x, 1.0) is x
    mixed = ncut(x, 0.5)
    full = ncut(x, 0.0)
    assert torch.allclose(mixed, 0.5 * full + 0.5 * x, atol=1e-5)


def test_ncut_chunking_matches():
    x = _two_cluster_tokens().repeat(5, 1, 1)
    a = NcutRelationalLeveling(chunk_size=2)(x, 0.0)
    b = NcutRelationalLeveling(chunk_size=16)(x, 0.0)
    assert torch.allclose(a, b, atol=1e-4)


def test_no_barrier_cuda_matches_cpu():
    if not torch.cuda.is_available():
        return
    x = _two_cluster_tokens(noise=0.0).cuda()
    out = NcutRelationalLeveling(chunk_size=2, barrier=False).cuda()(x, 0.0)
    assert torch.isfinite(out).all()
    assert out.shape == x.shape
    cpu = NcutRelationalLeveling(chunk_size=2, barrier=False)(x.cpu(), 0.0)
    assert torch.allclose(out.cpu(), cpu, atol=1e-4, rtol=1e-4)


def test_no_barrier_is_global_relu_cosine_leveling():
    """v37: skip Fiedler/median; P is row-normalize(W) on the full graph."""
    n, dim = 8, 4
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(1, n, dim)
    x[..., 0] = 1.0
    x = x + 0.05 * torch.randn(1, n, dim, generator=g)
    no_b = NcutRelationalLeveling(chunk_size=1, barrier=False)
    out = no_b(x, 0.0)
    assert float(out[0].var(0).mean()) < float(x[0].var(0).mean())
    exp = no_b.explain(x)
    assert bool(exp["region"].all())
    assert torch.allclose(exp["v2"], torch.zeros_like(exp["v2"]))
    exp_cut = NcutRelationalLeveling(chunk_size=1, barrier=True).explain(x)
    assert bool(exp_cut["region"].any()) and bool((~exp_cut["region"]).any())


def test_no_barrier_blockdiag_still_does_not_mix_orthogonal_clusters():
    # Cross-cluster ReLU cosine is 0, so global P is still block-diagonal.
    x = _two_cluster_tokens(noise=0.0)
    out = NcutRelationalLeveling(chunk_size=1, barrier=False)(x, 0.0)
    assert torch.allclose(out[0, :8], out[0, :8].mean(0), atol=1e-5)
    assert torch.allclose(out[0, 8:], out[0, 8:].mean(0), atol=1e-5)
    assert F.cosine_similarity(out[0, 0], out[0, 8], dim=0) < 0.2


def test_v36_configs_parse_ncut_key_and_purity_norm():
    from slotcurri import configuration

    for path, name, n_slots in (
        ("configs/slotcurri/ytvis2021_attnmass_v36.yaml", "ytvis_attnmass_v36", 7),
        ("configs/slotcurri/movi_c_attnmass_v36.yaml", "movi_c_attnmass_v36", 11),
    ):
        cfg = configuration.load_config(path)
        assert cfg.experiment_name == name
        fc = cfg.model.feature_curriculum
        assert fc["anneal"] == "ncut"
        assert fc["apply"] == "key"
        assert int(fc["anneal_steps"]) == 50000
        amc = cfg.model.attn_mass_curriculum
        assert amc["purity_normalize"] is True
        assert amc["conf_kind"] == "purity_sharp"
        assert abs(float(amc["mass_gamma"]) - 2.0) < 1e-9
        assert int(cfg.globals["NUM_SLOTS"]) == n_slots
    v33 = configuration.load_config("configs/slotcurri/ytvis2021_attnmass_v33.yaml")
    assert v33.model.feature_curriculum.get("anneal", "mix") in (None, "mix")
    assert not bool(v33.model.attn_mass_curriculum.get("purity_normalize", False))


class _TokenBackbone(nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.tokens = tokens

    def forward(self, images):
        return self.tokens.clone()


def test_frame_encoder_ncut_key_only_leaves_target_raw():
    tokens = _two_cluster_tokens().repeat(2, 1, 1)
    enc = FrameEncoder(backbone=_TokenBackbone(tokens), output_transform=nn.Identity())
    enc.feature_ncut = NcutRelationalLeveling(chunk_size=2)
    enc.feature_ncut_mix = 0.0
    enc.feature_curriculum_apply = "key"
    images = torch.zeros(2, 3, 4, 4)

    enc.train()
    out = enc(images)
    assert torch.equal(out["backbone_features"], tokens)
    assert "features_key" in out
    assert "backbone_key" in out
    assert not torch.allclose(out["features_key"], out["features"])
    # Value path is the original (Identity MLP)
    assert torch.allclose(out["features"], tokens)

    enc.eval()
    out_eval = enc(images)
    assert "features_key" not in out_eval
    assert torch.equal(out_eval["backbone_features"], tokens)

    enc.train()
    enc.feature_ncut_mix = 1.0
    out_done = enc(images)
    assert "features_key" not in out_done
    assert torch.equal(out_done["backbone_features"], tokens)


def test_n8_rejects_barrier():
    import pytest

    with pytest.raises(ValueError, match="n8=True cannot use barrier"):
        NcutRelationalLeveling(n8=True, barrier=True)


def test_n8_rejects_nonsquare():
    import pytest

    x = torch.randn(1, 12, 4)
    with pytest.raises(ValueError, match="square patch grid"):
        NcutRelationalLeveling(n8=True, barrier=False)(x, 0.0)


def test_n8_mix_identity_and_blend():
    x = torch.randn(2, 16, 8)
    n8 = NcutRelationalLeveling(n8=True, barrier=False, chunk_size=1)
    assert n8(x, 1.0) is x
    mixed = n8(x, 0.5)
    full = n8(x, 0.0)
    assert torch.allclose(mixed, 0.5 * full + 0.5 * x, atol=1e-5)
    exp = n8.explain(x)
    assert torch.allclose(exp["rel"], full, atol=1e-5)
    assert bool(exp["region"].all())
    assert torch.allclose(exp["v2"], torch.zeros_like(exp["v2"]))


def test_n8_does_not_mix_distant_same_feature_blobs():
    """Global P averages two distant e0 blobs; n8 levels each blob in place."""
    grid, dim = 8, 4
    x = torch.zeros(1, grid * grid, dim)
    x[..., 1] = 1.0  # background ~ e1
    e0 = torch.zeros(dim)
    e0[0] = 1.0
    # top-left 2x2 and bottom-right 2x2, same axis but different offset so
    # global ReLU-cosine still connects them while n8 cannot hop the gap.
    for y in (0, 1):
        for xx in (0, 1):
            x[0, y * grid + xx] = e0 + torch.tensor(
                [0.04 + 0.02 * y + 0.01 * xx, 0.0, 0.0, 0.0]
            )
    for y in (6, 7):
        for xx in (6, 7):
            x[0, y * grid + xx] = e0 + torch.tensor(
                [-0.04 - 0.02 * (7 - y) - 0.01 * (7 - xx), 0.0, 0.0, 0.0]
            )

    dense = NcutRelationalLeveling(chunk_size=1, barrier=False)(x, 0.0)
    n8 = NcutRelationalLeveling(n8=True, barrier=False, chunk_size=1)(x, 0.0)
    tl = 0 * grid + 0
    br = 6 * grid + 6
    dense_gap = float((dense[0, tl] - dense[0, br]).norm())
    n8_gap = float((n8[0, tl] - n8[0, br]).norm())
    assert n8_gap > dense_gap
    assert n8_gap > 0.02
    # n8 still levels inside a blob (neighbors mix).
    assert float((n8[0, 0] - n8[0, 1]).norm()) < float((x[0, 0] - x[0, 1]).norm())


def test_half_is_mean_of_global_and_n8():
    x = torch.randn(2, 16, 8)
    dense = NcutRelationalLeveling(chunk_size=1, barrier=False)(x, 0.0)
    n8 = NcutRelationalLeveling(n8=True, barrier=False, chunk_size=1)(x, 0.0)
    half = NcutRelationalLeveling(
        chunk_size=1, barrier=False, glob_n8_mix=0.5
    )(x, 0.0)
    assert torch.allclose(half, 0.5 * dense + 0.5 * n8, atol=1e-5)
    mixed = NcutRelationalLeveling(
        chunk_size=1, barrier=False, glob_n8_mix=0.5
    )(x, 0.5)
    assert torch.allclose(mixed, 0.5 * half + 0.5 * x, atol=1e-5)


def test_half_rejects_n8_flag_and_barrier():
    import pytest

    with pytest.raises(ValueError, match="glob_n8_mix>0 is the dense"):
        NcutRelationalLeveling(n8=True, barrier=False, glob_n8_mix=0.5)
    with pytest.raises(ValueError, match="cannot use barrier"):
        NcutRelationalLeveling(barrier=True, glob_n8_mix=0.5)


def test_row_center_matches_manual_formula():
    x = torch.randn(2, 16, 8)
    z = torch.nn.functional.normalize(x.float(), dim=-1)
    cos = torch.bmm(z, z.transpose(1, 2))
    relu_cos = cos.clamp_min(0.0)
    relu_cos.diagonal(dim1=-2, dim2=-1).zero_()
    tau = relu_cos.sum(dim=-1) / 15
    w = (cos - 0.5 * (tau.unsqueeze(-1) + tau.unsqueeze(-2))).clamp_min(0.0)
    w.diagonal(dim1=-2, dim2=-1).zero_()
    row = w.sum(dim=-1, keepdim=True)
    expected = torch.where(row < 1e-6, x.float(), torch.bmm(w, x.float()) / row.clamp_min(1e-6))
    out = NcutRelationalLeveling(
        chunk_size=1, barrier=False, row_center=True
    )(x, 0.0)
    assert torch.allclose(out, expected.to(dtype=x.dtype), atol=1e-5)
    mixed = NcutRelationalLeveling(
        chunk_size=1, barrier=False, row_center=True
    )(x, 0.5)
    assert torch.allclose(mixed, 0.5 * out + 0.5 * x, atol=1e-5)


def test_row_center_rejects_n8_barrier_and_half():
    import pytest

    with pytest.raises(ValueError, match="row_center=True is dense-graph"):
        NcutRelationalLeveling(n8=True, barrier=False, row_center=True)
    with pytest.raises(ValueError, match="cannot use barrier"):
        NcutRelationalLeveling(barrier=True, row_center=True)
    with pytest.raises(ValueError, match="cannot use glob_n8_mix"):
        NcutRelationalLeveling(barrier=False, glob_n8_mix=0.5, row_center=True)


def test_n8_isolated_token_kept():
    # Center token orthogonal to a ring of e0; ReLU cosine to all 8 nbrs is 0.
    grid, dim = 3, 2
    x = torch.zeros(1, grid * grid, dim)
    x[..., 0] = 1.0
    x[0, 4] = torch.tensor([0.0, 1.0])
    out = NcutRelationalLeveling(n8=True, barrier=False, chunk_size=1)(x, 0.0)
    assert torch.allclose(out[0, 4], x[0, 4], atol=1e-5)


def test_slot_attention_key_features_change_masks():
    inp_dim, slot_dim, n_patches, n_slots = 8, 8, 6, 3
    sa = SlotAttention(inp_dim, slot_dim, use_mlp=False)
    features = torch.randn(2, n_patches, inp_dim)
    key_features = torch.randn(2, n_patches, inp_dim)
    slots = torch.randn(2, n_slots, slot_dim)
    with torch.no_grad():
        shared = sa(slots, features)
        split = sa(slots, features, key_features=key_features)
    assert shared["masks"].shape == split["masks"].shape
    assert not torch.allclose(shared["masks"], split["masks"])
