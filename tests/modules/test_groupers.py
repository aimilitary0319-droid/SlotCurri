import pytest
import torch

from slotcurri.modules import groupers


@pytest.mark.parametrize("use_mlp", [False, True])
def test_slot_attention(use_mlp):
    inp_dim, slot_dim, n_patches, n_slots = 5, 8, 6, 3
    slot_attention = groupers.SlotAttention(inp_dim, slot_dim, use_mlp=use_mlp)

    features = torch.randn(1, n_patches, inp_dim)
    slots = torch.randn(1, n_slots, slot_dim)

    with torch.no_grad():
        outp = slot_attention(slots, features)

    assert outp["slots"].shape == (1, n_slots, slot_dim)
    assert outp["masks"].shape == (1, n_slots, n_patches)


def test_sparsemax_ste_forward_and_softmax_grad():
    dots = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0], [-2.0, -2.0]]],
        requires_grad=True,
    )
    coeff = torch.tensor([[[1.0, -1.0], [0.5, 0.25], [-0.5, 1.0]]])
    weights = groupers.attention_over_slots(dots, "sparsemax_ste")
    with torch.no_grad():
        sparse = groupers.sparsemax(dots, dim=1)
        soft = torch.softmax(dots, dim=1)
    assert torch.allclose(weights, sparse)
    assert not torch.allclose(weights, soft)
    (weights * coeff).sum().backward()

    dots_soft = dots.detach().clone().requires_grad_(True)
    soft_live = torch.softmax(dots_soft, dim=1)
    (soft_live * coeff).sum().backward()
    assert torch.allclose(dots.grad, dots_soft.grad)


def test_entmax15_between_softmax_and_sparsemax():
    from slotcurri.modules.usage_redistribute import entmax15, sparsemax

    uniform = entmax15(torch.zeros(1, 2))
    assert torch.allclose(uniform, torch.full((1, 2), 0.5))

    logits = torch.tensor([[1.0, 0.0]])
    weights = entmax15(logits)
    soft = torch.softmax(logits, dim=-1)
    sparse = sparsemax(logits, dim=-1)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1))
    assert torch.allclose(weights, torch.tensor([[0.8306, 0.1694]]), atol=1e-3)
    assert weights[0, 1] < soft[0, 1]
    assert weights[0, 1] > sparse[0, 1]

    peaked = entmax15(torch.tensor([[2.0, 0.0]]))
    assert torch.allclose(peaked, torch.tensor([[1.0, 0.0]]), atol=1e-5)


def test_entmax15_backward_matches_finite_difference():
    from slotcurri.modules.usage_redistribute import entmax15

    logits = torch.tensor([[1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    weights = entmax15(logits)
    weights[0, 0].backward(retain_graph=True)
    grad = logits.grad[0].clone()
    logits.grad = None

    eps = 1e-6
    base = logits.detach()
    numeric = []
    for i in range(2):
        plus = base.clone()
        minus = base.clone()
        plus[0, i] += eps
        minus[0, i] -= eps
        numeric.append((entmax15(plus)[0, 0] - entmax15(minus)[0, 0]) / (2 * eps))
    assert torch.allclose(grad, torch.stack(numeric), atol=1e-4)
    assert grad.abs().sum() > 0

    peaked = torch.tensor([[3.0, 0.0]], requires_grad=True)
    entmax15(peaked)[0, 0].backward()
    assert torch.allclose(peaked.grad, torch.zeros_like(peaked), atol=1e-6)


def test_entmax15_attention_uses_entmax_grad():
    dots = torch.tensor([[[1.0, 0.0], [0.2, -0.3]]], requires_grad=True)
    coeff = torch.tensor([[[1.0, -0.5], [0.25, 0.75]]])
    weights = groupers.attention_over_slots(dots, "entmax15")
    (weights * coeff).sum().backward()

    dots_ref = dots.detach().clone().requires_grad_(True)
    (groupers.entmax15(dots_ref, dim=1) * coeff).sum().backward()
    assert torch.allclose(dots.grad, dots_ref.grad)

    dots_soft = dots.detach().clone().requires_grad_(True)
    (torch.softmax(dots_soft, dim=1) * coeff).sum().backward()
    assert not torch.allclose(dots.grad, dots_soft.grad, atol=1e-5)


def test_margin_sigmoid_is_best_other_and_does_not_sum_to_one():
    dots = torch.tensor([[[1.0], [0.4], [-0.2]]], requires_grad=True)
    weights = groupers.attention_over_slots(dots, "margin")
    expected = torch.sigmoid(
        torch.tensor([1.0 - 0.4, 0.4 - 1.0, -0.2 - 1.0])
    ).view(1, 3, 1)
    assert torch.allclose(weights, expected, atol=1e-6)
    assert not torch.allclose(weights.sum(dim=1), torch.ones(1, 1), atol=1e-3)

    # An extra lower slot does not change the existing scores.
    with_ghost = torch.tensor([[[1.0], [0.4], [-0.2], [-1.0]]])
    ghost_w = groupers.margin_sigmoid_over_slots(with_ghost)
    assert torch.allclose(ghost_w[:, :3], expected, atol=1e-6)

    weights[0, 0, 0].backward()
    # Leader score rises with its own dot, falls with the runner-up, and
    # ignores a strictly lower third slot.
    assert dots.grad[0, 0, 0] > 0
    assert dots.grad[0, 1, 0] < 0
    assert torch.allclose(dots.grad[0, 2, 0], torch.zeros(()))


def test_margin_update_follows_the_won_patch():
    # 10 tied patches score 0.5 on every slot. One patch is slot 0's unique win.
    scores = torch.full((1, 4, 11), 0.5)
    scores[0, 0, -1] = 1.0
    scores[0, 1:, -1] = 0.0
    raw = groupers.margin_update_weights(scores, 1e-8, slot_norm=False)
    # Without the slot sum a tie stays 0.5 and a win stays 1, so the ratio is 2.
    assert raw[0, 0, -1] / raw[0, 0, 0] == pytest.approx(2.0, rel=1e-4)

    weights = groupers.margin_update_weights(scores, 1e-8, slot_norm=True)
    # Slot sum turns a tie into 1/S and leaves the win at 1, so the patch mean
    # weights them S:1. S=4.
    assert weights[0, 0, -1] / weights[0, 0, 0] == pytest.approx(4.0, rel=1e-4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 4), atol=1e-5)
    # The losing slots put no mass on the patch they lost.
    assert weights[0, 1, -1] < 1e-6


def test_margin_gate_slot_norm_renorms_pi_only():
    torch.manual_seed(0)
    slots = torch.randn(2, 4, 4)
    features = torch.randn(2, 6, 4)
    base = groupers.SlotAttention(
        4, 4, n_iters=1, use_mlp=False, use_gru=False,
        attn_norm="margin", margin_slot_norm=True,
    )
    gated = groupers.SlotAttention(
        4, 4, n_iters=1, use_mlp=False, use_gru=False,
        attn_norm="margin", margin_slot_norm=True, margin_gate_slot_norm=True,
    )
    gated.load_state_dict(base.state_dict())
    out_b = base(slots, features)
    out_g = gated(slots, features)
    assert "pi_masks" not in out_b
    assert torch.allclose(out_g["slots"], out_b["slots"])
    assert torch.allclose(out_g["masks"], out_b["masks"])
    pi = out_g["pi_masks"]
    assert not pi.requires_grad
    expected = out_g["masks"].float()
    expected = expected / expected.sum(dim=1, keepdim=True).clamp_min(1e-8)
    assert torch.allclose(pi.float(), expected, atol=1e-5)
    assert torch.allclose(pi.float().sum(dim=1), torch.ones(2, 6), atol=1e-4)

    # A full tie is 0.5 before the slot sum and 1/S after it.
    tie = torch.full((1, 4, 3), 0.5)
    normed = tie / tie.sum(dim=1, keepdim=True)
    assert torch.allclose(normed, torch.full_like(tie, 0.25))


def test_margin_update_power_is_slot_count_invariant():
    # 10 ties at 0.5 and one win at 1. Squaring makes the win:tie ratio 4,
    # and an extra losing slot does not change the winner's patch weights.
    def scores(n_slots):
        out = torch.full((1, n_slots, 11), 0.5)
        out[0, 0, -1] = 1.0
        out[0, 1:, -1] = 0.0
        return out

    small = groupers.margin_update_weights(scores(4), 1e-8, power=2)
    large = groupers.margin_update_weights(scores(18), 1e-8, power=2)
    assert small[0, 0, -1] / small[0, 0, 0] == pytest.approx(4.0, rel=1e-4)
    assert large[0, 0, -1] / large[0, 0, 0] == pytest.approx(4.0, rel=1e-4)
    assert torch.allclose(small[0, 0], large[0, 0])

    torch.manual_seed(0)
    slots = torch.randn(1, 4, 4)
    features = torch.randn(1, 6, 4)
    raw = groupers.SlotAttention(
        4, 4, n_iters=1, use_mlp=False, use_gru=False, attn_norm="margin"
    )
    powered = groupers.SlotAttention(
        4,
        4,
        n_iters=1,
        use_mlp=False,
        use_gru=False,
        attn_norm="margin",
        margin_update_power=2,
    )
    powered.load_state_dict(raw.state_dict())
    out_raw = raw(slots, features)
    out_pow = powered(slots, features)
    assert torch.allclose(out_pow["masks"], out_raw["masks"])
    assert not torch.allclose(out_pow["slots"], out_raw["slots"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 autocast needs CUDA")
def test_margin_fp16_dead_slot_update_is_finite():
    # A slot that loses every patch saturates to 0 in fp16. eps=1e-8 also
    # flushes, so the patch mean used to be 0/0 and poisoned the slot state.
    torch.manual_seed(0)
    slot_attention = groupers.SlotAttention(
        8, 8, n_iters=2, use_mlp=True, use_gru=True, attn_norm="margin"
    ).cuda()
    slots = torch.randn(2, 18, 8, device="cuda")
    features = torch.randn(2, 24, 8, device="cuda")
    with torch.no_grad():
        slot_attention.to_q.weight.mul_(80)
        slot_attention.to_k.weight.mul_(80)
    with torch.cuda.amp.autocast():
        out = slot_attention(slots, features)
    assert torch.isfinite(out["slots"]).all()
    assert torch.isfinite(out["masks"]).all()
    # Losers really did flush in the returned fp16 map, so this is the 0-row case.
    assert (out["masks"].abs().sum(-1) == 0).any()


def test_pi_attn_norm_entmax_does_not_change_slot_update():
    torch.manual_seed(0)
    slots = torch.randn(2, 3, 4)
    features = torch.randn(2, 5, 4)
    soft = groupers.SlotAttention(4, 4, n_iters=1, use_mlp=False, use_gru=False)
    pi = groupers.SlotAttention(
        4, 4, n_iters=1, use_mlp=False, use_gru=False, pi_attn_norm="entmax15"
    )
    pi.load_state_dict(soft.state_dict())

    slots_s = slots.clone().requires_grad_(True)
    feats_s = features.clone().requires_grad_(True)
    out_s = soft(slots_s, feats_s)
    slots_p = slots.clone().requires_grad_(True)
    feats_p = features.clone().requires_grad_(True)
    out_p = pi(slots_p, feats_p)

    assert "pi_masks" not in out_s
    assert torch.allclose(out_p["slots"], out_s["slots"])
    assert torch.allclose(out_p["masks"], out_s["masks"])
    assert not out_p["pi_masks"].requires_grad

    slots_n = pi.norm_slots(slots)
    dots = torch.einsum(
        "bsd,bfd->bsf", pi.to_q(slots_n), pi.to_k(pi.norm_features(features))
    ) * pi.scale
    assert torch.allclose(out_p["masks"], torch.softmax(dots, dim=1))
    assert torch.allclose(out_p["pi_masks"], groupers.entmax15(dots, dim=1))
    assert not torch.allclose(out_p["pi_masks"], out_p["masks"], atol=1e-4)

    out_s["slots"].sum().backward()
    out_p["slots"].sum().backward()
    assert torch.allclose(feats_s.grad, feats_p.grad)
    assert torch.allclose(slots_s.grad, slots_p.grad)


def test_pi_attn_norm_margin_keeps_softmax_update():
    torch.manual_seed(0)
    slots = torch.randn(2, 4, 4)
    features = torch.randn(2, 6, 4)
    soft = groupers.SlotAttention(4, 4, n_iters=1, use_mlp=False, use_gru=False)
    pi = groupers.SlotAttention(
        4, 4, n_iters=1, use_mlp=False, use_gru=False, pi_attn_norm="margin"
    )
    pi.load_state_dict(soft.state_dict())

    slots_s = slots.clone().requires_grad_(True)
    feats_s = features.clone().requires_grad_(True)
    out_s = soft(slots_s, feats_s)
    slots_p = slots.clone().requires_grad_(True)
    feats_p = features.clone().requires_grad_(True)
    out_p = pi(slots_p, feats_p)

    assert torch.allclose(out_p["slots"], out_s["slots"])
    assert torch.allclose(out_p["masks"], out_s["masks"])
    assert not out_p["pi_masks"].requires_grad

    slots_n = pi.norm_slots(slots)
    dots = torch.einsum(
        "bsd,bfd->bsf", pi.to_q(slots_n), pi.to_k(pi.norm_features(features))
    ) * pi.scale
    assert torch.allclose(out_p["masks"], torch.softmax(dots, dim=1))
    assert torch.allclose(out_p["pi_masks"], groupers.margin_sigmoid_over_slots(dots))
    assert not torch.allclose(
        out_p["pi_masks"].sum(dim=1), torch.ones_like(out_p["masks"][:, :1]), atol=1e-3
    )

    out_s["slots"].sum().backward()
    out_p["slots"].sum().backward()
    assert torch.allclose(feats_s.grad, feats_p.grad)
    assert torch.allclose(slots_s.grad, slots_p.grad)


def test_dot_ktemp_keeps_scale_and_repeated_width():
    import math

    torch.manual_seed(0)
    q = torch.randn(1, 15, 8)
    k = torch.randn(1, 4, 8)
    logits = groupers.dot_ktemp_logits(q, k)
    mean = torch.einsum("bsd,bfd->bsf", q, k) / 8
    assert torch.allclose(logits, mean * math.log(14))
    assert torch.allclose(groupers.dot_ktemp_logits(q * 2, k * 3), logits * 6)
    logits_rep = groupers.dot_ktemp_logits(
        q.repeat(1, 1, 2), k.repeat(1, 1, 2)
    )
    assert torch.allclose(logits, logits_rep)
    logits_99 = groupers.dot_ktemp_logits(q, k, odds=99)
    assert torch.allclose(logits_99, mean * math.log(99 * 14))
    dot = torch.einsum("bsd,bfd->bsf", q, k)
    logits_sqrt = groupers.dot_ktemp_logits(q, k, sqrt_width=True)
    assert torch.allclose(logits_sqrt, dot * (math.log(14) / math.sqrt(8)))


def test_dot_ktemp_step_softmax_sums_over_slots():
    module = groupers.SlotAttention(
        8, 8, n_iters=1, use_mlp=False, use_gru=False, dot_ktemp=True
    )
    module.norm_slots = torch.nn.Identity()
    module.to_q = torch.nn.Identity()
    slots = torch.randn(2, 15, 8)
    keys = torch.randn(2, 5, 8)
    values = torch.randn(2, 5, 8)
    _, masks, pi_attn = module.step(slots, keys, values)
    expected = torch.softmax(groupers.dot_ktemp_logits(slots, keys), dim=1)
    assert torch.allclose(masks, expected)
    assert torch.allclose(masks.sum(dim=1), torch.ones(2, 1, 5))
    assert pi_attn is None

    module_99 = groupers.SlotAttention(
        8, 8, n_iters=1, use_mlp=False, use_gru=False, dot_ktemp=True, dot_ktemp_odds=99
    )
    module_99.norm_slots = torch.nn.Identity()
    module_99.to_q = torch.nn.Identity()
    _, masks_99, _ = module_99.step(slots, keys, values)
    expected_99 = torch.softmax(groupers.dot_ktemp_logits(slots, keys, odds=99), dim=1)
    assert torch.allclose(masks_99, expected_99)
