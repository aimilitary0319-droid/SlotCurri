import torch

from slotcurri.modules.decoders import algebraic_slot_utility_loss


def test_algebraic_duplicate_slots_have_zero_delta():
    b, t, s, f, d = 2, 1, 2, 4, 3
    target = torch.randn(b, t, f, d)
    slot_recons = target[:, :, None].expand(b, t, s, f, d).clone()
    masks = torch.full((b, t, s, f), 0.5)
    mix = (slot_recons * masks.unsqueeze(-1)).sum(dim=2)
    z = torch.full((b, t, s), 0.5, requires_grad=True)
    loss, rel_delta = algebraic_slot_utility_loss(
        mix, slot_recons, masks, target, z, margin=1.0, min_mask=0.0
    )
    assert torch.allclose(rel_delta, torch.zeros_like(rel_delta), atol=1e-5)
    assert torch.allclose(loss, torch.ones((), dtype=loss.dtype), atol=1e-5)
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0


def test_algebraic_unique_slot_is_exempt():
    b, t, f, d = 1, 1, 4, 2
    target = torch.randn(b, t, f, d)
    recon0 = target.clone()
    recon1 = target + 3.0
    slot_recons = torch.stack([recon0, recon1], dim=2)
    masks = torch.zeros(b, t, 2, f)
    masks[:, :, 0] = 1.0
    mix = recon0
    z = torch.tensor([[[0.8, 0.2]]], requires_grad=True)
    loss, rel_delta = algebraic_slot_utility_loss(
        mix, slot_recons, masks, target, z, margin=1.0, min_mask=0.01
    )
    assert rel_delta[0, 0, 0] > 1.0
    rent0 = (1.0 - rel_delta[0, 0, 0]).clamp(min=0.0, max=1.0)
    assert float(rent0) == 0.0
    # empty slot has Δ≈0 → rent≈1, leftover z still pays
    assert abs(float(loss) - 0.2) < 1e-5


def test_algebraic_rent_is_per_frame():
    """A slot unique in t=0 and duplicated in t=1 must get different rent."""
    b, t, s, f, d = 1, 2, 2, 4, 2
    target = torch.randn(b, t, f, d)
    recon0 = target.clone()
    recon1 = target.clone()
    recon1[:, 0] = target[:, 0] + 3.0
    slot_recons = torch.stack([recon0, recon1], dim=2)
    masks = torch.zeros(b, t, s, f)
    masks[:, 0, 0] = 1.0
    masks[:, 1] = 0.5
    mix = (slot_recons * masks.unsqueeze(-1)).sum(dim=2)
    z = torch.full((b, t, s), 0.5, requires_grad=True)
    loss, rel_delta = algebraic_slot_utility_loss(
        mix, slot_recons, masks, target, z, margin=1.0, min_mask=0.0
    )
    rent = (1.0 - rel_delta / 1.0).clamp(0.0, 1.0)
    assert rent[0, 0, 0] < rent[0, 1, 0]
    assert torch.allclose(rent[0, 1], torch.ones(s), atol=1e-4)
    loss.backward()
    assert z.grad is not None
    assert z.grad[0, 1].abs().sum() > z.grad[0, 0].abs().sum()


def _gated_mix(p, sm, slot_recons):
    denom = (p.unsqueeze(-1) * sm).sum(dim=2, keepdim=True).clamp_min(1e-8)
    m = p.unsqueeze(-1) * sm / denom
    mix = (m.unsqueeze(-1) * slot_recons).sum(dim=2)
    return mix, m


def test_algebraic_add_off_at_uniform():
    """z = 1/K ⇒ m = σ, so insert is identity and must not change c vs drop-only."""
    b, t, s, f, d = 1, 1, 2, 6, 3
    target = torch.randn(b, t, f, d)
    slot_recons = torch.randn(b, t, s, f, d)
    sm = torch.full((b, t, s, f), 0.5)
    z = torch.full((b, t, s), 0.5, requires_grad=True)
    mix, m = _gated_mix(z.detach(), sm, slot_recons)
    mix_u = (sm.unsqueeze(-1) * slot_recons).sum(dim=2)
    loss_drop, rel_drop = algebraic_slot_utility_loss(
        mix, slot_recons, m, target, z, margin=1.0
    )
    loss_add, rel_add_d, aux = algebraic_slot_utility_loss(
        mix,
        slot_recons,
        m,
        target,
        z,
        margin=1.0,
        masks_ungated=sm,
        mix_ungated=mix_u,
        add_insert=True,
        return_aux=True,
    )
    rent_drop = (1.0 - rel_drop).clamp(0.0, 1.0)
    assert torch.allclose(rel_drop, rel_add_d, atol=1e-5)
    assert torch.allclose(aux["rel_add"], torch.zeros_like(aux["rel_add"]), atol=1e-5)
    assert torch.allclose(aux["cost"], rent_drop, atol=1e-5)
    assert torch.allclose(loss_drop, loss_add, atol=1e-5)
    assert float(aux["suppressed"].sum()) == 0.0


def test_algebraic_add_negative_cost_on_dead_unique():
    """A near-off slot that uniquely matches high-error patches gets c < 0."""
    torch.manual_seed(0)
    b, t, s, f, d = 1, 1, 2, 8, 4
    target = torch.randn(b, t, f, d)
    recon0 = target.clone()
    recon0[:, :, 4:] = 0.0
    recon1 = target.clone()
    recon1[:, :, :4] = 0.0
    slot_recons = torch.stack([recon0, recon1], dim=2)
    sm = torch.zeros(b, t, s, f)
    sm[:, :, 0, :4] = 0.9
    sm[:, :, 1, :4] = 0.1
    sm[:, :, 0, 4:] = 0.1
    sm[:, :, 1, 4:] = 0.9
    z = torch.tensor([[[0.99, 0.01]]], requires_grad=True)
    mix, m = _gated_mix(z.detach(), sm, slot_recons)
    mix_u = (sm.unsqueeze(-1) * slot_recons).sum(dim=2)
    _loss_drop, rel_drop = algebraic_slot_utility_loss(
        mix, slot_recons, m, target, z, margin=1.0
    )
    loss, _rel, aux = algebraic_slot_utility_loss(
        mix,
        slot_recons,
        m,
        target,
        z,
        margin=1.0,
        masks_ungated=sm,
        mix_ungated=mix_u,
        add_insert=True,
        return_aux=True,
    )
    assert float(aux["suppressed"][0, 0, 0]) == 0.0
    assert float(aux["suppressed"][0, 0, 1]) == 1.0
    assert float(aux["rel_add"][0, 0, 1]) > 0.0
    assert float(aux["rel_add"][0, 0, 0]) <= float(aux["rel_add"][0, 0, 1])
    drop_rent1 = float((1.0 - rel_drop[0, 0, 1]).clamp(min=0.0, max=1.0))
    assert float(aux["cost"][0, 0, 1]) < drop_rent1
    assert float(aux["cost"][0, 0, 1]) < 0.0
    loss.backward()
    assert z.grad is not None
    assert float(z.grad[0, 0, 1]) < 0.0


def test_algebraic_drop_ignores_gate_when_ungated_mix_given():
    """v44: a near-off unique slot still has large Δ↓ on softmax(α) mix."""
    torch.manual_seed(1)
    b, t, s, f, d = 1, 1, 2, 8, 4
    target = torch.randn(b, t, f, d)
    recon0 = target.clone()
    recon1 = target + 3.0
    slot_recons = torch.stack([recon0, recon1], dim=2)
    sm = torch.zeros(b, t, s, f)
    sm[:, :, 0] = 0.7
    sm[:, :, 1] = 0.3
    z = torch.tensor([[[0.01, 0.99]]])
    mix_g, m = _gated_mix(z, sm, slot_recons)
    mix_u = (sm.unsqueeze(-1) * slot_recons).sum(dim=2)
    _, rel_gated = algebraic_slot_utility_loss(
        mix_g, slot_recons, m, target, z, margin=1.0
    )
    _, rel_ungated = algebraic_slot_utility_loss(
        mix_g,
        slot_recons,
        m,
        target,
        z,
        margin=1.0,
        masks_ungated=sm,
        mix_ungated=mix_u,
    )
    assert float(rel_gated[0, 0, 0]) < float(rel_ungated[0, 0, 0])
    assert float(rel_ungated[0, 0, 0]) > 1.0


def test_algebraic_psi_uniform_and_onehot():
    k = 7
    z_u = torch.full((2, 3, k), 1.0 / k)
    psi_u = (1.0 - z_u.pow(2).sum(-1)).mean()
    assert abs(float(psi_u) - (1.0 - 1.0 / k)) < 1e-6
    z_h = torch.zeros(2, 3, k)
    z_h[..., 0] = 1.0
    psi_h = (1.0 - z_h.pow(2).sum(-1)).mean()
    assert abs(float(psi_h)) < 1e-6


def test_algebraic_ce_zero_delta_pulls_toward_uniform():
    """u=0 → π=1/K. Peaked z should raise the winner logit (z-π > 0)."""
    b, t, s, f, d = 1, 1, 3, 4, 2
    target = torch.randn(b, t, f, d)
    slot_recons = target[:, :, None].expand(b, t, s, f, d).clone()
    masks = torch.full((b, t, s, f), 1.0 / s)
    mix = (slot_recons * masks.unsqueeze(-1)).sum(dim=2)
    logits = torch.tensor([[[2.0, 0.0, 0.0]]], requires_grad=True)
    z = torch.softmax(logits, dim=-1)
    loss, rel_delta, aux = algebraic_slot_utility_loss(
        mix,
        slot_recons,
        masks,
        target,
        z,
        teacher="ce",
        ce_tau=0.5,
        logits=logits,
        return_aux=True,
    )
    assert torch.allclose(rel_delta, torch.zeros_like(rel_delta), atol=1e-5)
    pi = aux["pi"]
    assert torch.allclose(pi, torch.full_like(pi, 1.0 / s), atol=1e-5)
    loss.backward()
    # ∂CE/∂ℓ = z-π; winner z > 1/K so its logit gradient is positive (push down).
    assert logits.grad is not None
    assert float(logits.grad[0, 0, 0]) > 0.0
    assert float(logits.grad[0, 0, 1]) < 0.0


def test_algebraic_ce_revives_unique_dead_slot():
    """Large ungated Δ↓ on a near-off slot → π_s > z_s → ℓ_s increases."""
    torch.manual_seed(1)
    b, t, s, f, d = 1, 1, 2, 8, 4
    target = torch.randn(b, t, f, d)
    recon0 = target.clone()
    recon1 = target + 3.0
    slot_recons = torch.stack([recon0, recon1], dim=2)
    sm = torch.zeros(b, t, s, f)
    sm[:, :, 0] = 0.7
    sm[:, :, 1] = 0.3
    logits = torch.tensor([[[-4.0, 4.0]]], requires_grad=True)
    z = torch.softmax(logits, dim=-1)
    mix_g, m = _gated_mix(z.detach(), sm, slot_recons)
    mix_u = (sm.unsqueeze(-1) * slot_recons).sum(dim=2)
    loss, rel_delta, aux = algebraic_slot_utility_loss(
        mix_g,
        slot_recons,
        m,
        target,
        z,
        masks_ungated=sm,
        mix_ungated=mix_u,
        add_insert=True,
        teacher="ce",
        ce_tau=0.5,
        logits=logits,
        return_aux=True,
    )
    assert float(rel_delta[0, 0, 0]) > 1.0
    assert float(aux["pi"][0, 0, 0]) > float(aux["pi"][0, 0, 1])
    loss.backward()
    # unique slot 0 is near-off in z; π_0 > z_0 ⇒ ∂L/∂ℓ_0 = z_0-π_0 < 0 (raise ℓ_0).
    assert float(logits.grad[0, 0, 0]) < 0.0
    assert float(loss) > 0.0


def test_algebraic_ce_matches_log_softmax():
    b, t, s, f, d = 1, 1, 2, 4, 2
    target = torch.randn(b, t, f, d)
    slot_recons = target[:, :, None].expand(b, t, s, f, d).clone()
    masks = torch.full((b, t, s, f), 0.5)
    mix = (slot_recons * masks.unsqueeze(-1)).sum(dim=2)
    logits = torch.tensor([[[0.3, -0.1]]], requires_grad=True)
    z = torch.softmax(logits, dim=-1)
    loss, _, aux = algebraic_slot_utility_loss(
        mix,
        slot_recons,
        masks,
        target,
        z,
        teacher="ce",
        ce_tau=0.5,
        logits=logits,
        return_aux=True,
    )
    log_z = torch.log_softmax(logits, dim=-1)
    expect = -(aux["pi"] * log_z).sum(dim=-1).mean()
    assert torch.allclose(loss, expect, atol=1e-6)
