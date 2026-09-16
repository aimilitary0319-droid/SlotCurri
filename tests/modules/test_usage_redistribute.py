import torch

from slotcurri.modules.usage_redistribute import (
    reconstruction_usage_grad,
    simplex_redistribute,
    sparsemax,
    usage_gate_loss,
    usage_redistribute_direction,
)


def test_sparsemax_simplex_and_zeros():
    torch.manual_seed(0)
    logits = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.8, -3.0]])
    z = sparsemax(logits)
    assert torch.allclose(z.sum(-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(z[0], torch.full((3,), 1.0 / 3.0), atol=1e-6)
    assert z[1, 2] == 0.0
    assert z[1, 0] > z[1, 1] > 0.0


def test_sparsemax_backward():
    logits = torch.tensor([[1.0, 0.0, -2.0]], requires_grad=True)
    z = sparsemax(logits)
    z[0, 0].backward()
    assert logits.grad is not None
    # Inactive coordinate has zero sparsemax Jacobian.
    assert float(logits.grad[0, 2]) == 0.0


def test_simplex_redistribute_all_active_is_mean_center():
    r = torch.tensor([[1.0, 3.0, 5.0]])
    z = torch.ones(1, 3)
    v = simplex_redistribute(r, z)
    assert torch.allclose(v.sum(-1), torch.zeros(1), atol=1e-5)
    assert torch.allclose(v, r - r.mean(-1, keepdim=True), atol=1e-5)


def test_simplex_redistribute_revives_high_score_inactive():
    r = torch.tensor([[0.0, 10.0]])
    z = torch.tensor([[1.0, 0.0]])
    v = simplex_redistribute(r, z)
    assert torch.allclose(v.sum(-1), torch.zeros(1), atol=1e-5)
    assert v[0, 1] > 0.0
    assert v[0, 0] < 0.0


def test_simplex_redistribute_keeps_low_score_inactive_off():
    r = torch.tensor([[0.0, -1.0]])
    z = torch.tensor([[1.0, 0.0]])
    v = simplex_redistribute(r, z)
    assert torch.allclose(v, torch.zeros(1, 2), atol=1e-5)


def test_usage_gate_loss_gradient_is_neg_v():
    logits = torch.tensor([[0.2, -0.1, 0.4]], requires_grad=True)
    v = torch.tensor([[0.5, -0.2, -0.3]])
    loss = usage_gate_loss(logits, v)
    loss.backward()
    # L = 1/2 ||ℓ - sg(ℓ+v)||^2; one row so grad = -v.
    assert torch.allclose(logits.grad, -v, atol=1e-5)


def _mix_from_p(p, alpha, recon):
    w = p[:, :, None] * torch.exp(alpha)
    m = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return (m.unsqueeze(-1) * recon).sum(dim=1), torch.softmax(alpha, dim=1)


def test_reconstruction_usage_grad_matches_autograd():
    torch.manual_seed(0)
    b, s, f, d = 2, 3, 5, 4
    z = sparsemax(torch.randn(b, s))
    alpha = torch.randn(b, s, f)
    recon = torch.randn(b, s, f, d)
    target = torch.randn(b, f, d)
    mix, sm = _mix_from_p(z, alpha, recon)
    d_closed = reconstruction_usage_grad(mix, recon, sm, z, target)

    p = z.detach().clone().requires_grad_(True)
    mix_p, _ = _mix_from_p(p, alpha.detach(), recon.detach())
    # Closed d is ∂E_b/∂p; mean-over-batch autograd divides by B.
    e = (mix_p - target.detach()).pow(2).flatten(1).mean(-1).mean()
    e.backward()
    assert torch.allclose(d_closed, p.grad * b, atol=1e-4, rtol=1e-3)


def test_reconstruction_usage_grad_defined_at_zero_usage():
    b, s, f, d = 1, 2, 4, 3
    z = torch.tensor([[1.0, 0.0]])
    alpha = torch.zeros(b, s, f)
    recon = torch.randn(b, s, f, d)
    target = torch.randn(b, f, d)
    mix, sm = _mix_from_p(z, alpha, recon)
    d = reconstruction_usage_grad(mix, recon, sm, z, target)
    assert d.shape == (1, 2)
    assert torch.isfinite(d).all()


def test_usage_redistribute_direction_trains_logits_only():
    torch.manual_seed(0)
    b, s, f, d = 1, 3, 6, 2
    logits = torch.zeros(b, s, requires_grad=True)
    z = sparsemax(logits)
    alpha = torch.randn(b, s, f)
    recon = torch.randn(b, s, f, d)
    target = torch.randn(b, f, d)
    mix, sm = _mix_from_p(z.detach(), alpha, recon)
    mix.requires_grad_(True)
    loss, _, _, v = usage_redistribute_direction(
        mix, recon, sm, z.detach(), target, logits, lam=0.3
    )
    loss.backward()
    assert mix.grad is None or torch.allclose(mix.grad, torch.zeros_like(mix.grad))
    assert logits.grad is not None
    assert torch.allclose(logits.grad, -v, atol=1e-5)
    assert torch.allclose(v.sum(-1), torch.zeros(b), atol=1e-4)
