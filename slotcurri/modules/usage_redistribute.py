"""Reconstruction-based slot-usage redistribution (v43).

Usage z = sparsemax(ℓ). The usage head is trained to follow the simplex-
projected descent of

    J(p) = E(p) + (λ/2) (1 - ||p||^2)

which matches the document's concentration prose and the canonical
r = λz - d, not the minus sign in section 3.1. E is DINO-feature MSE of
the usage-weighted decoder mix. Gradients of E w.r.t. p are detached;
only ℓ receives L_gate.
"""

from typing import Tuple

import torch


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax (Martins & Astudillo, 2016). Output sums to 1; some entries are 0."""
    if dim != -1:
        logits = logits.transpose(dim, -1)
    original = logits.shape
    dim_size = original[-1]
    flat = logits.reshape(-1, dim_size)
    zs = torch.sort(flat, dim=-1, descending=True).values
    k_idx = torch.arange(1, dim_size + 1, device=flat.device, dtype=flat.dtype)
    cssv = zs.cumsum(dim=-1) - 1.0
    support = zs - cssv / k_idx.unsqueeze(0) > 0
    k = support.sum(dim=-1, keepdim=True).clamp_min(1)
    tau = cssv.gather(-1, k.long() - 1) / k
    out = (flat - tau).clamp_min(0.0)
    out = out.reshape(original)
    if dim != -1:
        out = out.transpose(dim, -1)
    return out


def reconstruction_usage_grad(
    mix: torch.Tensor,
    slot_recons: torch.Tensor,
    masks_ungated: torch.Tensor,
    gate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """d_s = ∂E/∂p_s at p = sg(z). Defined for z_s = 0.

    E = mean_{f,c} ||ŷ_f(p) - F_f||^2,
    d_s = (2/NC) Σ_f [exp(α_{s,f}) / Σ_j z_j exp(α_{j,f})]
          ⟨ŷ_f - F_f, R_{s,f} - ŷ_f⟩

    mix (B, [T,] F, D), slot_recons (B, [T,] S, F, D),
    masks_ungated = softmax_s(α) (B, [T,] S, F), gate (B, [T,] S),
    target like mix. Returns d (B, [T,] S). Detached.
    """
    squeeze_t = mix.ndim == 3
    if squeeze_t:
        mix = mix.unsqueeze(1)
        target = target.unsqueeze(1)
        slot_recons = slot_recons.unsqueeze(1)
        masks_ungated = masks_ungated.unsqueeze(1)
        if gate.ndim == 2:
            gate = gate.unsqueeze(1)
    if mix.ndim != 4 or slot_recons.ndim != 5 or masks_ungated.ndim != 4:
        raise ValueError(
            "reconstruction_usage_grad expected mix (B,T,F,D), "
            f"got mix {tuple(mix.shape)}, slot_recons {tuple(slot_recons.shape)}, "
            f"masks_ungated {tuple(masks_ungated.shape)}"
        )
    if gate.ndim != 3:
        raise ValueError(
            f"reconstruction_usage_grad expected gate (B,T,S), got {tuple(gate.shape)}"
        )

    with torch.no_grad():
        z = gate.float()
        sm = masks_ungated.float()
        mix_f = mix.float()
        tgt = target.float()
        n_slots = sm.shape[2]
        denom = (z.unsqueeze(-1) * sm).sum(dim=2, keepdim=True).clamp_min(eps)
        err = mix_f - tgt
        parts = []
        for s in range(n_slots):
            w_s = sm[:, :, s] / denom.squeeze(2)
            delta = slot_recons[:, :, s].float() - mix_f
            inner = (err * delta).mean(dim=-1)
            parts.append(2.0 * (w_s * inner).mean(dim=-1))
        d = torch.stack(parts, dim=-1)
    if squeeze_t:
        d = d.squeeze(1)
    return d.to(dtype=mix.dtype)


def simplex_redistribute(
    r: torch.Tensor,
    z: torch.Tensor,
    z_eps: float = 0.0,
) -> torch.Tensor:
    """Projected step v with Σ v = 0. Active: r-τ; inactive: [r-τ]+.

    τ is solved so the support of v is active ∪ {inactive: r > τ}.
    All-active reduces to τ = mean(r). r, z (..., S) -> v (..., S).
    """
    if r.shape != z.shape:
        raise ValueError(
            f"simplex_redistribute r {tuple(r.shape)} vs z {tuple(z.shape)}"
        )
    active = z > z_eps
    n_a = active.sum(dim=-1)
    sum_a = (r * active).sum(dim=-1)
    n_slots = r.shape[-1]
    neginf = torch.finfo(r.dtype).min
    inactive_r = torch.where(active, torch.full_like(r, neginf), r)
    sorted_i = inactive_r.sort(dim=-1, descending=True).values
    n_i = (~active).sum(dim=-1)

    # k = 0: τ = mean_active r. Valid if no inactive slot has r > τ.
    tau = sum_a / n_a.clamp_min(1).to(dtype=r.dtype)
    found = (n_i == 0) | (sorted_i[..., 0] <= tau)

    cum = torch.zeros_like(sum_a)
    for k in range(1, n_slots + 1):
        cum = cum + sorted_i[..., k - 1]
        tau_k = (sum_a + cum) / (n_a + k).to(dtype=r.dtype).clamp_min(1)
        rk = sorted_i[..., k - 1]
        if k < n_slots:
            next_r = sorted_i[..., k]
            remain_ok = (n_i <= k) | (next_r <= tau_k)
        else:
            remain_ok = torch.ones_like(tau_k, dtype=torch.bool)
        valid = (~found) & (n_i >= k) & (rk > tau_k) & remain_ok
        tau = torch.where(valid, tau_k, tau)
        found = found | valid

    delta = r - tau.unsqueeze(-1)
    v = torch.where(active, delta, delta.clamp_min(0.0))
    return v


def usage_gate_loss(
    logits: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """L_gate = (1/2) Σ_s (ℓ_s - sg(ℓ_s + v_s))^2, mean over batch (and time).

    ∇_ℓ L_gate = -v per sample (then averaged over the batch).
    """
    if logits.shape != v.shape:
        raise ValueError(
            f"usage_gate_loss logits {tuple(logits.shape)} vs v {tuple(v.shape)}"
        )
    target = (logits + v).detach()
    return 0.5 * (logits - target).pow(2).sum(dim=-1).mean()


def usage_redistribute_direction(
    mix: torch.Tensor,
    slot_recons: torch.Tensor,
    masks_ungated: torch.Tensor,
    gate: torch.Tensor,
    target: torch.Tensor,
    logits: torch.Tensor,
    lam: float,
    eps: float = 1e-8,
    z_eps: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(L_gate, d, r, v). d/r/v are detached; L_gate trains logits."""
    d = reconstruction_usage_grad(
        mix, slot_recons, masks_ungated, gate.detach(), target, eps=eps
    )
    z = gate.detach().float()
    r = float(lam) * z - d.float()
    v = simplex_redistribute(r, z, z_eps=z_eps).to(dtype=logits.dtype)
    loss = usage_gate_loss(logits, v)
    return loss, d, r.to(dtype=d.dtype), v
