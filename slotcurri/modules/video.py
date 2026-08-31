import math
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
from torch import nn

from slotcurri.utils import make_build_fn


@make_build_fn(__name__, "video module")
def build(config, name: str, **kwargs):
    pass  # No special module building needed


def _max_norm(gate: Optional[torch.Tensor], enabled: bool) -> Optional[torch.Tensor]:
    """g / max_s(g), so the winning slot always advances fully in the temporal mix.

    Without this the early curriculum can put p above every slot's mass, leaving no
    slot with a gate large enough to advance at all. Bool (hard) gates and the
    disabled case pass through untouched.
    """
    if not enabled or gate is None or gate.dtype == torch.bool:
        return gate
    return gate / gate.amax(dim=-1, keepdim=True).clamp_min(1e-8)


def _slot_identity_cos(
    prior: torch.Tensor, posterior: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Per-slot ReLU cosine between tracked prior û and current slot u (v39i).

    prior / posterior: (B, S, D) -> (B, S) in [0, 1]. Opposite hemispheres
    (cos < 0) are treated as an identity break and zero the mix.
    """
    u = torch.nn.functional.normalize(prior.float(), dim=-1, eps=eps)
    v = torch.nn.functional.normalize(posterior.float(), dim=-1, eps=eps)
    return (u * v).sum(dim=-1).clamp_min(0.0)


def _temporal_gate_smooth(
    cur: torch.Tensor,
    prev: Optional[torch.Tensor],
    momentum: float,
    hold: float,
) -> torch.Tensor:
    """Smooth the temporal-mix gate. The decoder must keep `cur` (instantaneous π).

    π̃_t = m π_t + (1-m) π̃_{t-1}, then π̃_t ← max(π̃_t, hold · π̃_{t-1}).
    m=1 and hold=0 is identity. `prev` is the previous smoothed gate.
    """
    if prev is None:
        return cur
    mom = float(momentum)
    out = cur if mom == 1.0 else mom * cur + (1.0 - mom) * prev
    h = float(hold)
    if h > 0.0:
        out = torch.maximum(out, h * prev)
    return out


def _threshold_is_open(p_thresh) -> bool:
    """p <= 0 means the mass curriculum has faded out: the mass gate is identity.

    Used by v26p (`p_end_mult: 0`): after the cosine anneal (and at eval) the decoder
    stops reweighting by coverage, while a different `state_gate_form` can keep the
    temporal mix selective. A log-ratio at p=0 would otherwise rely on log(eps).
    """
    if p_thresh is None:
        return False
    if torch.is_tensor(p_thresh):
        return bool((p_thresh <= 0).all().item())
    return float(p_thresh) <= 0.0


def _as_float_gate(gate: torch.Tensor) -> torch.Tensor:
    """Bool hard gates become 0/1 floats so they can multiply a soft statistic."""
    return gate.float() if gate.dtype == torch.bool else gate


def _apply_default_slots(gate: torch.Tensor, default_list) -> torch.Tensor:
    """Force listed slots fully on (float gates). Empty list is a no-op."""
    if not default_list:
        return gate
    default_vec = torch.zeros_like(gate[:1])
    for di in default_list:
        if 0 <= int(di) < default_vec.shape[1]:
            default_vec[0, int(di)] = 1.0
    return torch.maximum(gate, default_vec)


def _purity_weight_gate(
    gate_conf: torch.Tensor, purity_normalize: bool, default_list
) -> torch.Tensor:
    """v32/v36 purity gate: detached ownership c, optionally K-normalized to 0-baseline.

    p = clip((K c - 1)/(K - 1), 0, 1) maps uniform 1/K -> 0 and exclusive owner -> 1.
    """
    g = gate_conf
    if purity_normalize and g is not None:
        n_slots_g = g.shape[-1]
        if n_slots_g > 1:
            g = ((n_slots_g * g - 1.0) / (n_slots_g - 1.0)).clamp(0.0, 1.0)
    return _apply_default_slots(g, default_list)


def _top2_gap_wtw(w: torch.Tensor, n_iter: int, eps: float) -> torch.Tensor:
    """λ1-λ2 of C = W^T W by k=2 subspace iteration + 2x2 Rayleigh-Ritz.

    Same pattern as Ncut Fiedler: never form C or run a full D×D eigh. Matvecs
    are W^T (W Q). Ridge eps*Q keeps QR defined when a slot's mass is ~0; it
    cancels in the gap. w: (B, N, D) -> (B,) gap.
    """
    bsz, _, dim = w.shape
    if dim < 2:
        gram = torch.bmm(w.transpose(1, 2), w)
        return torch.linalg.eigvalsh(gram)[..., -1].clamp_min(0.0)
    ones = torch.ones(bsz, dim, 1, device=w.device, dtype=w.dtype)
    grid = torch.linspace(-1.0, 1.0, dim, device=w.device, dtype=w.dtype)
    q = torch.cat([ones, grid.view(1, dim, 1).expand(bsz, -1, -1)], dim=-1)
    q, _ = torch.linalg.qr(q)
    n_iter = max(int(n_iter), 1)
    for _ in range(n_iter):
        cq = torch.bmm(w.transpose(1, 2), torch.bmm(w, q))
        q, _ = torch.linalg.qr(cq + float(eps) * q)
    cq = torch.bmm(w.transpose(1, 2), torch.bmm(w, q))
    rax = torch.bmm(q.transpose(1, 2), cq)
    rax = 0.5 * (rax + rax.transpose(1, 2))
    evals = torch.linalg.eigvalsh(rax)
    return (evals[:, -1] - evals[:, 0]).clamp_min(0.0)


def _qr_mgs(q: torch.Tensor, eps: float) -> torch.Tensor:
    """Batched modified Gram-Schmidt. q: (..., D, k) -> (..., D, k).

    Avoids `torch.linalg.qr` / geqrf on many skinny tall factors (the v37
    spectral bottleneck when k=2, and v38 when k=8 on N-dimensional G_s).
    """
    cols = []
    for i in range(q.shape[-1]):
        v = q[..., i]
        for u in cols:
            v = v - (v * u).sum(dim=-1, keepdim=True) * u
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(eps)
        cols.append(v)
    return torch.stack(cols, dim=-1)


def _qr_k2(q: torch.Tensor, eps: float) -> torch.Tensor:
    """Batched modified Gram-Schmidt for k=2. q: (..., D, 2) -> (..., D, 2)."""
    return _qr_mgs(q, eps)


def _top2_gap_c_from_z(
    z: torch.Tensor, a: torch.Tensor, n_iter: int, eps: float
) -> torch.Tensor:
    """λ1-λ2 of C_s = Z^T diag(a_s^2) Z without forming (B, S, N, D).

    Matvecs are C q = Z^T (a^2 ⊙ (Z q)). z: (B, N, D), a: (B, S, N) -> (B, S).
    Same 2D subspace / Ritz as `_top2_gap_wtw`.
    """
    bsz, n_slots, _ = a.shape
    dim = z.shape[-1]
    if dim < 2:
        w = a.unsqueeze(-1) * z.unsqueeze(1)
        return _top2_gap_wtw(
            w.reshape(bsz * n_slots, z.shape[1], dim), n_iter, eps
        ).view(bsz, n_slots)

    a2 = a * a
    ones = torch.ones(
        bsz, n_slots, dim, 1, device=z.device, dtype=z.dtype
    )
    grid = torch.linspace(-1.0, 1.0, dim, device=z.device, dtype=z.dtype)
    q = torch.cat(
        [ones, grid.view(1, 1, dim, 1).expand(bsz, n_slots, -1, -1)], dim=-1
    )
    q = _qr_k2(q, eps)

    def apply_c(basis: torch.Tensor) -> torch.Tensor:
        n_k = basis.shape[-1]
        zq = torch.bmm(
            z, basis.permute(0, 2, 1, 3).reshape(bsz, dim, n_slots * n_k)
        )
        zq = zq.view(bsz, z.shape[1], n_slots, n_k).permute(0, 2, 1, 3)
        zq = a2.unsqueeze(-1) * zq
        cq = torch.bmm(
            z.transpose(1, 2),
            zq.permute(0, 2, 1, 3).reshape(bsz, z.shape[1], n_slots * n_k),
        )
        return cq.view(bsz, dim, n_slots, n_k).permute(0, 2, 1, 3)

    n_iter = max(int(n_iter), 1)
    for _ in range(n_iter):
        q = _qr_k2(apply_c(q) + float(eps) * q, eps)

    cq = apply_c(q)
    rax = torch.matmul(q.transpose(-1, -2), cq)
    rax = 0.5 * (rax + rax.transpose(-1, -2))
    evals = torch.linalg.eigvalsh(rax)
    return (evals[..., 1] - evals[..., 0]).clamp_min(0.0)


def spectral_slot_purity(
    att: torch.Tensor,
    features: torch.Tensor,
    eps: float = 1e-6,
    chunk_size: int = 64,
    n_iter: int = 16,
) -> torch.Tensor:
    """Detached spectral slot purity π_s (v37). No gamma, no 1/K normalization.

    Last-iter ownership a_s = A_{:,s} and L2-normalized bind tokens Z give

        C_s = Z^T diag(a_s^2) Z

    then π_s = clip((λ1 - λ2) / (sum_i A_{i,s} + eps), 0, 1). a^2 already sharpens,
    so mass_gamma is not applied here. Rank-1 exclusive ownership recovers the
    ownership purity Σ a^2 / Σ A; a slot covering two feature modes raises λ2 and
    drops π even when ownership is exclusive (the v32 blind spot N-cut was papering
    over). features is X^bind at DINO dim (Key-only curriculum's bind tokens;
    raw backbone tokens when the curriculum is off / mix=1).

    λ1, λ2 come from k=2 subspace iteration on C_s (same method as the Ncut
    Fiedler): π only needs that gap, not the full spectrum of the D×D matrix.
    Matvecs are Z^T (a^2 ⊙ (Z q)); the (B, S, N, D) token copies are never
    materialized. `chunk_size` only bounds peak memory.
    """
    if features is None:
        raise ValueError(
            "conf_kind='spectral' requires bind_features (encoder backbone tokens)"
        )
    if att.ndim != 3 or features.ndim != 3:
        raise ValueError(
            f"spectral purity expects att (B, S, N) and features (B, N, D), "
            f"got {tuple(att.shape)} and {tuple(features.shape)}"
        )
    if att.shape[0] != features.shape[0] or att.shape[-1] != features.shape[1]:
        raise ValueError(
            f"spectral purity batch/token mismatch: att {tuple(att.shape)} vs "
            f"features {tuple(features.shape)}"
        )

    amp_ctx = (
        torch.cuda.amp.autocast(enabled=False) if att.is_cuda else nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        z = torch.nn.functional.normalize(features.float(), dim=-1)
        a = att.float()
        mass = a.sum(dim=-1)
        bsz = a.shape[0]
        gaps = a.new_empty(bsz, a.shape[1])
        step = max(int(chunk_size), 1)
        for i in range(0, bsz, step):
            gaps[i : i + step] = _top2_gap_c_from_z(
                z[i : i + step], a[i : i + step], n_iter, eps
            )
        pi = (gaps / (mass + float(eps))).clamp(0.0, 1.0)
    return pi.detach().to(dtype=att.dtype)


def _relation_graph_S(z: torch.Tensor, eps: float) -> torch.Tensor:
    """Symmetric normalized ReLU-cosine graph S = D^{-1/2} R D^{-1/2}.

    z is L2-normalized (B, N, D). R_ij = ReLU(z_i^T z_j), R_ii = 0. Isolated
    rows get D_ii = eps so S stays defined; those rows are ~0.
    """
    rel = torch.bmm(z, z.transpose(1, 2)).clamp_min(0.0)
    rel.diagonal(dim1=-2, dim2=-1).zero_()
    deg = rel.sum(dim=-1).clamp_min(eps)
    d_inv = deg.rsqrt()
    return d_inv.unsqueeze(-1) * rel * d_inv.unsqueeze(-2)


_N8_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _square_grid(n_tokens: int) -> int:
    grid = int(math.sqrt(n_tokens))
    if grid * grid != n_tokens:
        raise ValueError(
            f"spectral_graph_n8 expects a square patch grid, got N={n_tokens}"
        )
    return grid


def _shift_hw(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Shift spatial dims 1,2 so out[y, x] = x[y-dy, x-dx], else 0."""
    if dy == 0 and dx == 0:
        return x
    height, width = x.shape[1], x.shape[2]
    pad_top = max(dy, 0)
    pad_bot = max(-dy, 0)
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    tail = x.ndim - 3
    pad = (0, 0) * tail + (pad_left, pad_right, pad_top, pad_bot)
    xp = torch.nn.functional.pad(x, pad)
    y0, x0 = pad_bot, pad_right
    return xp[:, y0 : y0 + height, x0 : x0 + width]


def _n8_affinity(z: torch.Tensor, eps: float):
    """Precompute 8-neighbor ReLU-cosine weights and d^{-1/2}.

    z: (B, N, D) L2-normalized -> d_inv (B, H, W), weights (B, 8, H, W).
    Affinity is fixed for the eigensolve, so matvecs must not recompute z·z_nb.
    """
    bsz, n_tokens, _ = z.shape
    grid = _square_grid(n_tokens)
    z_hw = z.view(bsz, grid, grid, -1)
    weights = []
    deg = z.new_zeros(bsz, grid, grid)
    for dy, dx in _N8_OFFSETS:
        r = (z_hw * _shift_hw(z_hw, dy, dx)).sum(dim=-1).clamp_min(0.0)
        weights.append(r)
        deg = deg + r
    d_inv = deg.clamp_min(eps).rsqrt()
    return d_inv, torch.stack(weights, dim=1)


def _apply_s_n8(
    d_inv: torch.Tensor, weights: torch.Tensor, vec: torch.Tensor
) -> torch.Tensor:
    """S v from precomputed 8-neighbor weights. Never forms N×N.

    d_inv: (B, H, W), weights: (B, 8, H, W), vec: (B, N, C) -> (B, N, C).
    """
    bsz, n_tokens, n_c = vec.shape
    grid = d_inv.shape[1]
    v_hw = vec.view(bsz, grid, grid, n_c)
    w_hw = d_inv.unsqueeze(-1) * v_hw
    acc = vec.new_zeros(bsz, grid, grid, n_c)
    for i, (dy, dx) in enumerate(_N8_OFFSETS):
        acc = acc + weights[:, i].unsqueeze(-1) * _shift_hw(w_hw, dy, dx)
    return (d_inv.unsqueeze(-1) * acc).view(bsz, n_tokens, n_c)


def _apply_g_n8(
    d_inv: torch.Tensor,
    weights: torch.Tensor,
    a: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """G_s v = a ⊙ (S (a ⊙ v)) with 8-neighbor S."""
    bsz, n_slots, n_tokens = a.shape
    n_k = basis.shape[-1]
    av = a.unsqueeze(-1) * basis
    packed = av.permute(0, 2, 1, 3).reshape(bsz, n_tokens, n_slots * n_k)
    sav = _apply_s_n8(d_inv, weights, packed)
    sav = sav.view(bsz, n_tokens, n_slots, n_k).permute(0, 2, 1, 3)
    return a.unsqueeze(-1) * sav


def _apply_g_diag_s_diag(
    s_mat: torch.Tensor, a: torch.Tensor, basis: torch.Tensor
) -> torch.Tensor:
    """G_s v = a ⊙ (S (a ⊙ v)) for a stack of k basis vectors.

    s_mat: (B, N, N), a: (B, S, N), basis: (B, S, N, k) -> (B, S, N, k)
    """
    bsz, n_slots, n_tokens = a.shape
    n_k = basis.shape[-1]
    av = a.unsqueeze(-1) * basis
    sav = torch.bmm(
        s_mat,
        av.permute(0, 2, 1, 3).reshape(bsz, n_tokens, n_slots * n_k),
    )
    sav = sav.view(bsz, n_tokens, n_slots, n_k).permute(0, 2, 1, 3)
    return a.unsqueeze(-1) * sav


def _top2_algebraic_g(
    apply_g: Callable[[torch.Tensor], torch.Tensor],
    a: torch.Tensor,
    n_iter: int,
    eps: float,
):
    """Algebraic λ1 ≥ λ2 of G_s given Gv = apply_g(v).

    G is not PSD (zero diagonal). For a nonnegative symmetric matrix the
    spectral radius is λ1, so |λ_min| ≤ λ1. Shift H = G + (λ1 + eps) I
    (Perron λ1 from a power iteration) so H is PSD with condition ~2, then
    a k=8 subspace iteration + Rayleigh-Ritz; the two largest Ritz values
    minus the shift are λ1, λ2 of G. Matvecs never form G.
    """
    bsz, n_slots, n_tokens = a.shape
    zero = a.new_zeros(bsz, n_slots)
    if n_tokens < 2:
        return zero, zero

    n_iter = max(int(n_iter), 1)
    ones = torch.ones(
        bsz, n_slots, n_tokens, 1, device=a.device, dtype=a.dtype
    )
    q1 = ones / ones.norm(dim=2, keepdim=True).clamp_min(eps)
    for _ in range(n_iter):
        gq = apply_g(q1)
        nrm = gq.norm(dim=2, keepdim=True)
        q1 = torch.where(nrm > float(eps), gq / nrm.clamp_min(eps), q1)
    lam1 = (q1 * apply_g(q1)).sum(dim=2).squeeze(-1)
    shift = lam1.clamp_min(0.0) + float(eps)

    n_k = int(min(8, n_tokens))
    grid = torch.linspace(0.0, 1.0, n_tokens, device=a.device, dtype=a.dtype)
    cols = [q1.squeeze(-1)]
    for freq in range(1, n_k):
        wave = torch.cos(float(freq) * math.pi * grid)
        cols.append(wave.view(1, 1, n_tokens).expand(bsz, n_slots, -1))
    q = torch.stack(cols, dim=-1)
    q = _qr_mgs(q, eps)

    def apply_h(basis: torch.Tensor) -> torch.Tensor:
        return apply_g(basis) + (shift.unsqueeze(-1).unsqueeze(-1) * basis)

    for _ in range(n_iter):
        q = _qr_mgs(apply_h(q) + float(eps) * q, eps)

    hq = apply_h(q)
    rax = torch.matmul(q.transpose(-1, -2), hq)
    rax = 0.5 * (rax + rax.transpose(-1, -2))
    evals_h = torch.linalg.eigvalsh(rax)
    lam = evals_h[..., -2:] - shift.unsqueeze(-1)
    return lam[..., 1], lam[..., 0]


def _top2_algebraic_diag_s_diag(
    s_mat: torch.Tensor, a: torch.Tensor, n_iter: int, eps: float
):
    """Algebraic λ1 ≥ λ2 of G_s = diag(a_s) S diag(a_s) for dense S (v38)."""

    def apply_g(basis: torch.Tensor) -> torch.Tensor:
        return _apply_g_diag_s_diag(s_mat, a, basis)

    return _top2_algebraic_g(apply_g, a, n_iter, eps)


def _top2_algebraic_n8(
    z: torch.Tensor, a: torch.Tensor, n_iter: int, eps: float
):
    """Algebraic λ1 ≥ λ2 of G_s on the 8-neighbor ReLU-cosine S (v39)."""
    d_inv, weights = _n8_affinity(z, eps)

    def apply_g(basis: torch.Tensor) -> torch.Tensor:
        return _apply_g_n8(d_inv, weights, a, basis)

    return _top2_algebraic_g(apply_g, a, n_iter, eps)


def spectral_graph_slot_purity(
    att: torch.Tensor,
    features: torch.Tensor,
    eps: float = 1e-6,
    chunk_size: int = 32,
    n_iter: int = 16,
) -> torch.Tensor:
    """Detached relation-graph spectral purity π_s (v38).

    Last-iter ownership a_s and L2-normalized bind tokens Z give the same
    ReLU-cosine graph as the Key curriculum, symmetrically normalized:

        R_ij = ReLU(z_i^T z_j),  R_ii = 0
        S = D^{-1/2} R D^{-1/2}
        G_s = diag(a_s) S diag(a_s)
        π_s = λ1^{(s)} - max(λ2^{(s)}, 0)

    Ghosts shrink with a_i a_j (no /mass). A slot that cuts a strong relation
    drops λ1; two communities in one slot raise λ2. π stays detached. features
    is X^bind at DINO dim (raw backbone tokens when the curriculum is off).
    """
    if features is None:
        raise ValueError(
            "conf_kind='spectral_graph' requires bind_features "
            "(encoder backbone tokens)"
        )
    if att.ndim != 3 or features.ndim != 3:
        raise ValueError(
            f"spectral-graph purity expects att (B, S, N) and features "
            f"(B, N, D), got {tuple(att.shape)} and {tuple(features.shape)}"
        )
    if att.shape[0] != features.shape[0] or att.shape[-1] != features.shape[1]:
        raise ValueError(
            f"spectral-graph purity batch/token mismatch: att "
            f"{tuple(att.shape)} vs features {tuple(features.shape)}"
        )

    amp_ctx = (
        torch.cuda.amp.autocast(enabled=False) if att.is_cuda else nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        z = torch.nn.functional.normalize(features.float(), dim=-1)
        a = att.float()
        bsz, n_slots, _ = a.shape
        pi = a.new_empty(bsz, n_slots)
        step = max(int(chunk_size), 1)
        for i in range(0, bsz, step):
            z_c = z[i : i + step]
            a_c = a[i : i + step]
            s_mat = _relation_graph_S(z_c, eps)
            lam1, lam2 = _top2_algebraic_diag_s_diag(s_mat, a_c, n_iter, eps)
            pi[i : i + step] = (lam1 - lam2.clamp_min(0.0)).clamp_min(0.0)
    return pi.detach().to(dtype=att.dtype)


def spectral_graph_n8_slot_purity(
    att: torch.Tensor,
    features: torch.Tensor,
    eps: float = 1e-6,
    chunk_size: int = 32,
    n_iter: int = 16,
    divide_by_lambda1: bool = False,
    l1_minus_imp: bool = False,
) -> torch.Tensor:
    """Detached 8-neighbor relation-graph spectral purity π_s (v39 / v39n / v39s).

    Same G_s = diag(a) S diag(a) as v38, but R is ReLU-cosine on 8-neighbors
    only (no dense N×N). Last-iter softmax a is used as-is (no argmax).

        π_s = λ1 - max(λ2, 0)                         # v39
        π_s = (λ1 - max(λ2, 0)) / max(λ1, eps)        # v39n (divide_by_lambda1)
        π_s = [λ1 - max(λ2, 0) / (λ1 + eps)]_+        # v39s (l1_minus_imp)

    Same gap as v38; only R is 8-neighbor. Z is L2-normalized X^bind.
    π stays detached. Curriculum P is unchanged (global ReLU-cosine);
    this graph is gate-only. The relative form is in [0, 1] and drops
    the G_s scale (ghosts no longer die from a_i a_j). v39s keeps the
    G_s scale (λ1) and subtracts relative impurity λ2⁺/(λ1+ε).
    """
    if divide_by_lambda1 and l1_minus_imp:
        raise ValueError(
            "spectral_graph_n8_slot_purity: divide_by_lambda1 and "
            "l1_minus_imp cannot both be True"
        )
    if features is None:
        raise ValueError(
            "conf_kind='spectral_graph_n8' requires bind_features "
            "(encoder backbone tokens)"
        )
    if att.ndim != 3 or features.ndim != 3:
        raise ValueError(
            f"spectral-graph-n8 purity expects att (B, S, N) and features "
            f"(B, N, D), got {tuple(att.shape)} and {tuple(features.shape)}"
        )
    if att.shape[0] != features.shape[0] or att.shape[-1] != features.shape[1]:
        raise ValueError(
            f"spectral-graph-n8 purity batch/token mismatch: att "
            f"{tuple(att.shape)} vs features {tuple(features.shape)}"
        )
    _square_grid(att.shape[-1])

    amp_ctx = (
        torch.cuda.amp.autocast(enabled=False) if att.is_cuda else nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        z = torch.nn.functional.normalize(features.float(), dim=-1)
        a = att.float()
        bsz = a.shape[0]
        pi = a.new_empty(bsz, a.shape[1])
        step = max(int(chunk_size), 1)
        for i in range(0, bsz, step):
            z_c = z[i : i + step]
            a_c = a[i : i + step]
            lam1, lam2 = _top2_algebraic_n8(z_c, a_c, n_iter, eps)
            lam2p = lam2.clamp_min(0.0)
            if l1_minus_imp:
                # v39s: [λ1 - λ2⁺ / (λ1 + ε)]_+
                pi[i : i + step] = (lam1 - lam2p / (lam1 + float(eps))).clamp_min(0.0)
            elif divide_by_lambda1:
                pi[i : i + step] = (
                    (lam1 - lam2p).clamp_min(0.0) / lam1.clamp_min(float(eps))
                ).clamp(0.0, 1.0)
            else:
                pi[i : i + step] = (lam1 - lam2p).clamp_min(0.0)
    return pi.detach().to(dtype=att.dtype)


def ownership_confidence(att: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """K-normalized ownership confidence (v40). att (..., S, F) -> c (..., S).

    u_s = sum_f A_{s,f}^2 / sum_f A_{s,f}, then
    c_s = clip((K u_s - 1)/(K - 1), 0, 1). Uniform 1/K -> 0, exclusive owner -> 1.
    Live: no detach. The gate applies sg(c) separately.
    """
    mass = att.sum(dim=-1).clamp_min(eps)
    u = (att * att).sum(dim=-1) / mass
    n_slots = att.shape[-2]
    if n_slots <= 1:
        return u.clamp(0.0, 1.0)
    return ((n_slots * u - 1.0) / (n_slots - 1.0)).clamp(0.0, 1.0)


def slot_confidence_entropy(c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalized entropy of the slot-wise confidence distribution (v40).

    q_s ∝ c_s + eps, H(q)/log K in [0, 1]. c (..., S) -> (...,).
    One-hot c -> 0; all-zero / uniform c -> 1. Live.
    """
    n_slots = c.shape[-1]
    q = (c + eps) / (c.sum(dim=-1, keepdim=True) + float(n_slots) * eps)
    ent = -(q * q.clamp_min(eps).log()).sum(dim=-1)
    return ent / math.log(max(n_slots, 2))


def spectral_graph_n8_impurity(
    att: torch.Tensor,
    features: torch.Tensor,
    eps: float = 1e-6,
    chunk_size: int = 32,
    n_iter: int = 16,
) -> torch.Tensor:
    """Differentiable 8-neighbor impurity ρ_s = max(λ2, 0) / (max(λ1, 0) + eps).

    Same G_s as v39. Z is detached (frozen DINO). a is live so the loss can
    split a slot that holds two communities. att (B, S, N), features (B, N, D)
    -> rho (B, S).
    """
    if features is None:
        raise ValueError(
            "spectral_graph_n8_impurity requires bind_features "
            "(encoder backbone tokens)"
        )
    if att.ndim != 3 or features.ndim != 3:
        raise ValueError(
            f"n8 impurity expects att (B, S, N) and features (B, N, D), "
            f"got {tuple(att.shape)} and {tuple(features.shape)}"
        )
    if att.shape[0] != features.shape[0] or att.shape[-1] != features.shape[1]:
        raise ValueError(
            f"n8 impurity batch/token mismatch: att {tuple(att.shape)} vs "
            f"features {tuple(features.shape)}"
        )
    _square_grid(att.shape[-1])

    amp_ctx = (
        torch.cuda.amp.autocast(enabled=False) if att.is_cuda else nullcontext()
    )
    with amp_ctx:
        z = torch.nn.functional.normalize(features.float(), dim=-1).detach()
        a = att.float()
        chunks = []
        step = max(int(chunk_size), 1)
        for i in range(0, a.shape[0], step):
            lam1, lam2 = _top2_algebraic_n8(
                z[i : i + step], a[i : i + step], n_iter, eps
            )
            denom = lam1.clamp_min(0.0) + float(eps)
            chunks.append(lam2.clamp_min(0.0) / denom)
    return torch.cat(chunks, dim=0).to(dtype=att.dtype)


class LatentProcessor(nn.Module):
    """Updates latent state based on inputs and state and predicts next state."""

    def __init__(
        self,
        corrector: nn.Module,
        predictor: Optional[nn.Module] = None,
        state_key: str = "slots",
        first_step_corrector_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.corrector = corrector # slot attention
        self.predictor = predictor # transformer encoder
        self.state_key = state_key
        if first_step_corrector_args is not None:
            self.first_step_corrector_args = first_step_corrector_args
        else:
            self.first_step_corrector_args = None
        # Velocity conditioning turns itself on when the predictor was built with vel_dim,
        # so a single config knob on the predictor controls the whole path.
        self.predictor_takes_vel = predictor is not None and any(
            getattr(block, "vel_proj", None) is not None
            for block in getattr(predictor, "blocks", [])
        )

    def forward(
        self, state: torch.Tensor, inputs: Optional[torch.Tensor], time_step: Optional[int] = None,
        onetoone: bool = False, gate_p: Optional[float] = None, default_idx: Any = 0,
        gate_mode: str = "hard", gate_tau: Optional[float] = None,
        mass_gamma: float = 1.0, purity_q: Optional[float] = None,
        purity_tau: Optional[float] = None,
        gate_p_state: Optional[float] = None,
        state_max_norm: bool = False,
        predictor_ungated: bool = False,
        state_gate_ema: float = 1.0,
        state_gate_hold: float = 0.0,
        state_gate_prev: Optional[torch.Tensor] = None,
        prev_state: Optional[torch.Tensor] = None,
        p_mode: str = "absolute",
        median_ema_prev: Optional[torch.Tensor] = None,
        median_ema_momentum: float = 0.9,
        p_residual_net: Optional[nn.Module] = None,
        p_residual_alpha: float = 0.0,
        gate_form: str = "linear",
        gate_beta: Optional[float] = None,
        gate_tau_log: Optional[float] = None,
        conf_kind: str = "entropy",
        gate_detach: bool = False,
        key_features: Optional[torch.Tensor] = None,
        purity_normalize: bool = False,
        state_gate_form: Optional[str] = None,
        state_mul_decoder: bool = False,
        decoder_mul_state: bool = False,
        bind_features: Optional[torch.Tensor] = None,
        gate_hysteresis: float = 0.0,
        gate_conf_prev: Optional[torch.Tensor] = None,
        state_identity_cos: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # state: batch x n_slots x slot_dim (1 7 64)
        if onetoone:
            # state: batch x n_slots x slot_dim (1 30 1369 64)
            assert state.ndim == 3
        else:
            # state: batch x n_slots x slot_dim (1 30 1369)
            assert state.ndim == 3
        # inputs: batch x n_inputs x input_dim (1 30 1369 64)
        assert inputs.ndim == 3
        if inputs is not None:
            corr_kwargs: Dict[str, Any] = {}
            if time_step == 0 and self.first_step_corrector_args:
                corr_kwargs.update(self.first_step_corrector_args)
            if key_features is not None:
                corr_kwargs["key_features"] = key_features
            corrector_output = self.corrector(state, inputs, **corr_kwargs)
            updated_state = corrector_output[self.state_key]
            state_attn_mask = corrector_output['masks'] if 'masks' in corrector_output else None
        else:
            # Run predictor without updating on current inputs
            corrector_output = None
            updated_state = state
            state_attn_mask = None

        # --- attention-mass gating: hold non-active slots' priors at their incoming state ---
        # `state_attn_mask` is the last-iteration softmax-over-slots attention (B, S, F).
        # Per-slot attention mass = fraction of patches claimed by that slot.
        # `gate_mode`:
        #   "hard" -> binary active/dormant (active_mask is bool); dormant slots removed.
        #   "soft" -> continuous gate g = sigmoid((mass - p) / tau) in [0, 1] (float);
        #            a small tau (annealed over training) sharpens toward the hard decision.
        #   "ste"  -> straight-through: forward uses the hard 0/1 gate (true dormancy) while
        #            the backward pass uses the sigmoid surrogate gradient (differentiable).
        active_mask = None
        state_mask = None
        mass_median_ema = None
        gate_p_eff = None
        gate_delta = None
        gate_conf = None
        # gate_form="purity_weight" has no threshold, so it activates on the form alone
        # (gate_p arrives as None from the model, every other form still requires it).
        # state_gate_form lets the temporal mix use a different statistic than the decoder
        # (v26p: decoder log-ratio mass, temporal v36 purity_weight).
        state_form = str(state_gate_form or gate_form).lower()
        gating_on = (
            gate_p is not None
            or gate_form == "purity_weight"
            or state_form == "purity_weight"
        )
        if gating_on and state_attn_mask is not None:
            att = state_attn_mask  # (B, S, F), per-patch softmax over slots
            if mass_gamma is not None and float(mass_gamma) != 1.0:
                # gamma-sharpened mass: per-patch attention raised to gamma and renormalized.
                # gamma=1 -> plain attention mass (unchanged); gamma -> inf -> per-patch argmax
                # winner share. Sharpening suppresses "always-second" ghost slots that gather
                # mass without ever winning a patch, while keeping the score differentiable
                # and normalized (sums to 1 over slots, mean 1/S -> p_mult semantics hold).
                att_sharp = att.pow(float(mass_gamma))
                att_sharp = att_sharp / att_sharp.sum(dim=1, keepdim=True).clamp_min(1e-8)
            else:
                att_sharp = att
            mass_frac = att_sharp.sum(dim=-1) / att_sharp.shape[-1]  # (B, S)

            # --- assignment confidence (evidence-aware gate, gate_form="logratio") ---
            # A low mass does not by itself mean "inactive": a small object is low-mass too.
            # What separates a small active object from a diffuse ghost slot is how the
            # (sharpened) attention is distributed over the features the slot does claim:
            #   P_{s,f} = A~_{s,f} / sum_f A~_{s,f}   (per-slot distribution over features)
            #   H_s     = -sum_f P log P,   c_s = 1 - H_s / log F   in [0, 1]
            # small+peaked -> c high, diffuse -> c low. Detached (sg) so the model cannot
            # open its gate by artificially sharpening attention; the mass branch keeps its
            # gradient, which is the path featrec is supposed to shape.
            ck = str(conf_kind).lower()
            need_conf = gate_form in ("logratio", "purity_weight") or state_form in (
                "logratio",
                "purity_weight",
            )
            if need_conf:
                if ck in ("purity", "purity_sharp"):
                    # Ownership quality instead of spatial concentration (v29): the
                    # attention-weighted mean of the slot's own per-patch share,
                    #   c_s = sum_f A_{s,f}^2 / sum_f A_{s,f}  in (0, 1].
                    # Size-invariant (a fully-owned object scores ~1 whether it covers 20
                    # patches or 800, where the entropy form gives the large one c ~ 0.1).
                    # "purity" computes it on the raw attention, "purity_sharp" on the same
                    # gamma-sharpened tensor as the mass branch (one distribution, two
                    # moments). On v20 @ 100k the sharp form separates ghosts from small
                    # objects best (conf-only AUC 0.999 vs 0.997 raw vs 0.647 entropy; see
                    # event_analysis/conf_vs_purity_probe.py). Detached like the entropy
                    # form so the model cannot open its gate by sharpening.
                    src = att if ck == "purity" else att_sharp
                    gate_conf = (
                        (src * src).sum(dim=-1) / src.sum(dim=-1).clamp_min(1e-8)
                    ).clamp(min=0.0, max=1.0).detach()
                elif ck == "spectral":
                    # v37: content-aware purity. Exclusive ownership of two feature
                    # modes is no longer c~=1; λ2 rises and π drops. Uses raw A (a^2
                    # in C_s is the sharpening) and the bind tokens, not att_sharp.
                    gate_conf = spectral_slot_purity(att, bind_features)
                elif ck == "spectral_graph":
                    # v38: relation-graph spectral purity. Same ReLU-cosine R as the
                    # Key curriculum, then S = D^{-1/2} R D^{-1/2} and
                    # G_s = diag(a) S diag(a). π = λ1 - max(λ2, 0) on G_s; no /mass.
                    # Raw A, bind tokens, detached. mass_gamma unused.
                    gate_conf = spectral_graph_slot_purity(att, bind_features)
                elif ck in (
                    "spectral_graph_n8",
                    "spectral_graph_n8_rel",
                    "spectral_graph_n8_l1imp",
                ):
                    # v39: 8-neighbor R only. π = λ1 - max(λ2, 0) (same as v38).
                    # v39n (spectral_graph_n8_rel): that gap divided by λ1.
                    # v39s (spectral_graph_n8_l1imp): [λ1 - λ2⁺/(λ1+ε)]_+.
                    # Curriculum P stays global. Raw A, bind tokens, detached.
                    gate_conf = spectral_graph_n8_slot_purity(
                        att,
                        bind_features,
                        divide_by_lambda1=(ck == "spectral_graph_n8_rel"),
                        l1_minus_imp=(ck == "spectral_graph_n8_l1imp"),
                    )
                else:
                    p_feat = att_sharp / att_sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    ent = -(p_feat * p_feat.clamp_min(1e-8).log()).sum(dim=-1)  # (B, S)
                    log_f = math.log(max(att_sharp.shape[-1], 2))
                    c_ent = (1.0 - ent / log_f).clamp(min=0.0, max=1.0)
                    if ck == "entropy_max":
                        # v26fmax: restore the height entropy drops. c_ent is computed
                        # on P = A~ / sum_f A~ and is invariant to per-patch ownership;
                        # an always-second ghost with the same spatial support as a
                        # small object gets the same c. max_f A_{s,f} is that height
                        # on the RAW softmax-over-slots attention (winner ~1, ghost
                        # never-argmax so < 0.5). Product, then detach like entropy.
                        peak = att.max(dim=-1).values.clamp(min=0.0, max=1.0)
                        gate_conf = (c_ent * peak).clamp(min=0.0, max=1.0).detach()
                    else:
                        gate_conf = c_ent.detach()

            # --- temporal hysteresis on the confidence statistic (occlusion memory) ---
            # π̃_t = max(π_t, γ π̃_{t-1}). Partial occlusion splits an object's visible
            # support into disconnected 8-nbr components, which the n8 gap misreads as a
            # two-community merge, so π collapses exactly while the object is occluded;
            # the leaky max keeps a recently-pure slot's gate alive for ~1/(1-γ) frames
            # so its prior is held and the decoder does not drop it instantly. Ghosts
            # gain nothing (their π was never high). `gate_conf_prev` is the previous
            # frame's SMOOTHED statistic (carried by ScanOverTime), so the decay is
            # recursive. Detached like π itself; γ=0 or a missing prev is a no-op.
            if (
                gate_conf is not None
                and gate_conf_prev is not None
                and float(gate_hysteresis) > 0.0
            ):
                gate_conf = torch.maximum(
                    gate_conf, float(gate_hysteresis) * gate_conf_prev.detach()
                )

            # Threshold p:
            #   absolute (default): gate_p is the annealed absolute mass threshold
            #   median_ema: gate_p is annealed p_mult; p = sg(EMA[median(m)]) * p_mult
            #               (recovers absolute p≈p_mult/S when masses are near-uniform)
            #   learnable_residual: gate_p is absolute p_sched;
            #               p = p_sched * (1 + alpha * tanh(f(sg[m])))
            p_mode_l = str(p_mode).lower()
            if gate_form == "purity_weight":
                p_use = None  # thresholdless form: p_mode machinery is inert
            elif p_mode_l == "median_ema":
                med = mass_frac.median(dim=-1).values.detach()  # (B,)
                mom = float(median_ema_momentum)
                mom = min(max(mom, 0.0), 1.0)
                if median_ema_prev is None:
                    mass_median_ema = med
                else:
                    mass_median_ema = mom * median_ema_prev.detach() + (1.0 - mom) * med
                # (B, 1) so it broadcasts over slots; detached scene statistic
                p_use = (mass_median_ema * float(gate_p)).unsqueeze(-1)
            elif p_mode_l == "learnable_residual" and p_residual_net is not None:
                p_sched = float(gate_p)
                # features detached; delta still receives loss grad via L_delta and via g(m-p)
                gate_delta = p_residual_net(mass_frac.detach())  # (B, 1), in (-1, 1)
                alpha = float(p_residual_alpha)
                p_use = p_sched * (1.0 + alpha * gate_delta)
                p_use = p_use.clamp(min=1e-6, max=0.999)
            else:
                p_use = gate_p  # scalar absolute threshold
            gate_p_eff = p_use

            purity = None
            if purity_q is not None:
                # size-invariant "ownership quality": attention-weighted mean of the slot's own
                # per-patch attention (sum A^2 / sum A, in (0, 1]). High for slots that dominantly
                # own the (few) patches they claim; low for diffuse ghost slots. Used as an OR
                # rescue so small-but-cleanly-owned objects survive the mass threshold.
                purity = (att * att).sum(dim=-1) / att.sum(dim=-1).clamp_min(1e-8)  # (B, S)
            default_list = [default_idx] if isinstance(default_idx, int) else list(default_idx)

            # --- evidence score for the log-ratio gate ---
            # log r = beta * log(m + eps) + (1 - beta) * log(sg(c) + eps). Independent of the
            # threshold, so computed once and shared by the decoder gate and the state gate.
            # beta=1 (or gate_beta None) degenerates to pure coverage in log domain.
            log_r = None
            if gate_form == "logratio" or state_form == "logratio":
                log_eps = 1e-6
                beta = 1.0 if gate_beta is None else float(gate_beta)
                log_r = beta * (mass_frac + log_eps).log()
                if gate_conf is not None and beta < 1.0:
                    log_r = log_r + (1.0 - beta) * (gate_conf + log_eps).log()

            def build_gate(p_thresh):
                """Gate from the evidence score, thresholded at `p_thresh`.

                Factored out so the decoder gate and the state gate below differ only in
                their threshold and cannot drift apart.

                gate_form:
                  "linear"  : score = (m - p) / tau            (mass units; v6..v25)
                  "logratio": score = (log r - log p) / tau_g  (relative evidence; scale-
                              invariant, so r=0.02,p=0.01 and r=0.2,p=0.1 gate identically,
                              and the gate stays discriminative wherever p sits w.r.t. the
                              mass distribution instead of saturating like the linear form)
                p <= 0 (v26p p_end_mult=0): the mass gate is identity rather than
                log(eps). The temporal mix can still use a different state_gate_form.
                """
                if _threshold_is_open(p_thresh):
                    if gate_mode in ("soft", "ste"):
                        return _apply_default_slots(torch.ones_like(mass_frac), default_list)
                    open_bool = torch.ones(
                        mass_frac.shape, dtype=torch.bool, device=mass_frac.device
                    )
                    for di in default_list:
                        if 0 <= int(di) < open_bool.shape[1]:
                            open_bool[:, int(di)] = True
                    return open_bool
                if gate_form == "logratio":
                    if torch.is_tensor(p_thresh):
                        log_p = (p_thresh + 1e-6).log()
                    else:
                        log_p = math.log(float(p_thresh) + 1e-6)
                    tau_g = gate_tau_log if (gate_tau_log is not None and gate_tau_log > 0) else 0.5
                    score = (log_r - log_p) / tau_g  # (B, S)
                    exceeds = log_r >= log_p
                else:
                    tau = gate_tau if (gate_tau is not None and gate_tau > 0) else 1e-2
                    score = (mass_frac - p_thresh) / tau  # (B, S)
                    exceeds = mass_frac >= p_thresh
                if gate_mode in ("soft", "ste"):
                    soft = torch.sigmoid(score)  # (B, S) in (0, 1)
                    if purity is not None:
                        ptau = purity_tau if (purity_tau is not None and purity_tau > 0) else 5e-2
                        # OR-combination: active if big enough (mass) OR cleanly owned (purity)
                        soft = torch.maximum(soft, torch.sigmoid((purity - purity_q) / ptau))
                    # default slots are always fully active (value 1, no gradient there)
                    soft = _apply_default_slots(soft, default_list)
                    if gate_mode == "ste":
                        # forward = hard 0/1 (defaults forced on); backward = soft gradient
                        hard_bool = exceeds
                        if purity is not None:
                            hard_bool = hard_bool | (purity >= purity_q)
                        default_vec = torch.zeros_like(soft[:1])
                        for di in default_list:
                            if 0 <= int(di) < default_vec.shape[1]:
                                default_vec[0, int(di)] = 1.0
                        hard = torch.maximum(hard_bool.float(), default_vec)
                        return hard + (soft - soft.detach())
                    return soft  # float gate weights in [0, 1]
                active = exceeds.clone()  # (B, S)
                if purity is not None:
                    active = active | (purity >= purity_q)
                # default slots are always active (exempt from the threshold)
                for di in default_list:
                    if 0 <= int(di) < active.shape[1]:
                        active[:, int(di)] = True
                return active  # bool

            # Decoder gate from gate_form. Temporal mix uses state_form when it differs
            # (v26p), otherwise the historical shared-or-threshold-split path.
            if gate_form == "purity_weight":
                # v32 (purity gating): the ownership statistic IS the gate. No threshold,
                # no temperature, no schedule -- the whole curriculum machine (p, tau,
                # beta, build_gate above) is bypassed and c is used directly:
                #   decoder : masks * c, renormalized  ==  softmax_s(alpha + log c)
                #   temporal: alpha = c / max_j(c) via state_max_norm further down
                # Untrained attention gives near-uniform c (~1/S), and a uniform gate is
                # cancelled exactly by both application points (decoder renorm, max-norm),
                # so early training is the ungated baseline; the gate phases itself in as
                # attention sharpens (self-annealing, no schedule to mistune). c stays
                # detached (anti-gaming), so the gate is pure forward modulation.
                # gate_p_state has no meaning here (there is no threshold to split).
                #
                # v36: purity_normalize maps the uniform baseline c=1/K onto 0
                #   p = clip((K c - 1)/(K - 1), 0, 1)
                # so ghosts go to 0 while exclusive owners stay at 1. Decoder adds
                # a floor on g so all-zero p recovers softmax(alpha) (ungated).
                active_mask = _purity_weight_gate(gate_conf, purity_normalize, default_list)
            else:
                active_mask = build_gate(p_use)

            if state_form == "purity_weight":
                if gate_form == "purity_weight":
                    state_mask = active_mask
                else:
                    state_mask = _purity_weight_gate(
                        gate_conf, purity_normalize, default_list
                    )
            elif gate_form == "purity_weight":
                state_mask = active_mask
            else:
                # The decoder's threshold has to anneal below the smallest object's mass or
                # small objects are never representable, but at that p the gate saturates and
                # goes flat across slots, which is exactly where the temporal mix loses its
                # selectivity. `gate_p_state` decouples the two: the returned active_mask
                # (decoder, logging, contrastive, aux losses) keeps the annealed p, while the
                # predictor re-gate further down uses a threshold of its own.
                state_mask = (
                    active_mask if gate_p_state is None else build_gate(float(gate_p_state))
                )
            # v26pg: temporal mix uses π ⊙ g_dec, then max-norm. Early mass keeps
            # unused slots from advancing (the v26m curriculum on the temporal path);
            # purity still ranks the admitted slots. p=0 => g_dec=1 => same as v26p.
            # Not the v29 mix: g_dec is already a 0-1 gate, not a term inside log r.
            #
            # v26pgd: decoder masks use the same product (g ⊙ π). Snapshot the
            # un-producted statistics first so enabling both flags recouples the
            # two paths instead of squaring one side. p=0 => decoder is π.
            g_dec = active_mask
            state_stat = state_mask
            product_ok = (
                g_dec is not None
                and state_stat is not None
                and state_stat is not g_dec
            )
            if state_mul_decoder and decoder_mul_state and product_ok:
                prod = _as_float_gate(g_dec) * _as_float_gate(state_stat)
                active_mask = prod
                state_mask = prod
            else:
                if state_mul_decoder and product_ok:
                    state_mask = _as_float_gate(state_stat) * _as_float_gate(g_dec)
                if decoder_mul_state and product_ok:
                    active_mask = _as_float_gate(g_dec) * _as_float_gate(state_stat)
            if gate_detach:
                # sg(g): keep the forward gate, drop the Jacobian into (m, c).
                same = state_mask is active_mask
                if active_mask is not None and torch.is_floating_point(active_mask):
                    active_mask = active_mask.detach()
                if same:
                    state_mask = active_mask
                elif state_mask is not None and torch.is_floating_point(state_mask):
                    state_mask = state_mask.detach()
            # The corrector output is deliberately NOT gated here. Gating it as well as the
            # predictor output puts the gate twice on the same path, and because the predictor
            # is a residual block (Pred(x) = x + D) the observation then lands at g^2 while the
            # dynamics arrive at g -- a slot at g = 0.3 would admit 9% of what it sees while
            # taking 30% of its predicted motion, which is not a ratio anything asked for:
            #   both gated:  hat{x}_{t+1} = hat{x}_t + g^2 (u_t - hat{x}_t) + g D
            #   here:        hat{x}_{t+1} = hat{x}_t + g   (u_t - hat{x}_t) + g D
            # The remaining application is the predictor re-gate, which is the temporal
            # propagation step the gate is meant to control.
            #
            # Consequence to keep in mind: `state` (what the decoder, loss_ss and the velocity
            # signal all read) now carries the full correction for every slot, so a low-gate
            # slot no longer hands stale content to the decoder. The gate restricts what that
            # slot's prior -- and hence the query it corrects from -- can become, not what it
            # reports this frame.

        # Velocity of the corrector output, one step behind the displacement the predictor is
        # asked to produce. Input and target are the same quantity, which makes this a plain
        # autoregressive second-order model d_t = f(x_t, d_{t-1}) rather than one that also
        # has to learn a change of coordinates. `prev_state` is None on the first frame, which
        # leaves the predictor exactly in its velocity-free configuration there.
        vel = None
        if self.predictor_takes_vel and prev_state is not None:
            vel = updated_state - prev_state

        if self.predictor:
            if vel is not None:
                predicted_state = self.predictor(updated_state, vel=vel)
            else:
                predicted_state = self.predictor(updated_state)
        else:
            # Just pass updated_state along as prediction
            predicted_state = updated_state
        # Kept for the dynamics loss, which must supervise the predictor module itself rather
        # than the gated mix: reading the mix would scale the predictor's gradient by g, so
        # the throttled slots that most need a dynamics prior would learn one the slowest.
        predicted_pregate = predicted_state

        # The one place the mass gate enters the temporal path. Low-gate slots keep their
        # incoming prior instead of advancing, so the gate sets how fast a slot's memory is
        # allowed to track what it sees:
        #   hat{x}_{t+1} = g * Pred(u_t) + (1-g) * hat{x}_t  =  hat{x}_t + g (u_t - hat{x}_t) + g D
        # `state_max_norm` normalizes by max_s(g) here so the winning slot always advances
        # fully; the decoder keeps the raw gate, being scale-invariant after its renorm.
        #
        # Optional EMA / hold on this mix gate only (decoder still sees instantaneous π):
        #   π̃_t = m π_t + (1-m) π̃_{t-1}, then π̃_t ← max(π̃_t, hold · π̃_{t-1}).
        # m=1 and hold=0 leave state_mask untouched (and still possibly aliased with
        # active_mask). Any smoothing copies, so active_mask stays the current-frame π.
        ema_m = 1.0 if state_gate_ema is None else float(state_gate_ema)
        hold_m = 0.0 if state_gate_hold is None else float(state_gate_hold)
        if (
            state_mask is not None
            and torch.is_floating_point(state_mask)
            and (ema_m < 1.0 or hold_m > 0.0)
        ):
            state_mask = _temporal_gate_smooth(
                state_mask, state_gate_prev, ema_m, hold_m
            )
        #
        # predictor_ungated=True removes this mix, and since the corrector output is no longer
        # gated either that leaves the gate out of the temporal path entirely -- it would then
        # only reweight decoder masks. Only v18 sets it.
        identity_cos = None
        if state_mask is not None and not predictor_ungated:
            a = _max_norm(state_mask, state_max_norm).unsqueeze(-1).to(predicted_state.dtype)
            # v39i: identity check on the mix only. Decoder keeps π.
            #   c = ReLU(cos(û_t, u_t)),  ρ = π̄ ⊙ c
            #   û_{t+1} = ρ Pred(u) + (1-ρ) û
            # Applied AFTER max-norm so a purity winner with a broken identity
            # does not still advance at 1. Detached like π. Default off is v39.
            if state_identity_cos and torch.is_floating_point(state_mask):
                with torch.no_grad():
                    identity_cos = _slot_identity_cos(state, updated_state).to(
                        dtype=a.dtype
                    )
                a = a * identity_cos.unsqueeze(-1)
            predicted_state = a * predicted_state + (1.0 - a) * state

        if active_mask is None:
            # keep a consistent output tree; all slots are "active" when gating is off
            active_mask = torch.ones(
                updated_state.shape[:2], dtype=torch.bool, device=updated_state.device
            )
        if state_mask is None:
            state_mask = active_mask

        out = {
            "state": updated_state,
            "state_predicted": predicted_state,
            "state_predicted_pregate": predicted_pregate,
            "corrector": corrector_output,
            "state_attn_mask": state_attn_mask,
            "active_mask": active_mask,
            # Always present so the output tree shape is constant; equals active_mask
            # unless gate_p_state or state_gate_form gave the temporal mix its own gate.
            "state_gate": state_mask,
        }
        if gate_conf is not None:
            # detached (B, S) assignment confidence. Logging, and when
            # gate_hysteresis > 0 this is the smoothed π̃ that ScanOverTime carries
            # into the next frame's `gate_conf_prev` (recursive leaky max).
            out["gate_conf"] = gate_conf
        if mass_median_ema is not None:
            out["mass_median_ema"] = mass_median_ema
        if gate_p_eff is not None and torch.is_tensor(gate_p_eff):
            out["gate_p_eff"] = gate_p_eff.squeeze(-1)  # (B,)
        if gate_delta is not None and torch.is_tensor(gate_delta):
            out["gate_delta"] = gate_delta.squeeze(-1)  # (B,)
        if state_mask is not None and torch.is_floating_point(state_mask):
            # next-frame prev for temporal EMA / hold (already smoothed if enabled)
            out["state_gate_carry"] = state_mask.detach()
        if identity_cos is not None:
            out["identity_cos"] = identity_cos
        return out


class MapOverTime(nn.Module):
    """Wrapper applying wrapped module independently to each time step.

    Assumes batch is first dimension, time is second dimension.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args):
        batch_size = None
        seq_len = None
        flattened_args = []
        for idx, arg in enumerate(args):
            B, T = arg.shape[:2]
            if not batch_size:
                batch_size = B
            elif batch_size != B:
                raise ValueError(
                    f"Inconsistent batch size of {B} of argument {idx}, was {batch_size} before."
                )

            if not seq_len:
                seq_len = T
            elif seq_len != T:
                raise ValueError(
                    f"Inconsistent sequence length of {T} of argument {idx}, was {seq_len} before."
                )

            flattened_args.append(arg.flatten(0, 1))

        outputs = self.module(*flattened_args)

        if isinstance(outputs, Mapping):
            unflattened_outputs = {
                k: v.unflatten(0, (batch_size, seq_len)) for k, v in outputs.items()
            }
        else:
            unflattened_outputs = outputs.unflatten(0, (batch_size, seq_len))

        return unflattened_outputs


class ScanOverTime(nn.Module):
    """Wrapper applying wrapped module recurrently over time steps"""

    def __init__(
        self, module: nn.Module, next_state_key: str = "state_predicted", pass_step: bool = True
    ) -> None:
        super().__init__()
        self.module = module
        self.next_state_key = next_state_key
        self.pass_step = pass_step
        # Anchor frame chosen per sample by the last "evidence"/"random" cycle (diagnostics
        # for eval scripts; None when the last forward used another mode).
        self.last_anchor_frames: Optional[torch.Tensor] = None

    def forward(
        self,
        initial_state: torch.Tensor,
        inputs: torch.Tensor,
        cycle: Any = False,
        gate_p: Optional[float] = None,
        default_idx: Any = 0,
        gate_mode: str = "hard",
        gate_tau: Optional[float] = None,
        mass_gamma: float = 1.0,
        purity_q: Optional[float] = None,
        purity_tau: Optional[float] = None,
        gate_p_state: Optional[float] = None,
        state_max_norm: bool = False,
        predictor_ungated: bool = False,
        state_gate_ema: float = 1.0,
        state_gate_hold: float = 0.0,
        p_mode: str = "absolute",
        median_ema_momentum: float = 0.9,
        p_residual_net: Optional[nn.Module] = None,
        p_residual_alpha: float = 0.0,
        gate_form: str = "linear",
        gate_beta: Optional[float] = None,
        gate_tau_log: Optional[float] = None,
        conf_kind: str = "entropy",
        gate_detach: bool = False,
        key_inputs: Optional[torch.Tensor] = None,
        purity_normalize: bool = False,
        state_gate_form: Optional[str] = None,
        state_mul_decoder: bool = False,
        decoder_mul_state: bool = False,
        bind_inputs: Optional[torch.Tensor] = None,
        gate_hysteresis: float = 0.0,
        state_identity_cos: bool = False,
    ):
        # initial_state: batch x ...
        # inputs: batch x n_frames x ...
        seq_len = inputs.shape[1]

        gate_kwargs = dict(
            gate_p=gate_p, default_idx=default_idx, gate_mode=gate_mode, gate_tau=gate_tau,
            mass_gamma=mass_gamma, purity_q=purity_q, purity_tau=purity_tau,
            gate_p_state=gate_p_state,
            state_max_norm=state_max_norm,
            predictor_ungated=predictor_ungated,
            state_gate_ema=state_gate_ema,
            state_gate_hold=state_gate_hold,
            p_mode=p_mode, median_ema_momentum=median_ema_momentum,
            p_residual_net=p_residual_net, p_residual_alpha=p_residual_alpha,
            gate_form=gate_form, gate_beta=gate_beta, gate_tau_log=gate_tau_log,
            conf_kind=conf_kind, gate_detach=gate_detach,
            purity_normalize=purity_normalize,
            state_gate_form=state_gate_form,
            state_mul_decoder=state_mul_decoder,
            decoder_mul_state=decoder_mul_state,
            gate_hysteresis=gate_hysteresis,
            state_identity_cos=state_identity_cos,
        )

        state = initial_state
        median_ema = None
        # Previous posterior, used by the predictor as a velocity reference. Kept in sweep
        # order, so during the backward pass below it is the temporally later frame and the
        # velocity correctly points the way that sweep is predicting.
        prev_state = None
        # Smoothed confidence π̃ carried frame-to-frame for gate_hysteresis (leaky-max
        # occlusion memory). None until the first frame emits gate_conf; inert at γ=0.
        gate_conf_prev = None
        # Temporal-mix EMA / hold carry (decoder-instantaneous π). None on frame 0.
        state_gate_prev = None
        outputs = []
        for t in range(seq_len):
            kwargs = dict(gate_kwargs)
            kwargs["median_ema_prev"] = median_ema
            kwargs["prev_state"] = prev_state
            kwargs["gate_conf_prev"] = gate_conf_prev
            kwargs["state_gate_prev"] = state_gate_prev
            if key_inputs is not None:
                kwargs["key_features"] = key_inputs[:, t]
            if bind_inputs is not None:
                kwargs["bind_features"] = bind_inputs[:, t]
            if self.pass_step:
                output = self.module(state, inputs[:, t], t, **kwargs)
            else:
                output = self.module(state, inputs[:, t], **kwargs)
            outputs.append(output)
            prev_state = output["state"]
            state = output[self.next_state_key]
            if "mass_median_ema" in output:
                median_ema = output["mass_median_ema"]
            gate_conf_prev = output.get("gate_conf", gate_conf_prev)
            state_gate_prev = output.get("state_gate_carry", state_gate_prev)

        self.last_anchor_frames = None
        if cycle:
            # `cycle` selects the re-inference protocol after the forward sweep:
            #   True / "last": legacy cyclic inference -- backward sweep anchored at the
            #       LAST frame's state, all frames replaced by the backward outputs.
            #   "evidence": evidence-anchored bidirectional inference (EABI). The anchor
            #       is the frame where the most objects are intactly bound (soft active
            #       count weighted by ownership purity), the backward sweep runs
            #       anchor -> 1, and only pre-anchor frames are replaced (re-running
            #       forward from the anchor state would reproduce the forward sweep
            #       exactly, so those frames are kept as-is).
            #   "evidence_mass": EABI with the coverage-weighted statistic sum_s g*m
            #       (mass share owned by trusted slots) -- kept as an A/B variant; it
            #       ignores object COUNT, so a frame where one background slot owns
            #       everything can outscore a frame with five cleanly-bound objects.
            #   "random": EABI with a uniformly random anchor -- control run isolating
            #       the value of evidence-based anchor selection.
            mode = cycle.strip().lower() if isinstance(cycle, str) else "last"
            if mode in ("evidence", "evidence_mass", "random"):
                if mode == "random":
                    anchors = torch.randint(
                        seq_len, (inputs.shape[0],), device=inputs.device
                    )
                else:
                    anchors = _evidence_anchors(
                        outputs,
                        gate_kwargs.get("mass_gamma") or 1.0,
                        stat="mass" if mode == "evidence_mass" else "count",
                    )
                self.last_anchor_frames = anchors.detach().to("cpu")
                return self._cycle_from_anchors(
                    outputs, inputs, gate_kwargs, median_ema, anchors, key_inputs,
                    bind_inputs,
                )
            if mode != "last":
                raise ValueError(f"unknown cycle mode {cycle!r}")
            # backward pass
            ### not last frame
            new_outputs = []
            new_outputs.append(outputs[-1])
            state = outputs[-1][self.next_state_key]
            prev_state = outputs[-1]["state"]
            # continue EMA through the backward sweep (same clip statistics); the
            # hysteresis carry restarts from the anchor frame's smoothed π̃ so the
            # sweep's temporal memory runs in sweep order.
            gate_conf_prev = outputs[-1].get("gate_conf")
            state_gate_prev = outputs[-1].get("state_gate_carry")
            for t in range(seq_len - 1):
                back_t = seq_len - t - 2
                kwargs = dict(gate_kwargs)
                kwargs["median_ema_prev"] = median_ema
                kwargs["prev_state"] = prev_state
                kwargs["gate_conf_prev"] = gate_conf_prev
                kwargs["state_gate_prev"] = state_gate_prev
                if key_inputs is not None:
                    kwargs["key_features"] = key_inputs[:, back_t]
                if bind_inputs is not None:
                    kwargs["bind_features"] = bind_inputs[:, back_t]
                out = self.module(state, inputs[:, back_t], **kwargs)
                new_outputs.append(out)
                prev_state = out["state"]
                state = out[self.next_state_key]
                if "mass_median_ema" in out:
                    median_ema = out["mass_median_ema"]
                gate_conf_prev = out.get("gate_conf", gate_conf_prev)
                state_gate_prev = out.get("state_gate_carry", state_gate_prev)
            new_outputs = new_outputs[::-1]  # reverse the order of outputs
            return merge_dict_trees(new_outputs, axis=1)

        return merge_dict_trees(outputs, axis=1)

    def _cycle_from_anchors(
        self,
        outputs: List[Dict[str, Any]],
        inputs: torch.Tensor,
        gate_kwargs: Dict[str, Any],
        median_ema: Optional[torch.Tensor],
        anchors: torch.Tensor,
        key_inputs: Optional[torch.Tensor] = None,
        bind_inputs: Optional[torch.Tensor] = None,
    ):
        """Backward sweep from a per-sample anchor frame, stitched with the forward sweep.

        Semantics per sample b with anchor a_b: frames >= a_b keep the forward outputs,
        frames < a_b come from a backward sweep warm-started at the anchor's predicted
        state (mirroring the legacy cycle, which is the special case a_b = T-1). Both
        sweeps share the anchor state, so slot identities stay consistent across the
        stitch -- required for video metrics.

        Batching: the sweep iterates t = max(a)-1 .. 0 for the whole batch at once; a
        sample's rows are (re-)injected with its anchor states at its own start step
        t = a_b - 1. Rows computed before a sample's sweep begins are finite but
        meaningless and are discarded by the per-sample stitch, so mixed anchors in one
        batch cost only wasted compute, never wrong outputs.
        """
        seq_len = len(outputs)
        b = inputs.shape[0]
        forward_tree = merge_dict_trees(outputs, axis=1)
        max_anchor = int(anchors.max().item())
        if max_anchor <= 0:
            return forward_tree

        # (B, S, D) anchor states gathered per sample. The backward step out of the
        # anchor consumes the anchor's *predicted* state, exactly like the legacy cycle
        # consumes outputs[-1][next_state_key]; the posterior becomes prev_state so the
        # predictor's velocity reference stays in sweep order.
        pred_states = torch.stack([o[self.next_state_key] for o in outputs], dim=1)
        post_states = torch.stack([o["state"] for o in outputs], dim=1)
        idx = anchors.view(b, 1, 1, 1).expand(-1, 1, *pred_states.shape[2:])
        anchor_pred = pred_states.gather(1, idx).squeeze(1)
        anchor_post = post_states.gather(1, idx).squeeze(1)

        # Hysteresis carry restarts from each sample's anchor-frame smoothed π̃
        # (mirrors the anchor state warm-start); None when the forward sweep never
        # emitted gate_conf (hysteresis / conf gate off).
        anchor_conf = None
        anchor_carry = None
        gate_conf_prev = None
        state_gate_prev = None
        if outputs and outputs[0].get("gate_conf") is not None:
            conf_states = torch.stack([o["gate_conf"] for o in outputs], dim=1)
            cidx = anchors.view(b, 1, 1).expand(-1, 1, conf_states.shape[-1])
            anchor_conf = conf_states.gather(1, cidx).squeeze(1)
            gate_conf_prev = anchor_conf
        if outputs and outputs[0].get("state_gate_carry") is not None:
            carry_states = torch.stack([o["state_gate_carry"] for o in outputs], dim=1)
            gidx = anchors.view(b, 1, 1).expand(-1, 1, carry_states.shape[-1])
            anchor_carry = carry_states.gather(1, gidx).squeeze(1)
            state_gate_prev = anchor_carry

        state = anchor_pred
        prev_state = anchor_post
        new_outputs = list(outputs)  # frames >= anchor keep the forward outputs
        for t in range(max_anchor - 1, -1, -1):
            starts = (anchors == t + 1).view(b, *([1] * (state.ndim - 1)))
            state = torch.where(starts, anchor_pred, state)
            prev_state = torch.where(starts, anchor_post, prev_state)
            if anchor_conf is not None:
                cstarts = (anchors == t + 1).view(b, *([1] * (anchor_conf.ndim - 1)))
                gate_conf_prev = torch.where(cstarts, anchor_conf, gate_conf_prev)
            if anchor_carry is not None:
                # per-sample re-injection at the sweep start, like state/conf above;
                # without it a late-anchor sample inherits carry contaminated by steps
                # computed before its sweep began (matters only when EMA/hold is on).
                gstarts = (anchors == t + 1).view(b, *([1] * (anchor_carry.ndim - 1)))
                state_gate_prev = torch.where(gstarts, anchor_carry, state_gate_prev)
            kwargs = dict(gate_kwargs)
            kwargs["median_ema_prev"] = median_ema
            kwargs["prev_state"] = prev_state
            kwargs["gate_conf_prev"] = gate_conf_prev
            kwargs["state_gate_prev"] = state_gate_prev
            if key_inputs is not None:
                kwargs["key_features"] = key_inputs[:, t]
            if bind_inputs is not None:
                kwargs["bind_features"] = bind_inputs[:, t]
            # no time_step: the first-step corrector args never apply on re-inference
            # sweeps (matches the legacy cycle)
            out = self.module(state, inputs[:, t], **kwargs)
            new_outputs[t] = out
            prev_state = out["state"]
            state = out[self.next_state_key]
            if "mass_median_ema" in out:
                median_ema = out["mass_median_ema"]
            gate_conf_prev = out.get("gate_conf", gate_conf_prev)
            state_gate_prev = out.get("state_gate_carry", state_gate_prev)

        backward_tree = merge_dict_trees(new_outputs, axis=1)
        use_backward = (
            torch.arange(seq_len, device=anchors.device).view(1, seq_len)
            < anchors.view(b, 1)
        )
        return _stitch_trees(forward_tree, backward_tree, use_backward)


def _evidence_anchors(
    outputs: List[Dict[str, Any]], mass_gamma: float, window: int = 5, stat: str = "count"
) -> torch.Tensor:
    """Per-sample anchor frame: argmax of the gate's own per-frame evidence.

    stat="count" (default): E_t = sum_s g_{t,s} * c_{t,s}, the soft number of active
    slots weighted by ownership purity c = sum_f A~^2 / sum_f A~. This implements "the
    frame where the most objects are INTACTLY present": sum_s g counts trusted slots,
    purity discounts objects that are only partially visible / entering / occluded
    (mixed ownership at their boundary lowers c; a fully-owned object scores ~1 at any
    size). The anchor exists to hand the backward sweep a state that has bound as many
    of the video's objects as possible, which is a count, not a mass share.

    stat="mass": E_t = sum_s g_{t,s} * m_{t,s}, the fraction of (gamma-sharpened)
    attention mass owned by trusted slots. Since sum_s m_s = 1 per frame this measures
    how confidently the frame is EXPLAINED, not how many objects are held -- one
    trusted background slot owning everything can outscore five cleanly-bound objects.
    Kept as an A/B variant for the eval.

    Both signals come from the forward sweep's outputs, so anchor selection costs no
    extra model evaluation. E_t is mean-smoothed over a `window`-frame neighborhood
    (replicate-padded) before the argmax so a single-frame noise spike cannot become
    the anchor. With gating disabled active_mask is all-ones and gamma=1 purity/mass
    are frame-independent constants only in degenerate cases; anchor quality then just
    falls back to whatever the statistic sees.
    """
    gates = torch.stack([o["active_mask"].float() for o in outputs], dim=1)  # (B, T, S)
    att = torch.stack([o["state_attn_mask"].float() for o in outputs], dim=1)  # (B,T,S,F)
    gamma = float(mass_gamma or 1.0)
    if gamma != 1.0:
        att = att.pow(gamma)
        att = att / att.sum(dim=2, keepdim=True).clamp_min(1e-8)
    if stat == "mass":
        per_slot = att.sum(dim=-1) / att.shape[-1]  # coverage m, (B, T, S)
    else:
        per_slot = (att * att).sum(dim=-1) / att.sum(dim=-1).clamp_min(1e-8)  # purity c
    evidence = (gates * per_slot).sum(dim=-1)  # (B, T)

    t_len = evidence.shape[1]
    k = min(window, t_len)
    if k % 2 == 0:
        k = max(k - 1, 1)
    if k > 1:
        kernel = torch.ones(1, 1, k, device=evidence.device, dtype=evidence.dtype) / k
        padded = torch.nn.functional.pad(
            evidence.unsqueeze(1), (k // 2, k // 2), mode="replicate"
        )
        evidence = torch.nn.functional.conv1d(padded, kernel).squeeze(1)
    return evidence.argmax(dim=1)  # (B,)


def _stitch_trees(
    forward_tree: Mapping, backward_tree: Mapping, use_backward: torch.Tensor
):
    """Per-(sample, frame) select between two stacked output trees.

    `use_backward` is (B, T) bool; leaves shaped (B, T, ...) are selected elementwise,
    anything else (non-tensors, oddly shaped leaves) keeps the forward version.
    """
    out = {}
    for key, fwd in forward_tree.items():
        bwd = backward_tree[key]
        if isinstance(fwd, Mapping):
            out[key] = _stitch_trees(fwd, bwd, use_backward)
        elif (
            isinstance(fwd, torch.Tensor)
            and fwd.ndim >= 2
            and fwd.shape[:2] == use_backward.shape
        ):
            mask = use_backward.view(*use_backward.shape, *([1] * (fwd.ndim - 2)))
            out[key] = torch.where(mask, bwd, fwd)
        else:
            out[key] = fwd
    return out


def merge_dict_trees(trees: List[Mapping], axis: int = 0):
    """Stack all leafs given a list of dictionaries trees.

    Example:
    x = merge_dict_trees([
        {
            "a": torch.ones(2, 1),
            "b": {"x": torch.ones(2, 2)}
        },
        {
            "a": torch.ones(3, 1),
            "b": {"x": torch.ones(1, 2)}
        }
    ])

    x == {
        "a": torch.ones(5, 1),
        "b": {"x": torch.ones(3, 2)}
    }
    """
    out = {}
    if len(trees) > 0:
        ref_tree = trees[0]
        for key, value in ref_tree.items():
            values = [tree[key] for tree in trees]
            if isinstance(value, torch.Tensor):
                out[key] = torch.stack(values, axis)
            elif isinstance(value, Mapping):
                out[key] = merge_dict_trees(values, axis)
            else:
                out[key] = values

    return out
