import math
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch
from torch import nn

from slotcurri.modules.networks import MLP, TransformerEncoder
from slotcurri.modules.usage_redistribute import sparsemax
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


def _src_gate_self_weights(gate: torch.Tensor) -> torch.Tensor:
    """Src-gate with ungated diagonal: A_ij = g_j (i≠j), A_ii = 1.

    Ghost queries can attend to their own key; live off-diagonal still uses g_j.
    (B, S) -> (B, S, S). Detached. Bool gates become 0/1 floats first.
    """
    g = gate.detach()
    if g.dtype == torch.bool:
        g = g.to(dtype=torch.float32)
    n_slots = g.shape[-1]
    # A[b, i, j] = g[b, j]
    a = g.unsqueeze(-2).expand(g.shape[0], n_slots, n_slots).clone()
    eye = torch.eye(n_slots, device=g.device, dtype=a.dtype)
    return a * (1.0 - eye) + eye


def _pair_isolate_weights(gate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Same-group Pred SA: A_ij = π̄_i π̄_j + (1-π̄_i)(1-π̄_j).

    Live queries see live keys, ghost queries see ghost keys, cross terms 0.
    Detached. Bool gates are already 0/1 so they skip max-norm.
    """
    g = gate.detach()
    if g.dtype == torch.bool:
        gbar = g.to(dtype=torch.float32)
    else:
        gbar = g.float()
        gbar = gbar / gbar.amax(dim=-1, keepdim=True).clamp_min(float(eps))
    live = gbar.unsqueeze(-1) * gbar.unsqueeze(-2)
    dead = (1.0 - gbar).unsqueeze(-1) * (1.0 - gbar).unsqueeze(-2)
    return live + dead


def _pi_gamma(gate: Optional[torch.Tensor], gamma: float) -> Optional[torch.Tensor]:
    """Eval-only π^γ. 1.0 is a no-op. Bool gates stay 0/1.

    Applied to the scalar occupancy before decoder / mix / Pred. With
    state_max_norm, mix uses (π^γ)/max(π^γ) = (π/max π)^γ.
    """
    if gate is None:
        return None
    g = 1.0 if gamma is None else float(gamma)
    if g == 1.0:
        return gate
    if g <= 0.0:
        raise ValueError(f"pi_gamma must be > 0, got {g}")
    if gate.dtype == torch.bool:
        return gate
    return gate.clamp_min(0.0).pow(g)


def _eval_hard_threshold(gate: Optional[torch.Tensor], thresh: float) -> Optional[torch.Tensor]:
    """Eval-only binarize: float π -> bool 1{π >= thresh}. Bool / None / thresh<=0 pass through.

    purity_weight ignores gate_mode, so hard dormancy at test time has to be applied
    after π is built. Decoder then takes the bool path (masked softmax); the mix
    and Pred src-gate see 0/1.
    """
    if gate is None or thresh is None or float(thresh) <= 0.0:
        return gate
    if gate.dtype == torch.bool:
        return gate
    return gate >= float(thresh)


def _perron_spatial_gate(
    pi: torch.Tensor, q1: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Decoder gate π_s * q̃_{1,s}. q1 is the Perron mode of the same G_s as π.

    q1 is L2-unit, so raw values confound support size with occupancy. Per-slot
    max-norm keeps the occupancy scale in π and uses q1 only as a spatial
    profile in [0, 1]. Sign is flipped if a numerical inversion left q1 in the
    negative cone (power iteration from +ones is nonnegative).

    pi: (B, S), q1: (B, S, N) or (B, S, N, 1) -> (B, S, N)
    """
    q = q1.squeeze(-1) if q1.ndim == 4 else q1
    q = q.to(dtype=pi.dtype)
    flip = (q.sum(dim=-1, keepdim=True) < 0).to(dtype=q.dtype)
    q = q * (1.0 - 2.0 * flip)
    q = q.clamp_min(0.0)
    q = q / q.amax(dim=-1, keepdim=True).clamp_min(float(eps))
    return pi.unsqueeze(-1) * q


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


def _leaky_max_gate(
    cur: torch.Tensor, prev: Optional[torch.Tensor], gamma: float
) -> torch.Tensor:
    """π̃_t = max(π_t, γ π̃_{t-1}). γ=0 or a missing prev is identity."""
    if prev is None or float(gamma) <= 0.0:
        return cur
    return torch.maximum(cur, float(gamma) * prev.detach())


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
    Columns stay separate tensors until `stack` so later writes cannot bump
    the version of an earlier column (live n8 impurity autograd).
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


_PROJ_CACHE: Dict[tuple, torch.Tensor] = {}
_EIGH_DIM = 96


def _orthonormal_proj(
    d_in: int, d_out: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Fixed orthonormal D→d map (seeded QR). Cached per (D, d, device, dtype)."""
    key = (int(d_in), int(d_out), str(device), str(dtype))
    cached = _PROJ_CACHE.get(key)
    if cached is not None:
        return cached
    gen = torch.Generator(device="cpu")
    gen.manual_seed(37000 + int(d_in) * 10007 + int(d_out))
    raw = torch.randn(int(d_in), int(d_out), generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    proj = q.to(device=device, dtype=dtype)
    _PROJ_CACHE[key] = proj
    return proj


def _weighted_grams(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """C_s = Z^T diag(a^2) Z as (B, S, d, d). W = a ⊙ Z, C = W^T W.

    Do not einsum `bnd,bsn,bne->bsde`: that path materializes a (B,S,N,d,d)
    intermediate (several GB at YTVIS/MOVi shapes) even when d=64.
    """
    w = a.unsqueeze(-1) * z.unsqueeze(1)
    cmat = torch.matmul(w.transpose(-1, -2), w)
    return 0.5 * (cmat + cmat.transpose(-1, -2))


def _top2_gap_c_eigh(z: torch.Tensor, a: torch.Tensor, eps: float) -> torch.Tensor:
    """λ1-λ2 of C_s = Z^T diag(a^2) Z by forming the d×d Gram. z: (B, N, d)."""
    dim = z.shape[-1]
    if dim < 2:
        lam1 = torch.einsum("bnd,bsn,bnd->bs", z, a * a, z).clamp_min(0.0)
        return lam1
    cmat = _weighted_grams(z, a)
    eye = torch.eye(dim, device=z.device, dtype=z.dtype).mul_(float(eps))
    evals = torch.linalg.eigvalsh(cmat + eye)
    return (evals[..., -1] - evals[..., -2]).clamp_min(0.0)


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


def _top2_eigs_c_eigh(
    z: torch.Tensor, a: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """λ1 ≥ λ2 of C_s = Z^T diag(a^2) Z by forming the d×d Gram. Live in a.

    No ridge: empty slots must stay λ1=λ2=0 so ρ=λ2/(λ1+eps)→0. The v37
    gate gap λ1-λ2 is invariant to eps I; the ratio is not.
    z: (B, N, d), a: (B, S, N) -> (B, S), (B, S).
    """
    dim = z.shape[-1]
    if dim < 2:
        lam1 = torch.einsum("bnd,bsn,bnd->bs", z, a * a, z).clamp_min(0.0)
        return lam1, torch.zeros_like(lam1)
    evals = torch.linalg.eigvalsh(_weighted_grams(z, a))
    return evals[..., -1].clamp_min(0.0), evals[..., -2].clamp_min(0.0)


def _top2_eigs_c_from_z(
    z: torch.Tensor, a: torch.Tensor, n_iter: int, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """λ1 ≥ λ2 of C_s via the same 2D subspace / Ritz as `_top2_gap_c_from_z`."""
    bsz, n_slots, _ = a.shape
    dim = z.shape[-1]
    if dim < 2:
        lam1 = torch.einsum("bnd,bsn,bnd->bs", z, a * a, z).clamp_min(0.0)
        return lam1, torch.zeros_like(lam1)

    a2 = a * a
    ones = torch.ones(bsz, n_slots, dim, 1, device=z.device, dtype=z.dtype)
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
    return evals[..., 1].clamp_min(0.0), evals[..., 0].clamp_min(0.0)


def spectral_slot_purity(
    att: torch.Tensor,
    features: torch.Tensor,
    eps: float = 1e-6,
    chunk_size: int = 256,
    n_iter: int = 16,
    divide_by_mass: bool = True,
    proj_dim: int = 0,
    ratio: bool = False,
) -> torch.Tensor:
    """Detached spectral slot purity π_s (v37 / v37g). No gamma; no 1/K here.

    Last-iter ownership a_s = A_{:,s} and L2-normalized bind tokens Z give

        C_s = Z^T diag(a_s^2) Z

    then the gap λ1-λ2. v37 (`divide_by_mass=True`) returns
    clip(gap / (sum_i A_{i,s} + eps), 0, 1): rank-1 exclusive ownership recovers
    Σ a^2 / Σ A. The v36 1/K map, if enabled, is applied later in
    `_purity_weight_gate`, not here. v37g (`divide_by_mass=False`) returns the
    raw gap, same as v38/v39's λ1-λ2 on their graphs: occupancy stays in λ1,
    two feature modes still raise λ2. v37r (`ratio=True`) returns
    (λ1-λ2)/(λ1+λ2+eps) in [0, 1], the bounded map of λ1/λ2:
    (r-1)/(r+1) with r=λ1/λ2. Empty / isotropic slots (λ1≈λ2) go to 0
    without a threshold; raw λ1/λ2 is not used (it explodes at λ2=0).
    a^2 is the within-slot moment (rank-1 ownership); across-slot gamma
    is not applied.
    features is X^bind at DINO dim (Key-only curriculum's bind tokens; raw
    backbone tokens when the curriculum is off / mix=1).

    `proj_dim` > 0 and < D maps Z through a fixed orthonormal D→d matrix
    (seeded QR) then re-normalizes. Person-vs-car modes survive a 64-d
    Johnson–Lindenstrauss map. Top-2 of C_s is 16 subspace iters + MGS
    (same solver family as v39 n8), not batched eigvalsh of the d×d Gram.
    `proj_dim=0` is the original full-D path. `chunk_size` bounds peak
    memory on that path.
    """
    if features is None:
        raise ValueError(
            "conf_kind='spectral' / 'spectral_gap' / 'spectral_ratio' "
            "requires bind_features "
            "(encoder backbone tokens)"
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
        d_out = int(proj_dim)
        if d_out > 0 and d_out < z.shape[-1]:
            z = torch.nn.functional.normalize(
                z @ _orthonormal_proj(z.shape[-1], d_out, z.device, z.dtype),
                dim=-1,
            )
        a = att.float()
        mass = a.sum(dim=-1)
        # Always 2D subspace / MGS for λ1-λ2. Batched eigvalsh of C_s is the
        # v37-only tax vs v39 n8: MOVi is ~256 frames × 11 slots = 2816 of
        # 64×64 syev, YTVIS 128×7. proj_dim only sets the matvec width; it
        # does not pick the eigensolver. chunk_size bounds peak memory when
        # proj_dim=0 (full DINO dim).
        bsz = a.shape[0]
        step = max(int(chunk_size), 1)
        if ratio:
            if z.shape[-1] < 2:
                lam1, lam2 = _top2_eigs_c_eigh(z, a)
            else:
                lam1 = a.new_empty(bsz, a.shape[1])
                lam2 = a.new_empty(bsz, a.shape[1])
                for i in range(0, bsz, step):
                    l1, l2 = _top2_eigs_c_from_z(
                        z[i : i + step], a[i : i + step], n_iter, eps
                    )
                    lam1[i : i + step] = l1
                    lam2[i : i + step] = l2
            pi = ((lam1 - lam2) / (lam1 + lam2 + float(eps))).clamp(0.0, 1.0)
        else:
            if z.shape[-1] < 2:
                gaps = _top2_gap_c_eigh(z, a, eps)
            else:
                gaps = a.new_empty(bsz, a.shape[1])
                for i in range(0, bsz, step):
                    gaps[i : i + step] = _top2_gap_c_from_z(
                        z[i : i + step], a[i : i + step], n_iter, eps
                    )
            if divide_by_mass:
                pi = (gaps / (mass + float(eps))).clamp(0.0, 1.0)
            else:
                pi = gaps.clamp_min(0.0)
    return pi.detach().to(dtype=att.dtype)


def spectral_cs_impurity(
    att: torch.Tensor,
    features: torch.Tensor,
    eps: float = 1e-6,
    chunk_size: int = 64,
    n_iter: int = 16,
    proj_dim: int = 0,
) -> torch.Tensor:
    """Differentiable v37 Gram impurity ρ_s = max(λ2, 0) / (max(λ1, 0) + eps).

    Same C_s = Z^T diag(a_s^2) Z as the v37 gate, but the loss uses the
    scale-free ratio (no /mass, no across-slot gamma). Z is detached
    (frozen DINO). a is live last-iter softmax so two feature modes can
    split a slot. att (B, S, N), features (B, N, D) -> rho (B, S).
    `proj_dim` matches the v37 gate: 64-d JL, then 16 subspace iters +
    2D Ritz for (λ1, λ2). Not batched eigvalsh of the d×d Gram.
    `proj_dim=0` is full D. `chunk_size` bounds peak memory on that path.
    """
    if features is None:
        raise ValueError(
            "spectral_cs_impurity requires bind_features "
            "(encoder backbone tokens)"
        )
    if att.ndim != 3 or features.ndim != 3:
        raise ValueError(
            f"C_s impurity expects att (B, S, N) and features (B, N, D), "
            f"got {tuple(att.shape)} and {tuple(features.shape)}"
        )
    if att.shape[0] != features.shape[0] or att.shape[-1] != features.shape[1]:
        raise ValueError(
            f"C_s impurity batch/token mismatch: att {tuple(att.shape)} vs "
            f"features {tuple(features.shape)}"
        )

    amp_ctx = (
        torch.cuda.amp.autocast(enabled=False) if att.is_cuda else nullcontext()
    )
    with amp_ctx:
        z = torch.nn.functional.normalize(features.float(), dim=-1).detach()
        d_out = int(proj_dim)
        if d_out > 0 and d_out < z.shape[-1]:
            z = torch.nn.functional.normalize(
                z @ _orthonormal_proj(z.shape[-1], d_out, z.device, z.dtype),
                dim=-1,
            )
        a = att.float()
        # Same 2D subspace as the v37 gate. Batched eigvalsh of C_s is the
        # old tax; proj_dim only sets the matvec width.
        if z.shape[-1] < 2:
            lam1, lam2 = _top2_eigs_c_eigh(z, a)
        else:
            step = max(int(chunk_size), 1)
            lam1_chunks = []
            lam2_chunks = []
            for i in range(0, a.shape[0], step):
                l1, l2 = _top2_eigs_c_from_z(
                    z[i : i + step], a[i : i + step], n_iter, eps
                )
                lam1_chunks.append(l1)
                lam2_chunks.append(l2)
            lam1 = torch.cat(lam1_chunks, dim=0)
            lam2 = torch.cat(lam2_chunks, dim=0)
        rho = lam2.clamp_min(0.0) / (lam1.clamp_min(0.0) + float(eps))
    return rho.to(dtype=att.dtype)


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


def _pad1_hw(x: torch.Tensor) -> torch.Tensor:
    """Zero-pad spatial dims 1,2 by 1. Same values as eight `_shift_hw` calls."""
    tail = x.ndim - 3
    pad = (0, 0) * tail + (1, 1, 1, 1)
    return torch.nn.functional.pad(x, pad)


def _n8_gather(xp: torch.Tensor, dy: int, dx: int, height: int, width: int) -> torch.Tensor:
    """Slice +1-padded `xp` so out[y, x] = orig[y-dy, x-dx] (0 outside).

    Identical to `_shift_hw(orig, dy, dx)` when `xp = _pad1_hw(orig)`.
    """
    return xp[:, 1 - dy : 1 - dy + height, 1 - dx : 1 - dx + width]


def _n8_weighted_sum(field: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """acc[y,x] = Σ_i weights[i,y,x] * field[y-dy_i, x-dx_i]; offset order is `_N8_OFFSETS`.

    field: (B, H, W, C), weights: (B, 8, H, W). One pad instead of eight `_shift_hw`.
    """
    height, width = field.shape[1], field.shape[2]
    fp = _pad1_hw(field)
    acc = field.new_zeros(field.shape)
    for i, (dy, dx) in enumerate(_N8_OFFSETS):
        acc = acc + weights[:, i].unsqueeze(-1) * _n8_gather(fp, dy, dx, height, width)
    return acc


def _n8_affinity(z: torch.Tensor, eps: float):
    """Precompute 8-neighbor ReLU-cosine weights and d^{-1/2}.

    z: (B, N, D) L2-normalized -> d_inv (B, H, W), weights (B, 8, H, W).
    Affinity is fixed for the eigensolve, so matvecs must not recompute z·z_nb.
    Same formula as eight `_shift_hw` dots; pad Z once.
    """
    bsz, n_tokens, _ = z.shape
    grid = _square_grid(n_tokens)
    z_hw = z.view(bsz, grid, grid, -1)
    zp = _pad1_hw(z_hw)
    weights = []
    deg = z.new_zeros(bsz, grid, grid)
    for dy, dx in _N8_OFFSETS:
        r = (z_hw * _n8_gather(zp, dy, dx, grid, grid)).sum(dim=-1).clamp_min(0.0)
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
    acc = _n8_weighted_sum(w_hw, weights)
    return (d_inv.unsqueeze(-1) * acc).view(bsz, n_tokens, n_c)


def _apply_r_n8(weights: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Unnormalized 8-neighbor affinity R v. Never forms N×N.

    weights: (B, 8, H, W), vec: (B, N, C) -> (B, N, C).
    """
    bsz, n_tokens, n_c = vec.shape
    grid = weights.shape[2]
    v_hw = vec.view(bsz, grid, grid, n_c)
    return _n8_weighted_sum(v_hw, weights).view(bsz, n_tokens, n_c)


def _support_threshold(
    a: torch.Tensor, support_rel: float, eps: float
) -> torch.Tensor:
    """Zero patches below support_rel * max_j a_j. support_rel<=0 keeps a."""
    if float(support_rel) <= 0.0:
        return a
    thr = float(support_rel) * a.amax(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(a >= thr, a, torch.zeros_like(a))


def _induced_n8_d_inv(
    weights: torch.Tensor, a: torch.Tensor, eps: float
) -> torch.Tensor:
    """D^{-1/2} of W = diag(a) R diag(a). a: (B, S, N) -> (B, S, N).

    Isolated / below-eps rows get d_inv=0 so they drop out of S_ind.
    """
    packed = a.permute(0, 2, 1)
    ra = _apply_r_n8(weights, packed).permute(0, 2, 1)
    deg = a * ra
    alive = (deg > float(eps)).to(dtype=deg.dtype)
    return deg.clamp_min(eps).rsqrt() * alive.detach()


def _apply_s_ind_n8(
    weights: torch.Tensor,
    d_inv: torch.Tensor,
    a: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Induced S v = d_inv ⊙ a ⊙ R(a ⊙ d_inv ⊙ v). Re-normalized on support."""
    bsz, n_slots, n_tokens = a.shape
    n_k = basis.shape[-1]
    av = a.unsqueeze(-1) * d_inv.unsqueeze(-1) * basis
    packed = av.permute(0, 2, 1, 3).reshape(bsz, n_tokens, n_slots * n_k)
    rav = _apply_r_n8(weights, packed)
    rav = rav.view(bsz, n_tokens, n_slots, n_k).permute(0, 2, 1, 3)
    return d_inv.unsqueeze(-1) * a.unsqueeze(-1) * rav


def _top2_induced_n8(
    z: torch.Tensor,
    a: torch.Tensor,
    n_iter: int,
    eps: float,
    support_rel: float,
):
    """Algebraic λ1 ≥ λ2 of the induced n8 normalized adjacency S_ind."""
    _, weights = _n8_affinity(z, eps)
    a_s = _support_threshold(a, support_rel, eps)
    d_inv = _induced_n8_d_inv(weights, a_s, eps)

    def apply_g(basis: torch.Tensor) -> torch.Tensor:
        return _apply_s_ind_n8(weights, d_inv, a_s, basis)

    return _top2_algebraic_g(apply_g, a_s, n_iter, eps), a_s


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
    return_q1: bool = False,
    lam1_only: bool = False,
):
    """Algebraic λ1 ≥ λ2 of G_s given Gv = apply_g(v).

    G is not PSD (zero diagonal). For a nonnegative symmetric matrix the
    spectral radius is λ1, so |λ_min| ≤ λ1. A +ones power iteration gives the
    Perron pair (q1, q1^T G q1). v39lam1 (`lam1_only`) stops there: occupancy
    only needs that pair, and the k=8 Ritz below exists to resolve λ2.

    Otherwise shift H = G + (λ1 + eps) I so H is PSD with condition ~2, then
    a k=8 subspace iteration + Rayleigh-Ritz; the two largest Ritz values
    minus the shift are λ1, λ2 of G. Matvecs never form G.
    """
    bsz, n_slots, n_tokens = a.shape
    zero = a.new_zeros(bsz, n_slots)
    if n_tokens < 2:
        if return_q1:
            q0 = torch.ones(
                bsz, n_slots, n_tokens, 1, device=a.device, dtype=a.dtype
            )
            return zero, zero, q0
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
    if lam1_only:
        if return_q1:
            return lam1, zero, q1
        return lam1, zero
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
    if return_q1:
        return lam[..., 1], lam[..., 0], q1
    return lam[..., 1], lam[..., 0]


def _top2_algebraic_diag_s_diag(
    s_mat: torch.Tensor, a: torch.Tensor, n_iter: int, eps: float
):
    """Algebraic λ1 ≥ λ2 of G_s = diag(a_s) S diag(a_s) for dense S (v38)."""

    def apply_g(basis: torch.Tensor) -> torch.Tensor:
        return _apply_g_diag_s_diag(s_mat, a, basis)

    return _top2_algebraic_g(apply_g, a, n_iter, eps)


def _top2_algebraic_n8(
    z: torch.Tensor,
    a: torch.Tensor,
    n_iter: int,
    eps: float,
    return_q1: bool = False,
    lam1_only: bool = False,
):
    """Algebraic λ1 ≥ λ2 of G_s on the 8-neighbor ReLU-cosine S (v39).

    lam1_only is v39lam1: Perron power pair, no k=8 Ritz / λ2.
    """
    d_inv, weights = _n8_affinity(z, eps)

    def apply_g(basis: torch.Tensor) -> torch.Tensor:
        return _apply_g_n8(d_inv, weights, a, basis)

    return _top2_algebraic_g(
        apply_g, a, n_iter, eps, return_q1=return_q1, lam1_only=lam1_only
    )


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


def spectral_graph_n8_eigs(
    att: torch.Tensor,
    features: torch.Tensor,
    eps: float = 1e-6,
    chunk_size: int = 32,
    n_iter: int = 16,
    divide_by_lambda1: bool = False,
    l1_minus_imp: bool = False,
    use_lambda1: bool = False,
    return_q1: bool = False,
):
    """Detached n8 spectrum: π, λ1, λ2 each (B, S). π is the v39 / v39n / v39s / v39lam1 gate."""
    if sum(bool(x) for x in (divide_by_lambda1, l1_minus_imp, use_lambda1)) > 1:
        raise ValueError(
            "spectral_graph_n8_eigs: divide_by_lambda1, "
            "l1_minus_imp and use_lambda1 are mutually exclusive"
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
        lam1_out = a.new_empty(bsz, a.shape[1])
        lam2_out = a.new_empty(bsz, a.shape[1])
        q1_out = a.new_empty(bsz, a.shape[1], a.shape[2]) if return_q1 else None
        step = max(int(chunk_size), 1)
        for i in range(0, bsz, step):
            z_c = z[i : i + step]
            a_c = a[i : i + step]
            packed = _top2_algebraic_n8(
                z_c,
                a_c,
                n_iter,
                eps,
                return_q1=return_q1,
                lam1_only=use_lambda1,
            )
            if return_q1:
                lam1, lam2, q1 = packed
                q1_out[i : i + step] = q1.squeeze(-1)
            else:
                lam1, lam2 = packed
            lam2p = lam2.clamp_min(0.0)
            lam1_out[i : i + step] = lam1
            lam2_out[i : i + step] = lam2
            if use_lambda1:
                pi[i : i + step] = lam1.clamp_min(0.0)
            elif l1_minus_imp:
                pi[i : i + step] = (lam1 - lam2p / (lam1 + float(eps))).clamp_min(0.0)
            elif divide_by_lambda1:
                pi[i : i + step] = (
                    (lam1 - lam2p).clamp_min(0.0) / lam1.clamp_min(float(eps))
                ).clamp(0.0, 1.0)
            else:
                pi[i : i + step] = (lam1 - lam2p).clamp_min(0.0)
    dtype = att.dtype
    out = (
        pi.detach().to(dtype=dtype),
        lam1_out.detach().to(dtype=dtype),
        lam2_out.detach().to(dtype=dtype),
    )
    if return_q1:
        return out + (q1_out.detach().to(dtype=dtype),)
    return out


def spectral_graph_n8_slot_purity(
    att: torch.Tensor,
    features: torch.Tensor,
    eps: float = 1e-6,
    chunk_size: int = 32,
    n_iter: int = 16,
    divide_by_lambda1: bool = False,
    l1_minus_imp: bool = False,
    use_lambda1: bool = False,
) -> torch.Tensor:
    """Detached 8-neighbor relation-graph spectral purity π_s (v39 / v39n / v39s / v39lam1).

    Same G_s = diag(a) S diag(a) as v38, but R is ReLU-cosine on 8-neighbors
    only (no dense N×N). Last-iter softmax a is used as-is (no argmax).

        π_s = λ1 - max(λ2, 0)                         # v39
        π_s = (λ1 - max(λ2, 0)) / max(λ1, eps)        # v39n (divide_by_lambda1)
        π_s = [λ1 - max(λ2, 0) / (λ1 + eps)]_+        # v39s (l1_minus_imp)
        π_s = max(λ1, 0)                              # v39lam1 (use_lambda1)
                                                      # Perron power Rayleigh, not k=8 Ritz

    Same gap as v38; only R is 8-neighbor. Z is L2-normalized X^bind.
    π stays detached. Curriculum P is unchanged (global ReLU-cosine);
    this graph is gate-only. The relative form is in [0, 1] and drops
    the G_s scale (ghosts no longer die from a_i a_j). v39s keeps the
    G_s scale (λ1) and subtracts relative impurity λ2⁺/(λ1+ε).
    v39lam1 keeps the G_s scale, ignores λ2, and uses the power pair
    (q1^T G q1, q1) instead of the gap solver's Ritz λ1.
    """
    pi, _, _ = spectral_graph_n8_eigs(
        att,
        features,
        eps=eps,
        chunk_size=chunk_size,
        n_iter=n_iter,
        divide_by_lambda1=divide_by_lambda1,
        l1_minus_imp=l1_minus_imp,
        use_lambda1=use_lambda1,
    )
    return pi


def _induced_n8_check_shapes(att: torch.Tensor, features: torch.Tensor, name: str):
    if features is None:
        raise ValueError(
            f"{name} requires bind_features (encoder backbone tokens)"
        )
    if att.ndim != 3 or features.ndim != 3:
        raise ValueError(
            f"{name} expects att (B, S, N) and features (B, N, D), "
            f"got {tuple(att.shape)} and {tuple(features.shape)}"
        )
    if att.shape[0] != features.shape[0] or att.shape[-1] != features.shape[1]:
        raise ValueError(
            f"{name} batch/token mismatch: att {tuple(att.shape)} vs "
            f"features {tuple(features.shape)}"
        )
    _square_grid(att.shape[-1])


def _induced_n8_pi_from_eigs(
    lam1: torch.Tensor, lam2: torch.Tensor, a_s: torch.Tensor, eps: float
) -> torch.Tensor:
    """π = λ1-λ2⁺ of S_ind. Empty support → 0; one patch → 1 (exclusive)."""
    gap = (lam1 - lam2.clamp_min(0.0)).clamp_min(0.0)
    n_sup = (a_s > float(eps)).sum(dim=-1)
    pi = torch.where(n_sup <= 0, torch.zeros_like(gap), gap)
    return torch.where(n_sup == 1, torch.ones_like(pi), pi)


def spectral_graph_n8_induced_eigs(
    att: torch.Tensor,
    features: torch.Tensor,
    eps: float = 1e-6,
    chunk_size: int = 32,
    n_iter: int = 16,
    support_rel: float = 0.0,
):
    """Detached induced n8 spectrum: π, λ1, λ2 each (B, S).

    Support Ω_s = {i: a_{s,i} ≥ support_rel * max_j a_{s,j}} (support_rel=0
    keeps every positive a). On Ω, W = diag(a) R diag(a) with n8 ReLU-cosine
    R, then S_ind is re-normalized with those induced degrees:

        S_ind = D_Ω^{-1/2} W D_Ω^{-1/2}
        π_s = λ1 - max(λ2, 0)

    Two n8-components ⇒ λ1=λ2=1 ⇒ π=0, independent of blob size.
    One connected blob ⇒ λ1=1 > λ2 ⇒ π = μ2 of the normalized Laplacian.
    Curriculum P stays global. This graph is gate/loss only.
    """
    _induced_n8_check_shapes(att, features, "conf_kind='spectral_graph_n8_ind'")
    amp_ctx = (
        torch.cuda.amp.autocast(enabled=False) if att.is_cuda else nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        z = torch.nn.functional.normalize(features.float(), dim=-1)
        a = att.float()
        bsz, n_slots, _ = a.shape
        pi = a.new_empty(bsz, n_slots)
        lam1_out = a.new_empty(bsz, n_slots)
        lam2_out = a.new_empty(bsz, n_slots)
        step = max(int(chunk_size), 1)
        for i in range(0, bsz, step):
            (lam1, lam2), a_s = _top2_induced_n8(
                z[i : i + step],
                a[i : i + step],
                n_iter,
                eps,
                support_rel,
            )
            lam1_out[i : i + step] = lam1
            lam2_out[i : i + step] = lam2
            pi[i : i + step] = _induced_n8_pi_from_eigs(lam1, lam2, a_s, eps)
    dtype = att.dtype
    return (
        pi.detach().to(dtype=dtype),
        lam1_out.detach().to(dtype=dtype),
        lam2_out.detach().to(dtype=dtype),
    )


def spectral_graph_n8_induced_purity(
    att: torch.Tensor,
    features: torch.Tensor,
    eps: float = 1e-6,
    chunk_size: int = 32,
    n_iter: int = 16,
    support_rel: float = 0.0,
) -> torch.Tensor:
    """Detached induced-n8 connectivity purity π_s (v39ind gate statistic)."""
    pi, _, _ = spectral_graph_n8_induced_eigs(
        att,
        features,
        eps=eps,
        chunk_size=chunk_size,
        n_iter=n_iter,
        support_rel=support_rel,
    )
    return pi


def spectral_graph_n8_induced_impurity(
    att: torch.Tensor,
    features: torch.Tensor,
    eps: float = 1e-6,
    chunk_size: int = 32,
    n_iter: int = 16,
    support_rel: float = 0.25,
    fiedler_tau: float = 0.05,
) -> torch.Tensor:
    """Live induced-n8 2-component impurity ρ = exp(-(λ1-λ2)/τ).

    Two n8-components on the thresholded support → gap 0 → ρ=1.
    One connected blob → gap>0 → ρ smaller. Empty/singleton → 0.
    Z is detached (frozen DINO). a is live so the loss can split a merge.
    """
    _induced_n8_check_shapes(att, features, "spectral_graph_n8_induced_impurity")
    tau = float(fiedler_tau)
    if tau <= 0.0:
        raise ValueError(f"fiedler_tau must be > 0, got {tau}")
    amp_ctx = (
        torch.cuda.amp.autocast(enabled=False) if att.is_cuda else nullcontext()
    )
    with amp_ctx:
        z = torch.nn.functional.normalize(features.float(), dim=-1).detach()
        a = att.float()
        chunks = []
        step = max(int(chunk_size), 1)
        for i in range(0, a.shape[0], step):
            (lam1, lam2), a_s = _top2_induced_n8(
                z[i : i + step],
                a[i : i + step],
                n_iter,
                eps,
                support_rel,
            )
            gap = (lam1 - lam2.clamp_min(0.0)).clamp_min(0.0)
            n_sup = (a_s > float(eps)).sum(dim=-1)
            rho = torch.exp(-gap / tau)
            rho = torch.where(n_sup <= 1, torch.zeros_like(rho), rho)
            chunks.append(rho)
    return torch.cat(chunks, dim=0).to(dtype=att.dtype)


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


def slot_confidence_entropy(
    c: torch.Tensor,
    eps: float = 1e-6,
    normalize: bool = True,
) -> torch.Tensor:
    """Entropy of the slot-wise confidence distribution.

    q_s ∝ c_s + eps. c (..., S) -> (...,). One-hot c -> 0.
    normalize=True (v40/v41): H(q)/log K in [0, 1]; uniform / all-zero -> 1.
    normalize=False (v42): raw H(q) in nats; uniform -> log K. Live.
    """
    n_slots = c.shape[-1]
    q = (c + eps) / (c.sum(dim=-1, keepdim=True) + float(n_slots) * eps)
    ent = -(q * q.clamp_min(eps).log()).sum(dim=-1)
    if normalize:
        return ent / math.log(max(n_slots, 2))
    return ent


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


class SlotUsageHead(nn.Module):
    """Per-slot usage from the last-iter attention map.

    Shared MLP over the flattened patch map, then one cross-slot Transformer
    block, then a linear. Input is stop-grad by default so the head cannot
    rewrite A to open its own gate.

    normalize='sigmoid' (v41): independent z_s in (0, 1). Decoder renorm only
    sees ratios, so the absolute scale of z is unidentified and can collapse.
    normalize='softmax' (v42): z is a distribution over slots. Same object the
    decoder mixture already uses; all-zero / scale drift is impossible.
    normalize='sparsemax' (v43): simplex with exact zeros so unused slots can
    be dropped and revived via logit-space L_gate.
    """

    def __init__(
        self,
        n_patches: int,
        mlp_hidden: int = 256,
        d_model: int = 64,
        n_blocks: int = 1,
        n_heads: int = 4,
        stopgrad_attn: bool = True,
        dropout: float = 0.0,
        normalize: str = "sigmoid",
    ):
        super().__init__()
        if n_patches <= 0:
            raise ValueError(f"SlotUsageHead n_patches must be positive, got {n_patches}")
        if d_model <= 0:
            raise ValueError(f"SlotUsageHead d_model must be positive, got {d_model}")
        norm = str(normalize).lower()
        if norm not in ("sigmoid", "softmax", "sparsemax"):
            raise ValueError(
                f"SlotUsageHead normalize must be 'sigmoid', 'softmax' or "
                f"'sparsemax', got {normalize!r}"
            )
        self.n_patches = int(n_patches)
        self.stopgrad_attn = bool(stopgrad_attn)
        self.normalize = norm
        self.patch_mlp = MLP(
            self.n_patches,
            d_model,
            [int(mlp_hidden)],
            activation="gelu",
        )
        self.mixer = TransformerEncoder(
            dim=d_model,
            n_blocks=int(n_blocks),
            n_heads=int(n_heads),
            dropout=float(dropout),
        )
        self.out = nn.Linear(d_model, 1)
        nn.init.zeros_(self.out.bias)

    def logits_and_usage(self, att: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """att (B, S, N) -> (logits, z). z is sigmoid / softmax / sparsemax(ℓ)."""
        if att.ndim != 3:
            raise ValueError(f"SlotUsageHead expects att (B, S, N), got {tuple(att.shape)}")
        if att.shape[-1] != self.n_patches:
            raise ValueError(
                f"SlotUsageHead n_patches={self.n_patches} vs att {tuple(att.shape)}"
            )
        x = att.detach() if self.stopgrad_attn else att
        e = self.patch_mlp(x)
        h = self.mixer(e)
        logits = self.out(h).squeeze(-1)
        if self.normalize == "softmax":
            z = torch.softmax(logits, dim=-1)
        elif self.normalize == "sparsemax":
            z = sparsemax(logits, dim=-1)
        else:
            z = torch.sigmoid(logits)
        return logits, z

    def forward(self, att: torch.Tensor) -> torch.Tensor:
        """att (B, S, N) -> z (B, S). softmax/sparsemax: sum_s z_s = 1."""
        return self.logits_and_usage(att)[1]


class LatentProcessor(nn.Module):
    """Updates latent state based on inputs and state and predicts next state."""

    def __init__(
        self,
        corrector: nn.Module,
        predictor: Optional[nn.Module] = None,
        state_key: str = "slots",
        first_step_corrector_args: Optional[Dict[str, Any]] = None,
        usage_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.corrector = corrector # slot attention
        self.predictor = predictor # transformer encoder
        self.state_key = state_key
        if first_step_corrector_args is not None:
            self.first_step_corrector_args = first_step_corrector_args
        else:
            self.first_step_corrector_args = None
        self.usage_head = usage_head
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
        predictor_src_gate: bool = False,
        predictor_src_max_norm: bool = False,
        predictor_src_self: bool = False,
        predictor_pair_isolate: bool = False,
        predictor_input_mix: bool = False,
        predictor_mix_hold: bool = True,
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
        decoder_gate_hysteresis: float = 0.0,
        decoder_gate_prev: Optional[torch.Tensor] = None,
        state_identity_cos: bool = False,
        state_conf_kind: Optional[str] = None,
        n8_support_rel: float = 0.0,
        spectral_proj_dim: int = 0,
        eval_hard_thresh: float = 0.0,
        eval_perron_readout: bool = False,
        pi_gamma: float = 1.0,
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
        if predictor_input_mix and predictor_ungated:
            raise ValueError(
                "predictor_input_mix cannot be combined with predictor_ungated"
            )
        if not predictor_mix_hold and not predictor_input_mix:
            raise ValueError(
                "predictor_mix_hold=False requires predictor_input_mix"
            )
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
        gate_logits = None
        n8_lam1 = None
        n8_q1 = None
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
                if ck == "usage":
                    # v41: z = sigmoid(head(sg(A))). v42: softmax. v43: sparsemax.
                    # Stop-grad of A is inside the head. Logits stay live for L_gate.
                    if self.usage_head is None:
                        raise ValueError(
                            "conf_kind='usage' requires LatentProcessor.usage_head"
                        )
                    if hasattr(self.usage_head, "logits_and_usage"):
                        gate_logits, gate_conf = self.usage_head.logits_and_usage(att)
                    else:
                        gate_conf = self.usage_head(att)
                elif ck in ("purity", "purity_sharp"):
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
                elif ck in ("spectral", "spectral_gap", "spectral_ratio"):
                    # v37: content-aware purity. Exclusive ownership of two feature
                    # modes is no longer c~=1; λ2 rises and π drops. Uses raw A
                    # (a^2 in C_s is the within-slot moment) and bind tokens, not
                    # att_sharp: gamma would ask "won patches" instead of "this
                    # slot's mass". v37g (spectral_gap): same C_s, π = λ1-λ2 with
                    # no /mass, so a one-color leftover no longer scores like a
                    # full object. v37r (spectral_ratio): (λ1-λ2)/(λ1+λ2), the
                    # bounded map of λ1/λ2. Size-free; isotropic ghosts → 0.
                    gate_conf = spectral_slot_purity(
                        att,
                        bind_features,
                        divide_by_mass=(ck == "spectral"),
                        proj_dim=spectral_proj_dim,
                        ratio=(ck == "spectral_ratio"),
                    )
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
                    "spectral_graph_n8_lam1",
                ):
                    # v39: 8-neighbor R only. π = λ1 - max(λ2, 0) (same as v38).
                    # v39n (spectral_graph_n8_rel): that gap divided by λ1.
                    # v39s (spectral_graph_n8_l1imp): [λ1 - λ2⁺/(λ1+ε)]_+.
                    # v39lam1 (spectral_graph_n8_lam1): π = max(λ1, 0); no λ2.
                    # λ1, q1 are the Perron power pair (skip k=8 Ritz).
                    # Curriculum P stays global. Raw A, bind tokens, detached.
                    # n8_lam1 is kept so a split temporal mix (v39l1) can use
                    # scale while the decoder keeps the gap π. v39lam1 sets
                    # gate_conf = λ1 so decoder and temporal share that scale.
                    packed_eigs = spectral_graph_n8_eigs(
                        att,
                        bind_features,
                        divide_by_lambda1=(ck == "spectral_graph_n8_rel"),
                        l1_minus_imp=(ck == "spectral_graph_n8_l1imp"),
                        use_lambda1=(ck == "spectral_graph_n8_lam1"),
                        return_q1=bool(eval_perron_readout),
                    )
                    if eval_perron_readout:
                        gate_conf, n8_lam1, _, n8_q1 = packed_eigs
                    else:
                        gate_conf, n8_lam1, _ = packed_eigs
                elif ck == "spectral_graph_n8_ind":
                    # v39ind: induced n8 S on thresholded support, re-normalized
                    # there. π = λ1-λ2⁺; two components → 0, one blob → μ2.
                    gate_conf, n8_lam1, _ = spectral_graph_n8_induced_eigs(
                        att,
                        bind_features,
                        support_rel=n8_support_rel,
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
                gate_conf = _leaky_max_gate(
                    gate_conf, gate_conf_prev, gate_hysteresis
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
                # v37 leaves this off: π ≤ 1/K (two-mode gaps, weak ghosts) must
                # stay distinct instead of clipping to the same 0.
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
            # Optional temporal statistic, decoder stays on π (v39 eval-only λ1 mix):
            #   lambda1: n8 λ1 (visible scale; drops under cover occlusion)
            #   mass:    attention-mass fraction
            # Applied after the decoder mask is built so featrec/ghosts are unchanged.
            sk = str(state_conf_kind or "").lower()
            if sk in ("lambda1", "l1"):
                if n8_lam1 is None:
                    raise ValueError(
                        "state_conf_kind='lambda1' requires conf_kind "
                        "spectral_graph_n8 / spectral_graph_n8_rel / "
                        "spectral_graph_n8_l1imp / spectral_graph_n8_lam1 / "
                        "spectral_graph_n8_ind"
                    )
                state_mask = n8_lam1
            elif sk == "mass":
                state_mask = mass_frac
            elif sk not in ("", "pi", "none"):
                raise ValueError(
                    f"state_conf_kind must be '', 'pi', 'lambda1' or 'mass', got {sk!r}"
                )
            # Eval-only hard gate (purity_weight ignores gate_mode). Applied after π is
            # built and before Pred / mix, so decoder, temporal g, and src-gate agree.
            # Training keeps the float π. thresh<=0 is a no-op.
            ht = 0.0 if eval_hard_thresh is None else float(eval_hard_thresh)
            if ht > 0.0:
                aliased = state_mask is active_mask
                active_mask = _eval_hard_threshold(active_mask, ht)
                state_mask = active_mask if aliased else _eval_hard_threshold(state_mask, ht)
            # Eval-only π^γ. After hard thresh, before Pred / mix / decoder
            # (Perron and pair-isolate read these masks).
            pg = 1.0 if pi_gamma is None else float(pi_gamma)
            if pg != 1.0:
                aliased = state_mask is active_mask
                active_mask = _pi_gamma(active_mask, pg)
                state_mask = active_mask if aliased else _pi_gamma(state_mask, pg)
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

        pred_kwargs: Dict[str, Any] = {}
        if vel is not None:
            pred_kwargs["vel"] = vel
        # Source-gated self-attention (v39lam1g): keys/values of low-g slots
        # are down-weighted inside Pred, before the temporal mix.
        #   B_{i,j}^g = A_{i,j} exp(b_{i,j}) / sum_k A_{i,k} exp(b_{i,k})
        # Uses the current-frame mix statistic (state_mask, else decoder g),
        # detached. Max-norm is applied later on the mix, not here, unless
        # predictor_src_max_norm (g = π̄ so Pred SA matches the mix). With
        # predictor_src_self, A_ii = 1 and A_ij = g_j (i≠j).
        if predictor_src_gate:
            src_g = state_mask if state_mask is not None else active_mask
            if src_g is not None:
                if predictor_src_max_norm:
                    src_g = _max_norm(src_g, True)
                src_w = src_g.detach()
                if predictor_src_self:
                    src_w = _src_gate_self_weights(src_w)
                pred_kwargs["key_weights"] = src_w.to(dtype=updated_state.dtype)
        # Same-group Pred SA (eval isolate): live-live and ghost-ghost only.
        # Overwrites per-key src-gate when both are on (pairwise is the stricter
        # mask; src-gate softmax would re-open ghost→live).
        if predictor_pair_isolate:
            src_g = state_mask if state_mask is not None else active_mask
            if src_g is not None:
                pred_kwargs["key_weights"] = _pair_isolate_weights(src_g).to(
                    dtype=updated_state.dtype
                )
        # v39lam1gu_umix defers Pred until π̄ and u_t are known.
        if self.predictor and not predictor_input_mix:
            predicted_state = self.predictor(updated_state, **pred_kwargs)
        elif not predictor_input_mix:
            predicted_state = updated_state
        else:
            predicted_state = None
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
        # Optional decoder-only leaky max (v39d). Applied after the two masks are
        # built, so the temporal mix keeps instantaneous π (a vanished object
        # freezes its prior). Shared gate_hysteresis (v39h) already smoothed
        # gate_conf before both paths; do not stack the two knobs.
        #   decoder: π̃_t = max(π_t, γ π̃_{t-1})
        #   temporal: π_t  (then max-norm / EMA below)
        dec_g = (
            0.0 if decoder_gate_hysteresis is None else float(decoder_gate_hysteresis)
        )
        if dec_g > 0.0 and float(gate_hysteresis or 0.0) > 0.0:
            raise ValueError(
                "decoder_gate_hysteresis and gate_hysteresis cannot both be > 0"
            )
        if (
            active_mask is not None
            and torch.is_floating_point(active_mask)
            and dec_g > 0.0
            and decoder_gate_prev is not None
        ):
            if state_mask is active_mask:
                state_mask = active_mask.clone()
            active_mask = _leaky_max_gate(active_mask, decoder_gate_prev, dec_g)

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
            a = _max_norm(state_mask, state_max_norm).unsqueeze(-1).to(
                updated_state.dtype
            )
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
            if predictor_input_mix:
                # Decoder still reads u^SA (`state` below). Pred input is the
                # occupancy mix so a vanished slot does not leak u^SA:
                #   u_t = π̄ u^SA + (1-π̄) û_t
                # umix (mix_hold True):  û_{t+1} = π̄ Pred(u_t) + (1-π̄) u_t
                # umix_pred (False):     û_{t+1} = Pred(u_t)
                mixed = a * updated_state + (1.0 - a) * state
                if self.predictor:
                    predicted_state = self.predictor(mixed, **pred_kwargs)
                else:
                    predicted_state = mixed
                predicted_pregate = predicted_state
                if predictor_mix_hold:
                    predicted_state = a * predicted_state + (1.0 - a) * mixed
            else:
                predicted_state = a * predicted_state + (1.0 - a) * state
        elif predictor_input_mix:
            if self.predictor:
                predicted_state = self.predictor(updated_state, **pred_kwargs)
            else:
                predicted_state = updated_state
            predicted_pregate = predicted_state

        if active_mask is None:
            # keep a consistent output tree; all slots are "active" when gating is off
            active_mask = torch.ones(
                updated_state.shape[:2], dtype=torch.bool, device=updated_state.device
            )
        if state_mask is None:
            state_mask = active_mask

        # Perron readout: decoder sees π_s q̃_{1,s,i}; mix / Pred keep scalar π.
        # Spatial gate is a separate tensor so losses / logging still read (B, S).
        decoder_gate = None
        if eval_perron_readout:
            if n8_q1 is None:
                raise ValueError(
                    "eval_perron_readout requires conf_kind in the n8 family "
                    "(needs G_s = diag(a) S diag(a) and its Perron vector)"
                )
            if active_mask is None or not torch.is_floating_point(active_mask):
                raise ValueError(
                    "eval_perron_readout needs a float decoder π; got "
                    f"{None if active_mask is None else active_mask.dtype}"
                )
            decoder_gate = _perron_spatial_gate(active_mask, n8_q1)

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
        if decoder_gate is not None:
            out["decoder_gate"] = decoder_gate
        if gate_conf is not None:
            # (B, S) assignment / usage score. Ownership and spectral kinds are
            # detached; conf_kind='usage' is live z so featrec and L_ent reach the
            # head. When gate_hysteresis > 0 this is the smoothed π̃ that
            # ScanOverTime carries into the next frame's `gate_conf_prev`.
            out["gate_conf"] = gate_conf
        if gate_logits is not None:
            out["gate_logits"] = gate_logits
        if mass_median_ema is not None:
            out["mass_median_ema"] = mass_median_ema
        if gate_p_eff is not None and torch.is_tensor(gate_p_eff):
            out["gate_p_eff"] = gate_p_eff.squeeze(-1)  # (B,)
        if gate_delta is not None and torch.is_tensor(gate_delta):
            out["gate_delta"] = gate_delta.squeeze(-1)  # (B,)
        if state_mask is not None and torch.is_floating_point(state_mask):
            # next-frame prev for temporal EMA / hold (already smoothed if enabled)
            out["state_gate_carry"] = state_mask.detach()
        if active_mask is not None and torch.is_floating_point(active_mask):
            # next-frame prev for decoder-only hysteresis (smoothed if enabled)
            out["decoder_gate_carry"] = active_mask.detach()
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

    def forward(self, *args, **kwargs):
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

        outputs = self.module(*flattened_args, **kwargs)

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
        # Anchor frame chosen per sample by the last "evidence*"/"random" cycle (diagnostics
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
        predictor_src_gate: bool = False,
        predictor_src_max_norm: bool = False,
        predictor_src_self: bool = False,
        predictor_pair_isolate: bool = False,
        predictor_input_mix: bool = False,
        predictor_mix_hold: bool = True,
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
        decoder_gate_hysteresis: float = 0.0,
        state_identity_cos: bool = False,
        state_conf_kind: Optional[str] = None,
        n8_support_rel: float = 0.0,
        spectral_proj_dim: int = 0,
        eval_hard_thresh: float = 0.0,
        eval_perron_readout: bool = False,
        pi_gamma: float = 1.0,
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
            predictor_src_gate=predictor_src_gate,
            predictor_src_max_norm=predictor_src_max_norm,
            predictor_src_self=predictor_src_self,
            predictor_pair_isolate=predictor_pair_isolate,
            predictor_input_mix=predictor_input_mix,
            predictor_mix_hold=predictor_mix_hold,
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
            decoder_gate_hysteresis=decoder_gate_hysteresis,
            state_identity_cos=state_identity_cos,
            state_conf_kind=state_conf_kind,
            n8_support_rel=n8_support_rel,
            spectral_proj_dim=spectral_proj_dim,
            eval_hard_thresh=eval_hard_thresh,
            eval_perron_readout=eval_perron_readout,
            pi_gamma=pi_gamma,
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
        # Decoder-only hysteresis carry (v39d). None on frame 0.
        decoder_gate_prev = None
        outputs = []
        for t in range(seq_len):
            kwargs = dict(gate_kwargs)
            kwargs["median_ema_prev"] = median_ema
            kwargs["prev_state"] = prev_state
            kwargs["gate_conf_prev"] = gate_conf_prev
            kwargs["state_gate_prev"] = state_gate_prev
            kwargs["decoder_gate_prev"] = decoder_gate_prev
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
            decoder_gate_prev = output.get("decoder_gate_carry", decoder_gate_prev)

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
            #   "evidence_sum": EABI with E_t = sum_s g_{t,s} (gate occupancy only).
            #       No extra ownership-purity c or mass m: for v39lam1gu, g is already
            #       λ1, so c is nearly a monotone of the same peakiness.
            #   "random": EABI with a uniformly random anchor -- control run isolating
            #       the value of evidence-based anchor selection.
            mode = cycle.strip().lower() if isinstance(cycle, str) else "last"
            if mode in ("evidence", "evidence_mass", "evidence_sum", "random"):
                if mode == "random":
                    anchors = torch.randint(
                        seq_len, (inputs.shape[0],), device=inputs.device
                    )
                else:
                    if mode == "evidence_mass":
                        stat = "mass"
                    elif mode == "evidence_sum":
                        stat = "sum"
                    else:
                        stat = "count"
                    anchors = _evidence_anchors(
                        outputs,
                        gate_kwargs.get("mass_gamma") or 1.0,
                        stat=stat,
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
            decoder_gate_prev = outputs[-1].get("decoder_gate_carry")
            for t in range(seq_len - 1):
                back_t = seq_len - t - 2
                kwargs = dict(gate_kwargs)
                kwargs["median_ema_prev"] = median_ema
                kwargs["prev_state"] = prev_state
                kwargs["gate_conf_prev"] = gate_conf_prev
                kwargs["state_gate_prev"] = state_gate_prev
                kwargs["decoder_gate_prev"] = decoder_gate_prev
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
                decoder_gate_prev = out.get("decoder_gate_carry", decoder_gate_prev)
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
        anchor_dec = None
        gate_conf_prev = None
        state_gate_prev = None
        decoder_gate_prev = None
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
        if outputs and outputs[0].get("decoder_gate_carry") is not None:
            dec_states = torch.stack([o["decoder_gate_carry"] for o in outputs], dim=1)
            didx = anchors.view(b, 1, 1).expand(-1, 1, dec_states.shape[-1])
            anchor_dec = dec_states.gather(1, didx).squeeze(1)
            decoder_gate_prev = anchor_dec

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
            if anchor_dec is not None:
                dstarts = (anchors == t + 1).view(b, *([1] * (anchor_dec.ndim - 1)))
                decoder_gate_prev = torch.where(dstarts, anchor_dec, decoder_gate_prev)
            kwargs = dict(gate_kwargs)
            kwargs["median_ema_prev"] = median_ema
            kwargs["prev_state"] = prev_state
            kwargs["gate_conf_prev"] = gate_conf_prev
            kwargs["state_gate_prev"] = state_gate_prev
            kwargs["decoder_gate_prev"] = decoder_gate_prev
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
            decoder_gate_prev = out.get("decoder_gate_carry", decoder_gate_prev)

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

    stat="sum": E_t = sum_s g_{t,s}. Gate occupancy only; no extra c or m. For
    v39lam1gu g is already λ1, so this is the EABI statistic that does not
    introduce a second definition of "intactness".

    Both signals come from the forward sweep's outputs, so anchor selection costs no
    extra model evaluation. E_t is mean-smoothed over a `window`-frame neighborhood
    (replicate-padded) before the argmax so a single-frame noise spike cannot become
    the anchor. With gating disabled active_mask is all-ones and gamma=1 purity/mass
    are frame-independent constants only in degenerate cases; anchor quality then just
    falls back to whatever the statistic sees.
    """
    gates = torch.stack([o["active_mask"].float() for o in outputs], dim=1)  # (B, T, S)
    stat_k = (stat or "count").strip().lower()
    if stat_k in ("sum", "gate"):
        evidence = gates.sum(dim=-1)  # (B, T)
    else:
        att = torch.stack([o["state_attn_mask"].float() for o in outputs], dim=1)  # (B,T,S,F)
        gamma = float(mass_gamma or 1.0)
        if gamma != 1.0:
            att = att.pow(gamma)
            att = att / att.sum(dim=2, keepdim=True).clamp_min(1e-8)
        if stat_k == "mass":
            per_slot = att.sum(dim=-1) / att.shape[-1]  # coverage m, (B, T, S)
        elif stat_k == "count":
            per_slot = (att * att).sum(dim=-1) / att.sum(dim=-1).clamp_min(1e-8)  # purity c
        else:
            raise ValueError(f"unknown evidence stat {stat!r}")
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
