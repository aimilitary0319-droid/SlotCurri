from typing import Optional

import torch
from torch import nn

from slotcurri.utils import make_build_fn


@make_build_fn(__name__, "initializer")
def build(config, name: str):
    pass  # No special module building needed


class RandomInit(nn.Module):
    """Sampled random initialization for all slots."""

    def __init__(self, n_slots: int, dim: int, initial_std: Optional[float] = None):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        self.mean = nn.Parameter(torch.zeros(1, 1, dim))
        if initial_std is None:
            initial_std = dim**-0.5
        self.log_std = nn.Parameter(torch.log(torch.ones(1, 1, dim) * initial_std))

    def forward(self, batch_size: int):
        noise = torch.randn(batch_size, self.n_slots, self.dim, device=self.mean.device)
        return self.mean + noise * self.log_std.exp()


class FixedLearnedInit(nn.Module):
    """Learned initialization with a fixed number of slots.

    Default draws each slot independently from N(0, 1/dim). `shared_init`
    draws one shared vector and one noise vector per slot, both N(0, 1),
    and stores slot = shared + noise. `identical_init` copies one N(0, 1)
    vector into every slot, so the rows start equal. Either draw happens
    once. Later steps train these vectors; they are not resampled.
    """

    def __init__(
        self,
        n_slots: int,
        dim: int,
        initial_std: Optional[float] = None,
        shared_init: bool = False,
        identical_init: bool = False,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        if initial_std is None:
            initial_std = dim**-0.5
        self.initial_std = initial_std
        self.shared_init = bool(shared_init)
        self.identical_init = bool(identical_init)
        if self.shared_init and self.identical_init:
            raise ValueError("shared_init and identical_init cannot both be set")
        if self.identical_init:
            shared = torch.randn(1, 1, dim)
            slots = shared.expand(1, n_slots, dim).clone()
        elif self.shared_init:
            shared = torch.randn(1, 1, dim)
            noise = torch.randn(1, n_slots, dim)
            slots = shared + noise
        else:
            slots = torch.randn(1, n_slots, dim) * initial_std
        self.slots = nn.Parameter(slots)

    def forward(self, batch_size: int):
        return self.slots.expand(batch_size, -1, -1)


class FixedOrthogonalInit(nn.Module):
    """Fixed (non-learnable) mutually-orthonormal slot initialization.

    Prepares ``n_slots`` orthonormal vectors in ``dim``-dimensional space via QR
    decomposition and keeps them frozen (registered as a buffer, no gradient). The
    per-slot norm is matched to ``sqrt(dim) * initial_std`` so the scale is comparable
    to ``FixedLearnedInit`` (~1 when ``initial_std = dim**-0.5``).

    Requires ``n_slots <= dim`` for exact orthogonality.
    """

    def __init__(
        self,
        n_slots: int,
        dim: int,
        initial_std: Optional[float] = None,
        seed: int = 0,
    ):
        super().__init__()
        assert n_slots <= dim, f"need n_slots({n_slots}) <= dim({dim}) for orthogonality"
        self.n_slots = n_slots
        self.dim = dim
        if initial_std is None:
            initial_std = dim**-0.5
        self.initial_std = initial_std

        g = torch.Generator().manual_seed(seed)
        A = torch.randn(dim, n_slots, generator=g)
        Q, _ = torch.linalg.qr(A)  # (dim, n_slots) with orthonormal columns
        ortho = Q.t()  # (n_slots, dim) orthonormal rows
        scale = (dim**0.5) * initial_std
        slots = (ortho * scale).unsqueeze(0)  # (1, n_slots, dim)
        # frozen: buffer (not a Parameter), so it is not optimized
        self.register_buffer("slots", slots)

    def forward(self, batch_size: int):
        return self.slots.expand(batch_size, -1, -1)
