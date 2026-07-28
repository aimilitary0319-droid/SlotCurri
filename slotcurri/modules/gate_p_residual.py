"""Learnable residual adjustment for attention-mass gate threshold p."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


class GatePResidualNet(nn.Module):
    """Predict delta in (-1, 1) from a detached attention-mass distribution.

    Features (per batch element):
      sort(m) descending, H(m)/log S, top2 sum, median(m)
    """

    def __init__(self, n_slots: int, hidden: int = 32):
        super().__init__()
        self.n_slots = int(n_slots)
        in_dim = self.n_slots + 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def build_features(mass_frac: torch.Tensor) -> torch.Tensor:
        """mass_frac: (B, S) -> features (B, S+3)."""
        # sort descending for permutation-invariant layout
        m_sorted, _ = mass_frac.sort(dim=-1, descending=True)
        s = mass_frac.shape[-1]
        # entropy of mass distribution
        q = mass_frac.clamp_min(1e-8)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ent = -(q * q.log()).sum(dim=-1, keepdim=True) / math.log(max(s, 2))
        top2 = m_sorted[..., : min(2, s)].sum(dim=-1, keepdim=True)
        med = mass_frac.median(dim=-1, keepdim=True).values
        return torch.cat([m_sorted, ent, top2, med], dim=-1)

    def forward(self, mass_frac: torch.Tensor) -> torch.Tensor:
        """Return delta in (-1, 1) with shape (B, 1)."""
        z = self.build_features(mass_frac)
        return torch.tanh(self.net(z))
