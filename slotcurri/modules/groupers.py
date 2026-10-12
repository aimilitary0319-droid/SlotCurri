import math
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
from torch import nn

from slotcurri.modules import networks
from slotcurri.modules.usage_redistribute import entmax15, sparsemax
from slotcurri.utils import make_build_fn


@make_build_fn(__name__, "grouper")
def build(config, name: str):
    pass  # No special module building needed


def margin_sigmoid_over_slots(dots: torch.Tensor) -> torch.Tensor:
    """Per-slot sigmoid against that slot's strongest rival. dots is (B, S, F).

    α_s = σ(z_s - max_{k≠s} z_k). Scores are not renormalized over slots, so the
    slot axis does not sum to 1. A tied pair scores 0.5 each. Requires S >= 2.
    """
    if dots.ndim != 3:
        raise ValueError(
            f"margin sigmoid expects dots (B, S, F), got {tuple(dots.shape)}"
        )
    if dots.shape[1] < 2:
        raise ValueError(
            f"margin sigmoid needs at least 2 slots, got {dots.shape[1]}"
        )
    top2 = torch.topk(dots, k=2, dim=1).values
    zmax = top2[:, :1]
    zsecond = top2[:, 1:2]
    rival = torch.where(dots == zmax, zsecond, zmax)
    return torch.sigmoid(dots - rival)


def dot_ktemp_logits(
    queries: torch.Tensor,
    keys: torch.Tensor,
    odds: float = 1.0,
    sqrt_width: bool = False,
) -> torch.Tensor:
    """ln(odds*(K-1)) times q·k over D or √D. queries (B, S, D), keys (B, F, D).

    Softmax over slots. odds=1 is ln(K-1). odds=99 puts a unit coordinate gap
    at leader mass 0.99 when dividing by D. sqrt_width divides by √D instead,
    so the logit is ln(odds*(K-1)) · q·k / √D. Dividing by D removes the
    coherent √D factor. Either way, doubling both vectors multiplies the logit by 4.
    """
    n_slots = queries.shape[-2]
    width = queries.shape[-1]
    odds = float(odds)
    if n_slots < 2:
        raise ValueError(f"dot temperature needs at least 2 slots, got {n_slots}")
    if width < 1:
        raise ValueError(f"dot temperature needs a positive width, got {width}")
    if odds <= 0:
        raise ValueError(f"dot temperature odds must be positive, got {odds}")
    denom = math.sqrt(float(width)) if sqrt_width else float(width)
    coeff = math.log(odds * (n_slots - 1)) / denom
    return torch.einsum("bsd, bfd -> bsf", queries, keys) * coeff


def attention_over_slots(dots: torch.Tensor, attn_norm: str) -> torch.Tensor:
    """Per-patch competition over slots. dots is (B, S, F).

    softmax is the Slot Attention default. sparsemax_ste uses sparsemax in the
    forward and the softmax Jacobian backward. entmax15 uses 1.5-entmax in the
    forward and its own Jacobian: a slot with weight 0 gets no gradient there.
    margin is σ(z_s - max_{k≠s} z_k) with no slot-axis renorm.
    """
    if attn_norm == "softmax":
        return torch.softmax(dots, dim=1)
    if attn_norm == "entmax15":
        return entmax15(dots, dim=1)
    if attn_norm == "sparsemax_ste":
        mapped = sparsemax(dots, dim=1)
        soft = torch.softmax(dots, dim=1)
        return mapped.detach() + soft - soft.detach()
    if attn_norm == "margin":
        return margin_sigmoid_over_slots(dots)
    raise ValueError(
        "SlotAttention attn_norm must be 'softmax', 'sparsemax_ste', "
        f"'entmax15', or 'margin', got {attn_norm!r}"
    )


def margin_update_weights(
    scores: torch.Tensor,
    eps: float,
    slot_norm: bool = False,
    power: float = 1.0,
) -> torch.Tensor:
    """Per-slot patch mean of margin scores. scores is (B, S, F).

    The mean always runs in the caller's dtype (fp32 under autocast-off) so a
    flushed row stays a finite weighted average. slot_norm divides by the slot
    sum first: a tie (~0.5) becomes 1/S and a unique win stays 1, so the patch
    mean follows the won patch. power raises each score before that mean, so a
    tie of 0.5 and a win of 1 are weighted 0.5**power : 1. The ratio does not
    depend on the slot count. π still reads the unnormalized scores.
    """
    if slot_norm and float(power) != 1.0:
        raise ValueError("margin slot_norm and update power are mutually exclusive")
    if slot_norm:
        scores = scores / scores.sum(dim=1, keepdim=True).clamp_min(float(eps))
    if float(power) != 1.0:
        scores = scores.clamp_min(0).pow(float(power))
    weights = scores + float(eps)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(float(eps))


class SlotAttention(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        slot_dim: int,
        kvq_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        n_iters: int = 3,
        eps: float = 1e-8,
        use_gru: bool = True,
        use_mlp: bool = True,
        attn_norm: str = "softmax",
        pi_attn_norm: str = "",
        margin_slot_norm: bool = False,
        margin_gate_slot_norm: bool = False,
        margin_update_power: float = 1.0,
        dot_ktemp: bool = False,
        dot_ktemp_odds: float = 1.0,
        dot_ktemp_sqrt: bool = False,
    ):
        super().__init__()
        assert n_iters >= 1
        attn_norm = str(attn_norm).lower()
        if attn_norm not in ("softmax", "sparsemax_ste", "entmax15", "margin"):
            raise ValueError(
                "SlotAttention attn_norm must be 'softmax', 'sparsemax_ste', "
                f"'entmax15', or 'margin', got {attn_norm!r}"
            )
        pi_attn_norm = str(pi_attn_norm).lower()
        if pi_attn_norm not in ("", "entmax15", "margin"):
            raise ValueError(
                "SlotAttention pi_attn_norm must be '', 'entmax15', or 'margin', "
                f"got {pi_attn_norm!r}"
            )
        if pi_attn_norm and attn_norm != "softmax":
            raise ValueError(
                "pi_attn_norm keeps the slot update on softmax; "
                f"got attn_norm={attn_norm!r}"
            )
        if margin_slot_norm and attn_norm != "margin":
            raise ValueError(
                "margin_slot_norm applies only when attn_norm='margin', "
                f"got attn_norm={attn_norm!r}"
            )
        if margin_gate_slot_norm and attn_norm != "margin":
            raise ValueError(
                "margin_gate_slot_norm applies only when attn_norm='margin', "
                f"got attn_norm={attn_norm!r}"
            )
        if margin_gate_slot_norm and pi_attn_norm:
            raise ValueError(
                "margin_gate_slot_norm and pi_attn_norm both set the π attention"
            )
        margin_update_power = float(margin_update_power)
        if margin_update_power <= 0:
            raise ValueError(
                "margin_update_power must be positive, "
                f"got {margin_update_power}"
            )
        if margin_update_power != 1.0 and attn_norm != "margin":
            raise ValueError(
                "margin_update_power applies only when attn_norm='margin', "
                f"got attn_norm={attn_norm!r}"
            )
        if margin_slot_norm and margin_update_power != 1.0:
            raise ValueError(
                "margin_slot_norm and margin_update_power are mutually exclusive"
            )
        self.dot_ktemp = bool(dot_ktemp)
        self.dot_ktemp_odds = float(dot_ktemp_odds)
        self.dot_ktemp_sqrt = bool(dot_ktemp_sqrt)
        if self.dot_ktemp_sqrt and not self.dot_ktemp:
            raise ValueError("dot_ktemp_sqrt applies only when dot_ktemp is true")
        if self.dot_ktemp_odds <= 0:
            raise ValueError(
                f"dot_ktemp_odds must be positive, got {self.dot_ktemp_odds}"
            )
        if self.dot_ktemp_odds != 1.0 and not self.dot_ktemp:
            raise ValueError("dot_ktemp_odds applies only when dot_ktemp is true")
        if self.dot_ktemp and attn_norm != "softmax":
            raise ValueError(
                "dot_ktemp keeps a softmax over slots, "
                f"got attn_norm={attn_norm!r}"
            )
        if self.dot_ktemp and pi_attn_norm:
            raise ValueError(
                "dot_ktemp uses one softmax for the update and π, "
                f"got pi_attn_norm={pi_attn_norm!r}"
            )
        self.attn_norm = attn_norm
        self.pi_attn_norm = pi_attn_norm
        self.margin_slot_norm = bool(margin_slot_norm)
        self.margin_gate_slot_norm = bool(margin_gate_slot_norm)
        self.margin_update_power = margin_update_power

        if kvq_dim is None:
            kvq_dim = slot_dim
        self.to_k = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_v = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_q = nn.Linear(slot_dim, kvq_dim, bias=False)

        if use_gru:
            self.gru = nn.GRUCell(input_size=kvq_dim, hidden_size=slot_dim)
        else:
            assert kvq_dim == slot_dim
            self.gru = None

        if hidden_dim is None:
            hidden_dim = 4 * slot_dim

        if use_mlp:
            self.mlp = networks.MLP(
                slot_dim, slot_dim, [hidden_dim], initial_layer_norm=True, residual=True
            )
        else:
            self.mlp = None

        self.norm_features = nn.LayerNorm(inp_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

        self.n_iters = n_iters
        self.eps = eps
        self.scale = kvq_dim**-0.5

    def step(
        self, slots: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Perform one iteration of slot attention."""
        slots = self.norm_slots(slots) # 1 7 64
        queries = self.to_q(slots)

        # fp16 flushes a losing slot's scores to 0, and eps=1e-8 flushes too,
        # so the patch mean is 0/0. Competition and the mean stay in fp32.
        amp_ctx = (
            torch.cuda.amp.autocast(enabled=False) if queries.is_cuda else nullcontext()
        )
        with amp_ctx:
            q = queries.float()
            k = keys.float()
            if self.dot_ktemp:
                dots = dot_ktemp_logits(
                    q, k, self.dot_ktemp_odds, sqrt_width=self.dot_ktemp_sqrt
                )
            else:
                dots = torch.einsum("bsd, bfd -> bsf", q, k) * self.scale
            pre_norm_attn = attention_over_slots(dots, self.attn_norm)  # b slot patch
            attn = margin_update_weights(
                pre_norm_attn,
                self.eps,
                slot_norm=self.attn_norm == "margin" and self.margin_slot_norm,
                power=self.margin_update_power if self.attn_norm == "margin" else 1.0,
            )
            updates = torch.einsum("bsf, bfd -> bsd", attn, values.float())
            # π reads this map. The slot update above stays on pre_norm_attn
            # unless margin_slot_norm / margin_update_power reweights it.
            # Detach: π is already a no-grad gate, so this Jacobian must not train dots.
            # margin_gate_slot_norm feeds π the same slot-sum used by the update:
            # a tie of 0.5 becomes 1/S, a unique win stays 1. The patch mean is unchanged.
            pi_attn = None
            if self.pi_attn_norm:
                pi_attn = attention_over_slots(dots, self.pi_attn_norm).detach()
            elif self.margin_gate_slot_norm:
                pi_attn = (
                    pre_norm_attn
                    / pre_norm_attn.sum(dim=1, keepdim=True).clamp_min(float(self.eps))
                ).detach()
        updates = updates.to(dtype=slots.dtype)
        pre_norm_attn = pre_norm_attn.to(dtype=slots.dtype)
        if pi_attn is not None:
            pi_attn = pi_attn.to(dtype=slots.dtype)

        if self.gru:
            updated_slots = self.gru(updates.flatten(0, 1), slots.flatten(0, 1))
            slots = updated_slots.unflatten(0, slots.shape[:2])
        else:
            slots = slots + updates

        if self.mlp is not None:
            slots = self.mlp(slots)

        return slots, pre_norm_attn, pi_attn

    def forward(
        self,
        slots: torch.Tensor,
        features: torch.Tensor,
        n_iters: Optional[int] = None,
        key_features: Optional[torch.Tensor] = None,
    ):
        # Value always comes from `features` (original / recon-aligned tokens).
        # Key may come from a separately projected bind tensor (v36 Key-only leveling).
        values_in = self.norm_features(features)
        keys_in = values_in if key_features is None else self.norm_features(key_features)
        keys = self.to_k(keys_in)
        values = self.to_v(values_in)

        if n_iters is None:
            n_iters = self.n_iters

        pi_masks = None
        for _ in range(n_iters):
            slots, pre_norm_attn, pi_masks = self.step(slots, keys, values)

        out = {"slots": slots, "masks": pre_norm_attn}
        if pi_masks is not None:
            out["pi_masks"] = pi_masks
        return out
