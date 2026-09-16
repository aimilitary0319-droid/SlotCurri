from typing import Dict, List, Optional, Tuple, Union

import einops
import timm.layers.pos_embed
import torch
from torch import nn

from slotcurri.modules import networks, utils
from slotcurri.utils import config_as_kwargs, make_build_fn


@make_build_fn(__name__, "decoder")
def build(config, name: str):
    if name == "SpatialBroadcastDecoder":
        output_transform = None
        if config.get("output_transform"):
            output_transform = utils.build_module(config.output_transform)

        return SpatialBroadcastDecoder(
            backbone=utils.build_module(config.backbone, default_group="networks"),
            output_transform=output_transform,
            **config_as_kwargs(config, ("backbone", "output_transform")),
        )
    elif name == "SlotMixerDecoder":
        output_transform = None
        if config.get("output_transform"):
            output_transform = utils.build_module(config.output_transform)

        return SlotMixerDecoder(
            allocator=utils.build_module(config.allocator, default_group="networks"),
            renderer=utils.build_module(config.renderer, default_group="networks"),
            output_transform=output_transform,
            pos_embed_mode=config.get("pos_embed_mode", "add"),
            **config_as_kwargs(
                config, ("allocator", "renderer", "output_transform", "pos_embed_mode")
            ),
        )
    else:
        return None


class MLPDecoder(nn.Module):
    """Decoder that reconstructs independently for every position and slot."""

    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        hidden_dims: List[int],
        n_patches: int,
        activation: str = "relu",
        eval_output_size: Optional[Tuple[int]] = None,
    ):
        super().__init__()
        self.outp_dim = outp_dim
        self.n_patches = n_patches
        self.eval_output_size = list(eval_output_size) if eval_output_size else None

        self.mlp = networks.MLP(inp_dim, outp_dim + 1, hidden_dims, activation=activation)
        self.pos_emb = nn.Parameter(torch.randn(1, 1, n_patches, inp_dim) * inp_dim**-0.5)

    def forward(
        self,
        slots: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        return_slot_recons: bool = False,
        return_ungated: bool = False,
    ) -> Dict[str, torch.Tensor]:
        bs, n_slots, dims = slots.shape

        if not self.training and self.eval_output_size is not None:
            pos_emb = timm.layers.pos_embed.resample_abs_pos_embed(
                self.pos_emb.squeeze(1),
                new_size=self.eval_output_size,
                num_prefix_tokens=0,
            ).unsqueeze(1)
        else:
            pos_emb = self.pos_emb

        slots = slots.view(bs, n_slots, 1, dims).expand(bs, n_slots, pos_emb.shape[2], dims)
        slots = slots + pos_emb

        recons, alpha = self.mlp(slots).split((self.outp_dim, 1), dim=-1)
        # Softmax-over-slots mix with no gate. Same α / recon_s as the gated
        # path; only the mix weights change. Used by loss_featrec_ungated.
        masks_ungated = torch.softmax(alpha, dim=1) if return_ungated else None

        if active_mask is None:
            masks = torch.softmax(alpha, dim=1)
        elif active_mask.dtype == torch.bool:
            # hard gating: exclude non-active slots by driving their alpha to a large
            # negative value so their softmax mask becomes ~0 (no contribution).
            m = active_mask
            if m.dim() == 2:
                m = m[:, :, None, None]  # (bs, n_slots, 1, 1)
            alpha = alpha.masked_fill(~m, torch.finfo(alpha.dtype).min)
            masks = torch.softmax(alpha, dim=1)
        else:
            # soft gating: down-weight each slot's mask by its gate g, then
            # renormalize over slots so the per-patch masks still sum to 1.
            # g may be (B, S) broadcast to every patch, or (B, S, N) spatial
            # (eval Perron readout: π_s * q̃_{1,s,i}).
            g = active_mask
            if g.dim() == 2:
                g = g[:, :, None, None]  # (bs, n_slots, 1, 1)
            elif g.dim() == 3:
                g = g.unsqueeze(-1)  # (bs, n_slots, n_patches, 1)
            elif g.dim() != 4:
                raise ValueError(
                    f"decoder active_mask expected (B,S), (B,S,N) or "
                    f"(B,S,N,1), got {tuple(g.shape)}"
                )
            # g+eps so all-zero normalized purity recovers softmax(alpha)
            # (softmax(alpha + log(g+eps))), instead of a zero reconstruction.
            masks = torch.softmax(alpha, dim=1) * (g + 1e-8)
            masks = masks / masks.sum(dim=1, keepdim=True).clamp_min(1e-8)

        recon = torch.sum(recons * masks, dim=1)

        out = {"reconstruction": recon, "masks": masks.squeeze(-1)}
        if return_slot_recons:
            out["slot_recons"] = recons
        if return_ungated:
            out["reconstruction_ungated"] = torch.sum(recons * masks_ungated, dim=1)
            out["masks_ungated"] = masks_ungated.squeeze(-1)
        return out


class SpatialBroadcastDecoder(nn.Module):
    """Decoder that reconstructs a spatial map independently per slot."""

    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        backbone: nn.Module,
        initial_size: Union[int, Tuple[int, int]] = 8,
        backbone_dim: Optional[int] = None,
        pos_embed: Optional[nn.Module] = None,
        output_transform: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.outp_dim = outp_dim
        if isinstance(initial_size, int):
            self.initial_size = (initial_size, initial_size)
        else:
            self.initial_size = initial_size

        if pos_embed is None:
            self.pos_embed = utils.CoordinatePositionEmbed(inp_dim, initial_size)
        else:
            self.pos_embed = pos_embed

        self.backbone = backbone

        if output_transform is None:
            if backbone_dim is None:
                raise ValueError("Need to provide backbone dim if output_transform is unspecified")
            self.output_transform = nn.Conv2d(backbone_dim, outp_dim + 1, 1, 1)
        else:
            self.output_transform = output_transform

        self.init_parameters()

    def init_parameters(self):
        if isinstance(self.output_transform, nn.Conv2d):
            utils.init_parameters(self.output_transform)

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        bs, n_slots, _ = slots.shape

        slots = einops.repeat(
            slots, "b s d -> (b s) d h w", h=self.initial_size[0], w=self.initial_size[1]
        )

        slots = self.pos_embed(slots)
        features = self.backbone(slots)
        outputs = self.output_transform(features)

        outputs = einops.rearrange(outputs, "(b s) ... -> b s ...", b=bs, s=n_slots)
        recons, alpha = einops.unpack(outputs, [[self.outp_dim], [1]], "b s * h w")

        masks = torch.softmax(alpha, dim=1)
        recon = torch.sum(recons * masks, dim=1)

        return {"reconstruction": recon, "masks": masks.squeeze(2)}


class SlotMixerDecoder(nn.Module):
    """Slot mixer decoder reconstructing jointly over all slots, but independent per position.

    Introduced in Sajjadi et al., 2022: Object Scene Representation Transformer,
    http://arxiv.org/abs/2206.06922
    """

    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        embed_dim: int,
        n_patches: int,
        allocator: nn.Module,
        renderer: nn.Module,
        renderer_dim: Optional[int] = None,
        output_transform: Optional[nn.Module] = None,
        pos_embed_mode: Optional[str] = None,
        use_layer_norms: bool = False,
        norm_memory: bool = True,
        temperature: Optional[float] = None,
        eval_output_size: Optional[Tuple[int]] = None,
    ):
        super().__init__()
        self.allocator = allocator
        self.renderer = renderer
        self.eval_output_size = list(eval_output_size) if eval_output_size else None

        att_dim = max(embed_dim, inp_dim)
        self.scale = att_dim**-0.5 if temperature is None else temperature**-1
        self.to_q = nn.Linear(embed_dim, att_dim, bias=False)
        self.to_k = nn.Linear(inp_dim, att_dim, bias=False)

        if use_layer_norms:
            self.norm_k = nn.LayerNorm(inp_dim, eps=1e-5)
            self.norm_q = nn.LayerNorm(embed_dim, eps=1e-5)
            self.norm_memory = norm_memory
            if norm_memory:
                self.norm_memory = nn.LayerNorm(inp_dim, eps=1e-5)
            else:
                self.norm_memory = nn.Identity()
        else:
            self.norm_k = nn.Identity()
            self.norm_q = nn.Identity()
            self.norm_memory = nn.Identity()

        if output_transform is None:
            if renderer_dim is None:
                raise ValueError("Need to provide render_mlp_dim if output_transform is unspecified")
            self.output_transform = nn.Linear(renderer_dim, outp_dim)
        else:
            self.output_transform = output_transform

        if pos_embed_mode is not None and pos_embed_mode not in ("none", "add", "concat"):
            raise ValueError("If set, `pos_embed_mode` should be 'none', 'add' or 'concat'")
        self.pos_embed_mode = pos_embed_mode
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, embed_dim) * embed_dim**-0.5)
        self.init_parameters()

    def init_parameters(self):
        layers = [self.to_q, self.to_k]
        if isinstance(self.output_transform, nn.Linear):
            layers.append(self.output_transform)
        utils.init_parameters(layers, "xavier_uniform")

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.training and self.eval_output_size is not None:
            pos_emb = timm.layers.pos_embed.resample_abs_pos_embed(
                self.pos_emb,
                new_size=self.eval_output_size,
                num_prefix_tokens=0,
            )
        else:
            pos_emb = self.pos_emb

        pos_emb = pos_emb.expand(len(slots), -1, -1)
        memory = self.norm_memory(slots)
        query_features = self.allocator(pos_emb, memory=memory)
        q = self.to_q(self.norm_q(query_features))  # B x P x D
        k = self.to_k(self.norm_k(slots))  # B x S x D

        dots = torch.einsum("bpd, bsd -> bps", q, k) * self.scale
        attn = dots.softmax(dim=-1)

        mixed_slots = torch.einsum("bps, bsd -> bpd", attn, slots)  # B x P x D

        if self.pos_embed_mode == "add":
            mixed_slots = mixed_slots + pos_emb
        elif self.pos_embed_mode == "concat":
            mixed_slots = torch.cat((mixed_slots, pos_emb), dim=-1)

        features = self.renderer(mixed_slots)
        recons = self.output_transform(features)

        return {"reconstruction": recons, "masks": attn.transpose(-2, -1)}


def algebraic_slot_utility_loss(
    mix: torch.Tensor,
    slot_recons: torch.Tensor,
    masks: torch.Tensor,
    target: torch.Tensor,
    gate: torch.Tensor,
    margin: float = 1.0,
    min_mask: float = 0.0,
    mix_eps: float = 1e-8,
    masks_ungated: Optional[torch.Tensor] = None,
    mix_ungated: Optional[torch.Tensor] = None,
    add_insert: bool = False,
    return_aux: bool = False,
    teacher: str = "rent",
    ce_tau: float = 0.5,
    logits: Optional[torch.Tensor] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
    """All-slot leave-one-out teacher (no extra decode).

    Drop (Δ↓) is z-free when mix_ungated / masks_ungated = softmax(α) mix are
    given (v44). Then mix_{-s} = (ŷ^u - σ_s R_s)/(1-σ_s) on ungated territory.
    v42 omits those and drop uses the gated mix (legacy).

    Add (Δ↑, v44): restore slot s to its α-share on the gated canvas,
    ŷ^{+s} = σ_s R_s + (1-σ_s) ŷ^{g,-s}, only on patches where σ_s > m_s.

    teacher='rent' (v42/v44):
        L = mean_{b,t} sum_s z_{t,s} * sg(c_{t,s})
        c = clamp( [1-Δ↓/margin]_+ - [Δ↑]_+, -1, 1 )   (add_insert=False → drop only)
        Rent is detached; live `gate` (`z`) only enters the inner product.

    teacher='ce' (v45):
        u = [Δ↓]_+ + [Δ↑]_+ ,  π = softmax(u / τ)
        L = mean CE(π, z) = -⟨π, log z⟩.  ∂L/∂ℓ = z-π if z=softmax(ℓ).
        π is detached. Optional live `logits` uses log_softmax (stable).
    """
    del min_mask  # drop-mode candidate cutoff; algebraic taxes all slots
    if mix.ndim == 3:
        mix = mix.unsqueeze(1)
        target = target.unsqueeze(1)
        slot_recons = slot_recons.unsqueeze(1)
        masks = masks.unsqueeze(1)
        if gate.ndim == 2:
            gate = gate.unsqueeze(1)
        if masks_ungated is not None and masks_ungated.ndim == 3:
            masks_ungated = masks_ungated.unsqueeze(1)
        if mix_ungated is not None and mix_ungated.ndim == 3:
            mix_ungated = mix_ungated.unsqueeze(1)
        if logits is not None and logits.ndim == 2:
            logits = logits.unsqueeze(1)
    if mix.ndim != 4 or slot_recons.ndim != 5 or masks.ndim != 4:
        raise ValueError(
            "algebraic_slot_utility_loss expected mix (B,T,F,D), "
            f"got mix {tuple(mix.shape)}, slot_recons {tuple(slot_recons.shape)}, "
            f"masks {tuple(masks.shape)}"
        )
    if gate.ndim != 3:
        raise ValueError(
            f"algebraic_slot_utility_loss expected gate (B,T,S), got {tuple(gate.shape)}"
        )
    if logits is not None and logits.ndim == 2:
        logits = logits.unsqueeze(1)
    teacher_key = str(teacher).strip().lower()
    if teacher_key in ("inner", "dot"):
        teacher_key = "rent"
    if teacher_key not in ("rent", "ce"):
        raise ValueError(
            f"algebraic_slot_utility_loss teacher must be 'rent' or 'ce', got {teacher!r}"
        )
    if teacher_key == "ce" and float(ce_tau) <= 0.0:
        raise ValueError(f"algebraic_slot_utility_loss ce_tau must be > 0, got {ce_tau}")

    add_insert = bool(add_insert)
    if add_insert:
        if masks_ungated is None:
            raise ValueError("algebraic σ-m insert add requires masks_ungated = softmax(α)")
        if masks_ungated.shape != masks.shape:
            raise ValueError(
                f"masks_ungated {tuple(masks_ungated.shape)} vs masks {tuple(masks.shape)}"
            )

    # v44: drop Δ on softmax(α) mix so rent does not see the current gate.
    # v42 omits masks_ungated and keeps gated drop. Add still uses gated mix.
    use_ungated_drop = masks_ungated is not None
    if use_ungated_drop:
        drop_masks = masks_ungated
        if mix_ungated is None:
            mix_ungated = (drop_masks.unsqueeze(-1) * slot_recons).sum(dim=2)
        if mix_ungated.shape != mix.shape:
            raise ValueError(
                f"mix_ungated {tuple(mix_ungated.shape)} vs mix {tuple(mix.shape)}"
            )
        drop_mix = mix_ungated
    else:
        drop_mix = mix
        drop_masks = masks

    n_slots = drop_masks.shape[2]
    err_drop_full = (drop_mix - target).pow(2).mean(-1)  # (B, T, F)
    err_drop_mean = err_drop_full.mean(dim=-1).clamp_min(mix_eps)
    err_gate_full = (mix - target).pow(2).mean(-1)
    err_gate_mean = err_gate_full.mean(dim=-1).clamp_min(mix_eps)
    mar = max(float(margin), mix_eps)

    rel_parts = []
    add_parts = []
    with torch.no_grad():
        for s in range(n_slots):
            m_s = drop_masks[:, :, s].unsqueeze(-1)
            r_s = slot_recons[:, :, s]
            denom = (1.0 - drop_masks[:, :, s]).clamp_min(mix_eps).unsqueeze(-1)
            mix_drop = (drop_mix - m_s * r_s) / denom
            err_drop = (mix_drop - target).pow(2).mean(-1)
            w = drop_masks[:, :, s]
            wsum = w.sum(dim=-1).clamp_min(mix_eps)
            delta = ((err_drop - err_drop_full) * w).sum(dim=-1) / wsum
            rel_parts.append(delta / err_drop_mean)
        rel_delta = torch.stack(rel_parts, dim=-1)
        rent_drop = (1.0 - rel_delta / mar).clamp(min=0.0, max=1.0)

        rel_add = None
        suppressed = None
        if add_insert:
            sm = masks_ungated.float()
            m_g = masks.float()
            rec = slot_recons.float()
            mix_g = mix.float()
            tgt = target.float()
            err_g = err_gate_full.float()
            err_g_mean = err_gate_mean.float()
            supp_parts = []
            for s in range(n_slots):
                m_s = m_g[:, :, s]
                sm_s = sm[:, :, s]
                r_s = rec[:, :, s]
                denom_g = (1.0 - m_s).clamp_min(mix_eps).unsqueeze(-1)
                mix_wo = (mix_g - m_s.unsqueeze(-1) * r_s) / denom_g
                mix_ins = sm_s.unsqueeze(-1) * r_s + (1.0 - sm_s).unsqueeze(-1) * mix_wo
                err_ins = (mix_ins - tgt).pow(2).mean(-1)
                w = (sm_s - m_s).clamp(min=0.0)
                wsum = w.sum(dim=-1)
                d_up = ((err_g - err_ins) * w).sum(dim=-1) / wsum.clamp_min(mix_eps)
                d_up = torch.where(wsum > mix_eps, d_up, torch.zeros_like(d_up))
                add_parts.append(d_up / err_g_mean)
                supp_parts.append((wsum > mix_eps).to(dtype=rel_delta.dtype))
            rel_add = torch.stack(add_parts, dim=-1).to(dtype=rel_delta.dtype)
            suppressed = torch.stack(supp_parts, dim=-1)
            bonus = rel_add.clamp(min=0.0)
            rent = (rent_drop - bonus).clamp(min=-1.0, max=1.0)
        else:
            rent = rent_drop

    z = gate.float()
    if z.shape != rent.shape:
        raise ValueError(
            f"algebraic_slot_utility_loss gate {tuple(z.shape)} vs rent {tuple(rent.shape)}"
        )
    if teacher_key == "ce":
        u = rel_delta.clamp(min=0.0)
        if rel_add is not None:
            u = u + rel_add.clamp(min=0.0)
        pi = torch.softmax(u / float(ce_tau), dim=-1)
        if logits is not None:
            if logits.shape != z.shape:
                raise ValueError(
                    f"algebraic_slot_utility_loss logits {tuple(logits.shape)} "
                    f"vs gate {tuple(z.shape)}"
                )
            log_z = torch.log_softmax(logits.float(), dim=-1)
        else:
            log_z = z.clamp_min(mix_eps).log()
        loss = -(pi * log_z).sum(dim=-1).mean()
    else:
        pi = None
        u = None
        loss = (z * rent).sum(dim=-1).mean()
    if not return_aux:
        return loss, rel_delta
    aux: Dict[str, torch.Tensor] = {"cost": rent, "rel_delta": rel_delta}
    if rel_add is not None:
        aux["rel_add"] = rel_add
        if suppressed is not None:
            aux["suppressed"] = suppressed
    if pi is not None:
        aux["pi"] = pi
        aux["u"] = u
    return loss, rel_delta, aux

