from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import timm
import torch
import torchvision
from torch import nn

from slotcurri.modules import utils
from slotcurri.utils import config_as_kwargs, make_build_fn


@make_build_fn(__name__, "encoder")
def build(config, name: str):
    if name == "FrameEncoder":
        pos_embed = None
        if config.get("pos_embed"):
            pos_embed = utils.build_module(config.pos_embed)

        output_transform = None
        if config.get("output_transform"):
            output_transform = utils.build_module(config.output_transform)
        return FrameEncoder(
            backbone=utils.build_module(config.backbone, default_group="encoders"),
            pos_embed=pos_embed,
            output_transform=output_transform,
            **config_as_kwargs(config, ("backbone", "pos_embed", "output_transform")),
        )
    else:
        return None


class FeatureSmoothing(nn.Module):
    """Affinity-based within-object feature homogenization (v33 task curriculum).

    Given backbone tokens x (B, F, D), one step of non-parametric self-attention

        P = softmax_j( cos(x_i, x_j) / tau ),   x <- P x    (n_steps times)

    averages every token with its feature-space neighbours. Because DINO affinities are
    block-structured by object, this shrinks WITHIN-object feature variance toward zero
    while approximately preserving object boundaries (cross-object affinities are small)
    -- unlike a spatial blur, which mixes across boundaries. The returned tensor is the
    scheduled blend

        x_used = (1 - mix) * x_tilde + mix * x

    with mix = 0 fully smoothed (object-level features, a part-split earns nothing and
    cannot hold clean ownership) and mix = 1 raw (exact no-op, the module returns x
    itself). `window` is a Chebyshev radius on a reference grid (`window_ref_grid`,
    default 37 = YTVIS 518/14). At runtime it is scaled to the actual token grid

        w_eff = max(1, round(window * side / window_ref_grid))

    so the same yaml keeps a constant *relative* neighbourhood on MOVi-C (24x24)
    and YTVIS (37x37). Set `window_ref_grid` to null for an absolute radius.
    `n_steps` > 1 walks that local graph so nearby parts can mix without a
    global same-class hop. `window` may be rewritten each training step (v34
    window-anneal); `window <= 0` is an identity, not a 1-hop neighbourhood.

    P is computed under no_grad and x_tilde is detached: the smoothing is a pure data
    transform of the (frozen) backbone tokens, introducing no new gradient paths. The
    module holds no parameters, so checkpoints stay byte-compatible either way.
    """

    def __init__(
        self,
        tau: float = 0.1,
        window: Optional[int] = None,
        n_steps: int = 1,
        chunk_size: int = 16,
        window_ref_grid: Optional[int] = 37,
    ):
        super().__init__()
        self.tau = float(tau)
        self.window = int(window) if window is not None else None
        self.n_steps = max(int(n_steps), 1)
        self.chunk_size = max(int(chunk_size), 1)
        self.window_ref_grid = (
            int(window_ref_grid) if window_ref_grid is not None else None
        )
        self._mask_cache: Dict[Any, Optional[torch.Tensor]] = {}

    def effective_window(self, n_tokens: int) -> Optional[int]:
        """Chebyshev radius on this token grid, or None for global affinity.

        `window is None` is global. `window <= 0` is "smoothing off" (radius 0),
        not a 1-hop neighbourhood -- the v33 `max(1, ...)` scale must not revive
        a hop after the window clock has reached raw.
        """
        if self.window is None:
            return None
        if self.window <= 0:
            return 0
        side = int(round(n_tokens**0.5))
        if side * side != n_tokens:
            return None
        if self.window_ref_grid is None or self.window_ref_grid <= 0:
            return self.window
        return max(1, int(round(self.window * side / float(self.window_ref_grid))))

    def _window_mask(self, n_tokens: int, device: torch.device) -> Optional[torch.Tensor]:
        """True where the affinity is masked out (grid Chebyshev distance > w_eff)."""
        if self.window is None:
            return None
        key = (n_tokens, str(device), self.window, self.window_ref_grid)
        if key in self._mask_cache:
            return self._mask_cache[key]
        # w<=0 is "smoothing off". None would mean global affinity -- the opposite.
        if self.window <= 0:
            idx = torch.arange(n_tokens, device=device)
            mask = idx[:, None] != idx[None, :]
            self._mask_cache[key] = mask
            return mask
        w_eff = self.effective_window(n_tokens)
        if w_eff is None:
            self._mask_cache[key] = None
            return None
        side = int(round(n_tokens**0.5))
        idx = torch.arange(n_tokens, device=device)
        ys, xs = idx // side, idx % side
        cheb = torch.maximum(
            (ys[:, None] - ys[None, :]).abs(), (xs[:, None] - xs[None, :]).abs()
        )
        mask = cheb > w_eff
        self._mask_cache[key] = mask
        return mask

    @torch.no_grad()
    def _smooth_once(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, D). Chunked over the batch so the (F, F) affinity is transient.
        mask = self._window_mask(x.shape[1], x.device)
        out = torch.empty_like(x)
        for i in range(0, x.shape[0], self.chunk_size):
            xc = x[i : i + self.chunk_size].float()
            xn = torch.nn.functional.normalize(xc, dim=-1)
            logits = torch.bmm(xn, xn.transpose(1, 2)) / self.tau
            if mask is not None:
                logits = logits.masked_fill(mask, float("-inf"))
            p = logits.softmax(dim=-1)
            out[i : i + self.chunk_size] = torch.bmm(p, xc).to(x.dtype)
        return out

    @torch.no_grad()
    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for _ in range(self.n_steps):
            y = self._smooth_once(y)
        return y

    def forward(self, x: torch.Tensor, mix: float) -> torch.Tensor:
        mix = float(mix)
        if mix >= 1.0:
            return x
        # window-anneal clock: w<=0 means raw, not "mask is None so go global".
        if self.window is not None and self.window <= 0:
            return x
        smoothed = self._smooth(x.detach())
        if mix <= 0.0:
            return smoothed
        return (1.0 - mix) * smoothed + mix * x


class NcutRelationalLeveling(nn.Module):
    """ReLU-cosine graph leveling, optionally with a 2-way Ncut region barrier.

    Frozen DINO tokens x (B, N, D) become a cosine graph

        Z = x / ||x||,   W = ReLU(Z Z^T),   W_ii = 0

    Then x_rel = P x with P the row-normalized affinity. `barrier=True` (v36) first
    takes the Fiedler vector of L = I - D^{-1/2} W D^{-1/2}, median-cuts into two
    regions, and zeros W across the cut so leveling cannot mix the parts.
    `barrier=False` (v37) skips that cut: P is global on W, same mix / Key-only
    schedule as v36.

    The Fiedler is batched subspace iteration (k=2) plus a 2x2 Rayleigh-Ritz, not
    a full `eigh`. Isolated rows keep the original token.

    Same mix convention as FeatureSmoothing: mix=0 is fully leveled (X^rel),
    mix=1 is raw (identity). P / Ncut run under no_grad; the module has no
    parameters. `chunk_size` bounds the (N, N) work, not the math.
    """

    def __init__(
        self,
        chunk_size: int = 8,
        eps: float = 1e-6,
        n_iter: int = 16,
        barrier: bool = True,
    ):
        super().__init__()
        self.chunk_size = max(int(chunk_size), 1)
        self.eps = float(eps)
        self.n_iter = max(int(n_iter), 1)
        self.barrier = bool(barrier)

    def _fiedler(self, w_norm: torch.Tensor) -> torch.Tensor:
        """2nd-largest eigenvector of W_norm (Fiedler of L = I - W_norm).

        Batched subspace iteration on k=2, then Rayleigh-Ritz. Deterministic
        init (constant + linspace) so chunking is bit-stable.
        """
        bsz, n_tokens, _ = w_norm.shape
        ones = torch.ones(bsz, n_tokens, 1, device=w_norm.device, dtype=w_norm.dtype)
        grid = torch.linspace(-1.0, 1.0, n_tokens, device=w_norm.device, dtype=w_norm.dtype)
        x = torch.cat([ones, grid.view(1, n_tokens, 1).expand(bsz, -1, -1)], dim=-1)
        for _ in range(self.n_iter):
            x, _ = torch.linalg.qr(torch.bmm(w_norm, x))
        rax = torch.bmm(x.transpose(1, 2), torch.bmm(w_norm, x))
        rax = 0.5 * (rax + rax.transpose(1, 2))
        _, evecs = torch.linalg.eigh(rax)  # ascending: col 0 = smaller = Fiedler
        return torch.bmm(x, evecs)[:, :, 0]

    @torch.no_grad()
    def explain(self, x: torch.Tensor) -> dict:
        """Same mix=0 graph as training. Returns rel, region, v2. x: (B, N, D)."""
        rels, regions, v2s = [], [], []
        for i in range(0, x.shape[0], self.chunk_size):
            rel, region, v2 = self._graph_chunk(x[i : i + self.chunk_size])
            rels.append(rel.to(dtype=x.dtype))
            regions.append(region)
            v2s.append(v2.to(dtype=x.dtype))
        return {
            "rel": torch.cat(rels, dim=0),
            "region": torch.cat(regions, dim=0),
            "v2": torch.cat(v2s, dim=0),
        }

    @torch.no_grad()
    def _graph_chunk(self, x: torch.Tensor):
        # AMP would downcast bmm/qr to fp16; CUDA geqrf has no Half kernel.
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            z = torch.nn.functional.normalize(x, dim=-1)
            w = torch.bmm(z, z.transpose(1, 2)).clamp_min(0.0)
            w.diagonal(dim1=-2, dim2=-1).zero_()

            if self.barrier:
                degree = w.sum(dim=-1).clamp_min(self.eps)
                d_inv_sqrt = degree.rsqrt()
                w_norm = d_inv_sqrt.unsqueeze(-1) * w * d_inv_sqrt.unsqueeze(-2)
                v2 = self._fiedler(w_norm)
                threshold = v2.median(dim=-1, keepdim=True).values
                region = v2 >= threshold  # (B, N)
                same = region.unsqueeze(-1) == region.unsqueeze(-2)
                w_tilde = w.masked_fill(~same, 0.0)
            else:
                # v37: same W / P / mix as v36, no 2-way region. One region, v2 unused.
                v2 = torch.zeros(x.shape[:2], device=x.device, dtype=x.dtype)
                region = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
                w_tilde = w

            row = w_tilde.sum(dim=-1, keepdim=True)
            p = w_tilde / row.clamp_min(self.eps)
            rel = torch.bmm(p, x)
            rel = torch.where(row < self.eps, x, rel)
            return rel, region, v2

    @torch.no_grad()
    def _level_chunk(self, x: torch.Tensor) -> torch.Tensor:
        rel, _, _ = self._graph_chunk(x)
        return rel

    @torch.no_grad()
    def _level(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(0, x.shape[0], self.chunk_size):
            chunk = x[i : i + self.chunk_size]
            rel = self._level_chunk(chunk.float()).to(dtype=chunk.dtype)
            outs.append(rel)
        return torch.cat(outs, dim=0)

    def forward(self, x: torch.Tensor, mix: float) -> torch.Tensor:
        mix = float(mix)
        if mix >= 1.0:
            return x
        leveled = self._level(x.detach())
        if mix <= 0.0:
            return leveled
        return (1.0 - mix) * leveled + mix * x


class FeatureModeCollapse(nn.Module):
    """Cover tokens by density-mode means (v35 task curriculum).

    Affinity smoothing keeps F distinct keys, so leftover slots still carve exclusive
    seats. This module replaces each token by the mean of its mean-shift attractor so

        #{k_f} = C(scene, h)

    and slot attention cannot split two patches that share a key. C is not a
    hyperparameter: it is the number of kernel modes at bandwidth `h`.

    Blurring mean-shift on (a probe of) the L2-normalized tokens

        w_ij = exp((cos(m_i, m_j) - 1) / h),   m <- normalize(W m)

    then clumps attractors with cos > 1 - delta and covers every token with the
    mean of the original tokens assigned to that clump.

    `bandwidth` is rewritten each training step (v35 modes-anneal). `bandwidth <=
    h_min` or mix >= 1 is identity. No parameters; P is computed under no_grad.
    """

    def __init__(
        self,
        n_iter: int = 8,
        delta: float = 0.02,
        h_min: float = 0.02,
        probe: int = 128,
        chunk_size: int = 16,
        clump_iters: int = 32,
    ):
        super().__init__()
        self.n_iter = max(int(n_iter), 1)
        self.delta = float(delta)
        self.h_min = float(h_min)
        self.probe = int(probe)
        self.chunk_size = max(int(chunk_size), 1)
        self.clump_iters = max(int(clump_iters), 1)
        self.bandwidth: float = 0.0
        self.last_c: Optional[torch.Tensor] = None

    def _probe_idx(self, n_tokens: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.probe <= 0 or self.probe >= n_tokens:
            return None
        return torch.linspace(0, n_tokens - 1, self.probe, device=device).round().long().unique()

    def _clump(self, m: torch.Tensor) -> torch.Tensor:
        """Min-label connected components on {cos(m_i, m_j) > 1 - delta}."""
        # m: (B, P, D) unit rows
        cos = torch.bmm(m, m.transpose(1, 2))
        adj = cos > (1.0 - self.delta)
        bsz, n_probe, _ = m.shape
        labels = torch.arange(n_probe, device=m.device).expand(bsz, n_probe).contiguous()
        sentinel = n_probe
        for _ in range(self.clump_iters):
            lab = labels.unsqueeze(1).expand(bsz, n_probe, n_probe)
            nxt = lab.masked_fill(~adj, sentinel).amin(dim=-1)
            if torch.equal(nxt, labels):
                break
            labels = nxt
        return labels

    def _cover_chunk(self, x: torch.Tensor, h: float) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, F, D)
        xf = x.float()
        xn = torch.nn.functional.normalize(xf, dim=-1)
        bsz, n_tokens, dim = xf.shape
        probe = self._probe_idx(n_tokens, x.device)
        xp = xn if probe is None else xn.index_select(1, probe)
        src = xf if probe is None else xf.index_select(1, probe)

        m = xp
        inv_h = 1.0 / max(h, 1e-6)
        for _ in range(self.n_iter):
            logits = torch.bmm(m, m.transpose(1, 2))
            weight = torch.exp((logits - 1.0) * inv_h)
            m = torch.nn.functional.normalize(torch.bmm(weight, m), dim=-1)

        labels = self._clump(m)
        n_probe = src.shape[1]
        onehot = torch.nn.functional.one_hot(labels, n_probe).to(dtype=src.dtype)
        counts = onehot.sum(dim=1).clamp_min(1e-8)
        attract = torch.bmm(onehot.transpose(1, 2), src) / counts.unsqueeze(-1)
        used = onehot.sum(dim=1) > 0

        attr_n = torch.nn.functional.normalize(attract, dim=-1)
        assign_logits = torch.bmm(xn, attr_n.transpose(1, 2))
        assign_logits = assign_logits.masked_fill(~used.unsqueeze(1), -2.0)
        assign = assign_logits.argmax(dim=-1)

        onehot_f = torch.nn.functional.one_hot(assign, n_probe).to(dtype=src.dtype)
        counts_f = onehot_f.sum(dim=1).clamp_min(1e-8)
        mu = torch.bmm(onehot_f.transpose(1, 2), xf) / counts_f.unsqueeze(-1)
        covered = mu.gather(1, assign.unsqueeze(-1).expand(bsz, n_tokens, dim))
        n_modes = (onehot_f.sum(dim=1) > 0).sum(dim=-1)
        return covered.to(dtype=x.dtype), n_modes

    @torch.no_grad()
    def _cover(self, x: torch.Tensor, h: float) -> torch.Tensor:
        outs = []
        counts = []
        for i in range(0, x.shape[0], self.chunk_size):
            cov, n_modes = self._cover_chunk(x[i : i + self.chunk_size], h)
            outs.append(cov)
            counts.append(n_modes)
        self.last_c = torch.cat(counts, dim=0)
        return torch.cat(outs, dim=0)

    def forward(self, x: torch.Tensor, mix: float) -> torch.Tensor:
        mix = float(mix)
        if mix >= 1.0:
            self.last_c = None
            return x
        h = float(self.bandwidth)
        if h <= self.h_min:
            self.last_c = None
            return x
        return self._cover(x.detach(), h)


class FrameEncoder(nn.Module):
    """Module reducing image to set of features."""

    def __init__(
        self,
        backbone: nn.Module,
        pos_embed: Optional[nn.Module] = None,
        output_transform: Optional[nn.Module] = None,
        spatial_flatten: bool = False,
        main_features_key: str = "vit_block12",
    ):
        super().__init__()
        self.backbone = backbone
        self.pos_embed = pos_embed
        self.output_transform = output_transform
        self.spatial_flatten = spatial_flatten
        self.main_features_key = main_features_key
        # v33 feature curriculum: optional affinity smoothing of the backbone tokens,
        # attached post-build by the model. With apply=tokens it runs BEFORE the
        # features/backbone_features split, so grouper and recon target both derive
        # from the smoothed tensor. apply=key (v36/v37) leaves the target raw and
        # only feeds the bind tensor to Keys. Mix 0 = fully coarsened, 1 = raw;
        # train()-only, so validation and eval always see raw features.
        self.feature_smoothing: Optional[nn.Module] = None
        self.feature_smoothing_mix: float = 1.0
        # v35 modes curriculum: attached post-build like FeatureSmoothing. Cover is
        # train()-only; mix >= 1 or bandwidth <= h_min is identity (eval stays raw).
        self.feature_modes: Optional[nn.Module] = None
        self.feature_modes_mix: float = 1.0
        # v36 Ncut leveling: attached post-build. Mix 0 = fully leveled, 1 = raw.
        # `feature_curriculum_apply == "key"` keeps the reconstruction target (and the
        # Value path) on the original tokens and only feeds the bind tensor to Keys.
        # v37 uses the same Key-only split with Ncut leveling, barrier=false.
        self.feature_ncut: Optional[nn.Module] = None
        self.feature_ncut_mix: float = 1.0
        self.feature_curriculum_apply: str = "tokens"

    def _curriculum_bind(self, tokens: torch.Tensor) -> torch.Tensor:
        """Train-only coarsened tokens (X^bind). Eval and mix=1 skip this."""
        if not self.training or tokens.ndim != 3:
            return tokens
        if self.feature_ncut is not None and self.feature_ncut_mix < 1.0:
            return self.feature_ncut(tokens, self.feature_ncut_mix)
        if self.feature_modes is not None and self.feature_modes_mix < 1.0:
            return self.feature_modes(tokens, self.feature_modes_mix)
        if self.feature_smoothing is not None and self.feature_smoothing_mix < 1.0:
            return self.feature_smoothing(tokens, self.feature_smoothing_mix)
        return tokens

    def _curriculum_target_and_key(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(tokens used as recon target / Value, tokens used as Key).

        `apply=tokens` (v33/v34/v35): both sides are the curriculumed tensor.
        `apply=key` (v36/v37 Ncut mix): target/Value stay raw; Key gets
        the mixed bind tensor. Eval and mix=1 leave both as the original tokens.
        """
        if tokens.ndim != 3:
            return tokens, tokens
        bind = self._curriculum_bind(tokens)
        if str(self.feature_curriculum_apply).lower() == "key":
            return tokens, bind
        return bind, bind

    def _project_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        features = tokens.clone()
        if self.pos_embed:
            features = self.pos_embed(features)
        if self.spatial_flatten:
            features = einops.rearrange(features, "b c h w -> b (h w) c")
        if self.output_transform:
            features = self.output_transform(features)
        assert (
            features.ndim == 3
        ), f"Expect output shape (batch, tokens, dims), but got {features.shape}"
        return features

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # images: batch x n_channels x height x width
        backbone_features = self.backbone(images)
        if isinstance(backbone_features, dict):
            main = backbone_features[self.main_features_key]
            target_tokens, key_tokens = self._curriculum_target_and_key(main)
            backbone_features[self.main_features_key] = target_tokens
        else:
            target_tokens, key_tokens = self._curriculum_target_and_key(backbone_features)
            backbone_features = target_tokens

        features = self._project_tokens(target_tokens)
        features_key = (
            None if key_tokens is target_tokens else self._project_tokens(key_tokens)
        )

        if isinstance(backbone_features, dict):
            for k, backbone_feature in backbone_features.items():
                if self.spatial_flatten:
                    backbone_features[k] = einops.rearrange(backbone_feature, "b c h w -> b (h w) c")
                assert (
                    backbone_feature.ndim == 3
                ), f"Expect output shape (batch, tokens, dims), but got {backbone_feature.shape}"
            main_backbone_features = backbone_features[self.main_features_key]
            out = {
                "features": features,
                "backbone_features": main_backbone_features,
                **backbone_features,
            }
        else:
            if self.spatial_flatten:
                backbone_features = einops.rearrange(backbone_features, "b c h w -> b (h w) c")
            assert (
                backbone_features.ndim == 3
            ), f"Expect output shape (batch, tokens, dims), but got {backbone_features.shape}"
            out = {
                "features": features,
                "backbone_features": backbone_features,
            }
        if features_key is not None:
            out["features_key"] = features_key
            # DINO-dim X^bind for spectral purity; projected Keys are features_key.
            out["backbone_key"] = key_tokens
        return out


class TimmExtractor(nn.Module):
    """Feature extractor utilizing models from timm library."""

    # Convenience aliases for feature keys
    FEATURE_ALIASES = {
        **{f"resnet_block{i}": f"layer{i}" for i in range(1, 5)},
        **{f"vit_block{i + 1}": f"blocks.{i}" for i in range(12)},
        **{f"vit_block_values{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_queries{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_keys{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        "vit_output": "norm",
    }
    FEATURE_MAPPING = {
        **{f"layer{i}": f"resnet_block{i}" for i in range(1, 5)},
        **{f"blocks.{i}": f"vit_block{i + 1}" for i in range(12)},
        **{f"blocks.{i}.attn.qkv": f"vit_block_keys{i + 1}" for i in range(12)},
        "norm": "vit_output",
    }

    def __init__(
        self,
        model: str,
        pretrained: bool = False,
        frozen: bool = False,
        features: Optional[Union[str, List[str]]] = None,
        checkpoint_path: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        model_name = model
        self.frozen = frozen
        self.features = [features] if isinstance(features, str) else features
        self.is_vit = model_name.startswith("vit")

        model = TimmExtractor._create_model(model_name, pretrained, checkpoint_path, model_kwargs)

        if self.features is not None:
            nodes = torchvision.models.feature_extraction.get_graph_node_names(model)[0]

            features = []
            for name in self.features:
                if name in TimmExtractor.FEATURE_ALIASES:
                    name = TimmExtractor.FEATURE_ALIASES[name]

                if not any(node.startswith(name) for node in nodes):
                    raise ValueError(
                        f"Requested features under node {name}, but this node does "
                        f"not exist in model {model_name}. Available nodes: {nodes}"
                    )

                features.append(name)

            model = torchvision.models.feature_extraction.create_feature_extractor(model, features)

        self.model = model

        if self.frozen:
            self.requires_grad_(False)

    @staticmethod
    def _create_model(
        model_name: str,
        pretrained: bool,
        checkpoint_path: Optional[str],
        model_kwargs: Optional[Dict[str, Any]],
        trials: int = 0,
    ) -> nn.Module:
        if model_kwargs is None:
            model_kwargs = {}

        try:
            model = timm.create_model(
                model_name, pretrained=pretrained, checkpoint_path=checkpoint_path, **model_kwargs
            )
        except (FileExistsError, FileNotFoundError):
            # Timm uses Hugginface hub for loading the files, which does some symlinking in the
            # background when loading the checkpoint. When multiple concurrent jobs attempt to
            # load the checkpoint, this can create conflicts, because the symlink is first removed,
            # then created again by each job. We attempt to catch the resulting errors here, and
            # retry creating the model, up to 3 times.
            if trials == 2:
                raise
            else:
                model = None

        if model is None:
            model = TimmExtractor._create_model(
                model_name, pretrained, checkpoint_path, model_kwargs, trials=trials + 1
            ) # vit_small_patch14_dinov2

        return model

    def forward(self, inp):
        if self.frozen:
            with torch.no_grad():
                outputs = self.model(inp)

                # blocks = list(self.model.blocks.children())
                # x = inp
                # for blk in blocks:
                #     x = blk(x)
                # for blk in self.model.blocks:
                # x = blk(x)
        else:
            outputs = self.model(inp)
            # for blk in self.model.blocks:
            #     x = blk(x)

        # print()
        # print('encoders keys: ', outputs.keys()) # encoders keys: dict_keys(['blocks.11.attn.qkv', 'blocks.11'])
        # print(outputs['blocks.11'].shape) # 384 577 384
        # print(outputs['blocks.11.attn.qkv'].shape) # 384 577 1152


        # exit(1)

        if self.features is not None:
            if self.is_vit:
                outputs = {k: v[:, 1:] for k, v in outputs.items()}  # Remove CLS token
            outputs = {self.FEATURE_MAPPING[key]: value for key, value in outputs.items()}
            # print(outputs.keys())
            # print(self.features)
            # exit(1)
            for name in self.features:
                if ("keys" in name) or ("queries" in name) or ("values" in name):
                    feature_name = name.replace("queries", "keys").replace("values", "keys")
                    B, N, C = outputs[feature_name].shape
                    qkv = outputs[feature_name].reshape(B, N, 3, C // 3)  # outp has shape B, N, 3 * H * (C // H)
                    # print(feature_name, name) # vit_block_keys12 vit_block_keys12
                    q, k, v = qkv.unbind(2)
                    # print(k.shape, q.shape, v.shape)
                    if "keys" in name: # default
                        outputs[name] = k
                    elif "queries" in name:
                        outputs[name] = q
                    elif "values" in name:
                        outputs[name] = v
                    else:
                        raise ValueError(f"Unknown feature name {name}.")
            # exit(1)
            if len(outputs) == 1:
                # Unpack single output for now
                return next(iter(outputs.values()))
            else:
                return outputs
        else:
            return outputs
