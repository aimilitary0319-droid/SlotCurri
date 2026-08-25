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
    """2-way Ncut barrier + region-constrained global leveling (v36 task curriculum).

    Frozen DINO tokens x (B, N, D) become a cosine graph

        Z = x / ||x||,   W = ReLU(Z Z^T),   W_ii = 0

    The symmetric normalized Laplacian L = I - D^{-1/2} W D^{-1/2} is decomposed
    with `eigh`; the Fiedler vector (second-smallest eigenvalue, index 1) is split
    at the per-frame median into two coarse regions r in {0, 1}. Affinity across
    regions is zeroed, the remaining graph is row-normalized, and

        x_rel = P x

    globally levels tokens inside each region while blocking mix across the cut.
    Isolated rows (no remaining neighbours) keep the original token.

    Same mix convention as FeatureSmoothing: mix=0 is fully leveled (X^rel),
    mix=1 is raw (identity). P / Ncut run under no_grad; the module has no
    parameters. `chunk_size` bounds the (N, N) eigenproblem, not the math.
    """

    def __init__(self, chunk_size: int = 8, eps: float = 1e-6):
        super().__init__()
        self.chunk_size = max(int(chunk_size), 1)
        self.eps = float(eps)

    @torch.no_grad()
    def _level_chunk(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) float32
        z = torch.nn.functional.normalize(x, dim=-1)
        w = torch.bmm(z, z.transpose(1, 2)).clamp_min(0.0)
        w.diagonal(dim1=-2, dim2=-1).zero_()

        degree = w.sum(dim=-1).clamp_min(self.eps)
        d_inv_sqrt = degree.rsqrt()
        w_norm = d_inv_sqrt.unsqueeze(-1) * w * d_inv_sqrt.unsqueeze(-2)
        n_tokens = x.shape[1]
        eye = torch.eye(n_tokens, device=x.device, dtype=x.dtype).unsqueeze(0)
        laplacian = eye - w_norm
        _, eigvecs = torch.linalg.eigh(laplacian)
        v2 = eigvecs[:, :, 1]
        threshold = v2.median(dim=-1, keepdim=True).values
        region = v2 >= threshold  # (B, N)

        same = region.unsqueeze(-1) == region.unsqueeze(-2)
        w_tilde = w.masked_fill(~same, 0.0)
        row = w_tilde.sum(dim=-1, keepdim=True)
        p = w_tilde / row.clamp_min(self.eps)
        rel = torch.bmm(p, x)
        return torch.where(row < self.eps, x, rel)

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
        # attached post-build by the model. It runs BEFORE the features/backbone_features
        # split, so the grouper input and the reconstruction target both derive from the
        # smoothed tensor. `feature_smoothing_mix` follows the model's schedule
        # (0 = fully smoothed, 1 = raw); only active in train() mode, so validation and
        # eval always see raw features.
        self.feature_smoothing: Optional[nn.Module] = None
        self.feature_smoothing_mix: float = 1.0
        # v35 modes curriculum: attached post-build like FeatureSmoothing. Cover is
        # train()-only; mix >= 1 or bandwidth <= h_min is identity (eval stays raw).
        self.feature_modes: Optional[nn.Module] = None
        self.feature_modes_mix: float = 1.0
        # v36 Ncut leveling: attached post-build. Mix 0 = fully leveled, 1 = raw.
        # `feature_curriculum_apply == "key"` keeps the reconstruction target (and the
        # Value path) on the original tokens and only feeds the bind tensor to Keys.
        self.feature_ncut: Optional[nn.Module] = None
        self.feature_ncut_mix: float = 1.0
        self.feature_curriculum_apply: str = "tokens"

    def _apply_feature_curriculum_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Train-only token cover / smooth. Eval and mix=1 skip this."""
        if not self.training:
            return tokens
        if tokens.ndim != 3:
            return tokens
        if self.feature_modes is not None and self.feature_modes_mix < 1.0:
            return self.feature_modes(tokens, self.feature_modes_mix)
        if self.feature_smoothing is not None and self.feature_smoothing_mix < 1.0:
            return self.feature_smoothing(tokens, self.feature_smoothing_mix)
        return tokens

    def _curriculum_target_and_key(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(tokens used as recon target / Value, tokens used as Key).

        v33/v34/v35 (`apply=tokens`): both sides are the curriculumed tensor.
        v36 (`apply=key`): target/Value stay raw; Key gets Ncut-leveled mix.
        Eval and mix=1 leave both as the original tokens.
        """
        if tokens.ndim != 3:
            return tokens, tokens
        key_only = str(self.feature_curriculum_apply).lower() == "key"
        if (
            self.training
            and self.feature_ncut is not None
            and self.feature_ncut_mix < 1.0
        ):
            bind = self.feature_ncut(tokens, self.feature_ncut_mix)
            if key_only:
                return tokens, bind
            return bind, bind
        used = self._apply_feature_curriculum_tokens(tokens)
        return used, used

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
