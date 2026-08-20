from typing import Any, Dict, List, Optional, Union

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

        P = softmax_j( cos(x_i, x_j) / tau ),   x_tilde = P x

    averages every token with its feature-space neighbours. Because DINO affinities are
    block-structured by object, this shrinks WITHIN-object feature variance toward zero
    while approximately preserving object boundaries (cross-object affinities are small)
    -- unlike a spatial blur, which mixes across boundaries. The returned tensor is the
    scheduled blend

        x_used = (1 - mix) * x_tilde + mix * x

    with mix = 0 fully smoothed (object-level features, a part-split earns nothing and
    cannot hold clean ownership) and mix = 1 raw (exact no-op, the module returns x
    itself). `window` optionally restricts the affinity to a local square neighbourhood
    (Chebyshev radius on the patch grid) so that distant same-appearance regions -- two
    instances of one class -- do not exchange features.

    P is computed under no_grad and x_tilde is detached: the smoothing is a pure data
    transform of the (frozen) backbone tokens, introducing no new gradient paths. The
    module holds no parameters, so checkpoints stay byte-compatible either way.
    """

    def __init__(
        self,
        tau: float = 0.1,
        window: Optional[int] = None,
        chunk_size: int = 16,
    ):
        super().__init__()
        self.tau = float(tau)
        self.window = int(window) if window is not None else None
        self.chunk_size = max(int(chunk_size), 1)
        self._mask_cache: Dict[Any, Optional[torch.Tensor]] = {}

    def _window_mask(self, n_tokens: int, device: torch.device) -> Optional[torch.Tensor]:
        """True where the affinity is masked out (grid Chebyshev distance > window)."""
        if self.window is None:
            return None
        key = (n_tokens, str(device))
        if key in self._mask_cache:
            return self._mask_cache[key]
        side = int(round(n_tokens**0.5))
        if side * side != n_tokens:
            # non-square token grid: cannot localize, fall back to global affinity
            self._mask_cache[key] = None
            return None
        idx = torch.arange(n_tokens, device=device)
        ys, xs = idx // side, idx % side
        cheb = torch.maximum(
            (ys[:, None] - ys[None, :]).abs(), (xs[:, None] - xs[None, :]).abs()
        )
        mask = cheb > self.window
        self._mask_cache[key] = mask
        return mask

    @torch.no_grad()
    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, mix: float) -> torch.Tensor:
        mix = float(mix)
        if mix >= 1.0:
            return x
        smoothed = self._smooth(x.detach())
        if mix <= 0.0:
            return smoothed
        return (1.0 - mix) * smoothed + mix * x


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

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # images: batch x n_channels x height x width
        backbone_features = self.backbone(images)
        if (
            self.feature_smoothing is not None
            and self.training
            and self.feature_smoothing_mix < 1.0
        ):
            if isinstance(backbone_features, dict):
                main = backbone_features[self.main_features_key]
                if main.ndim == 3:
                    backbone_features[self.main_features_key] = self.feature_smoothing(
                        main, self.feature_smoothing_mix
                    )
            elif backbone_features.ndim == 3:
                backbone_features = self.feature_smoothing(
                    backbone_features, self.feature_smoothing_mix
                )
        if isinstance(backbone_features, dict):
            features = backbone_features[self.main_features_key].clone()
        else:
            features = backbone_features.clone()

        if self.pos_embed:
            features = self.pos_embed(features)

        if self.spatial_flatten:
            features = einops.rearrange(features, "b c h w -> b (h w) c")
        if self.output_transform:
            features = self.output_transform(features)

        assert (
            features.ndim == 3
        ), f"Expect output shape (batch, tokens, dims), but got {features.shape}"
        if isinstance(backbone_features, dict):
            for k, backbone_feature in backbone_features.items():
                if self.spatial_flatten:
                    backbone_features[k] = einops.rearrange(backbone_feature, "b c h w -> b (h w) c")
                assert (
                    backbone_feature.ndim == 3
                ), f"Expect output shape (batch, tokens, dims), but got {backbone_feature.shape}"
            main_backbone_features = backbone_features[self.main_features_key]

            return {
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

            return {
                "features": features,
                "backbone_features": backbone_features,
            }


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
