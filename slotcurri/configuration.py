import pathlib
from dataclasses import MISSING, dataclass, field
from functools import reduce
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

ModuleConfig = Dict[str, Any]


@dataclass
class ModelConfig:
    initializer: ModuleConfig
    encoder: ModuleConfig
    grouper: ModuleConfig
    decoder: ModuleConfig
    predictor: Optional[ModuleConfig] = None
    target_encoder: Optional[ModuleConfig] = None
    latent_processor: Optional[ModuleConfig] = None
    mask_resizers: Optional[Dict[str, ModuleConfig]] = None
    losses: Optional[Dict[str, ModuleConfig]] = None
    loss_weights: Optional[Dict[str, float]] = None
    input_type: str = "image"
    target_type: str = "features"
    target_encoder_input: Optional[str] = None
    visualize: bool = True
    eval_mode_config: Optional[Dict[str, Any]] = None
    visualize_every_n_steps: Optional[int] = 25000
    max_steps: int = 100000
    noise_scale: float = 0.1
    # Run the extra backward sweep over each clip at validation/eval time. Training is
    # always single-pass; this only affects inference. bool = legacy on/off (backward
    # sweep anchored at the last frame); the strings "evidence" / "evidence_mass" /
    # "evidence_sum" / "random" select the anchored EABI variants (see ScanOverTime).
    cyclic_inference: Any = True
    # Reconstruction-guided slot expansion (start with 2 slots, split into the full budget
    # at max_steps//10 and max_steps//4). Only applies when attn_mass_curriculum is off;
    # set False to train with the full slot budget from step 0 (SlotContrast baseline).
    slot_expansion: bool = True
    attn_mass_curriculum: Optional[Dict[str, Any]] = None
    # Direction-only supervision on the predictor's residual step. Requires the predictor to
    # be built with vel_dim, since without the velocity input there is little for it to fit.
    predictor_dynamics: Optional[Dict[str, Any]] = None
    # Predictive feature reconstruction (v27): decode Pred(x_t) and reconstruct F_{t+1}.
    # Dense, feature-space supervision of motion; the only training signal that penalizes
    # holding two independently-moving instances in one slot.
    pred_recon: Optional[Dict[str, Any]] = None
    # Counterfactual slot-utility rent: drop one gated slot (v27) or algebraic
    # all-slot mix_{-s}=(mix-m_s recon_s)/(1-m_s) with no extra decode (v42), per frame.
    # Charges z if reconstruction on that slot's territory barely degrades.
    # v44: drop Δ on ungated softmax(α) mix (z-free); add: insert restores
    # a suppressed slot to its α-share (ŷ^{+s}=σ R+(1-σ)ŷ^{g,-s}); psi_weight.
    # v45 teacher='ce': L = CE(softmax(([Δ↓]_+ + [Δ↑]_+)/ce_tau), z); no ψ.
    slot_utility: Optional[Dict[str, Any]] = None
    # Feature curriculum (v33/v36): anneal backbone tokens from a coarse relational
    # representation to raw patch features. v33 affinity-smooths the tokens themselves
    # (grouper + recon target). v36 (anneal=ncut, apply=key) levels Keys only via a
    # 2-way Ncut barrier. v37/v38 keep that Key-only mix/schedule with barrier=false
    # (global ReLU-cosine P, no region cut). anneal=n8 is the same mix / Key-only
    # split on the 8-neighbor ReLU-cosine graph (gate R, not dense W).
    # anneal=half: X^rel = 0.5 P_dense X + 0.5 P_n8 X (token average).
    # anneal=rowcenter: W_ij = ReLU(cos_ij - (τ_i+τ_j)/2), P = row-normalize(W).
    # Train-time only; eval sees raw features.
    feature_curriculum: Optional[Dict[str, Any]] = None
    # v40/v41: coupled slot-confidence entropy (early, concentrate) and a
    # two-mode impurity (late, split merged communities). Train-only aux.
    # impurity_kind: g_s (v40 λ2/λ1 on diag(a)S diag(a) n8), induced_n8
    # (v39ind exp(-(λ1-λ2)/τ) of S on thresholded support), or c_s (v41
    # λ2/λ1 of the v37 Gram C_s = Z^T diag(a^2) Z; optional proj_dim;
    # top-2 via the same 2D subspace as the v37 gate, not d×d eigh).
    # target: c (v40 ownership) or z (v41 usage-head gate).
    # w_ent/w_imp: optional endpoints (default v40: w_ent w0→0, w_imp 0→w0).
    # v41 uses w_ent 0.3→0.1 and w_imp 0.1→0.3 so neither term dies.
    # entropy_normalize: True = H/log K (v40/v41); False = raw H nats (v42).
    slot_ent_impurity: Optional[Dict[str, Any]] = None
    # Learned slot-usage head from sg(A). Applied as the purity_weight gate
    # when conf_kind=usage. normalize: sigmoid (v41), softmax (v42), or
    # sparsemax (v43, exact zeros).
    usage_head: Optional[Dict[str, Any]] = None
    # v43: reconstruction-based usage redistribution. Train the usage head
    # on simplex-projected ∇J, J = E(p) + (λ/2)(1-||p||^2). Weight 0 disables.
    slot_usage_redistribute: Optional[Dict[str, Any]] = None
    masks_to_visualize: Optional[List[str]] = None
    load_weights: Optional[str] = None
    modules_to_load: Optional[Dict[str, str]] = None
    experiment_name: Optional[str] = "default_experiment"
    experiment_group: Optional[str] = "default_group"


@dataclass
class Config:
    optimizer: ModuleConfig = MISSING
    model: ModelConfig = MISSING
    dataset: ModuleConfig = MISSING
    trainer: Optional[ModuleConfig] = field(default_factory=lambda: {})
    train_metrics: Optional[Dict[str, ModuleConfig]] = None
    val_metrics: Optional[Dict[str, ModuleConfig]] = None

    globals: Optional[Dict[str, Any]] = None
    experiment_name: Optional[str] = None
    experiment_group: Optional[str] = None
    seed: Optional[int] = None
    checkpoint_every_n_steps: int = 1000


def load_config(path: pathlib.Path, overrides: Optional[List[str]] = None) -> OmegaConf:
    schema = OmegaConf.structured(Config)
    config = OmegaConf.load(path)

    if overrides is not None:
        if isinstance(overrides, list):

            overrides = OmegaConf.from_dotlist(overrides)
        elif isinstance(overrides, dict):
            overrides = OmegaConf.create(overrides)
        else:
            ValueError("overrides should be dotlist or dict")
        config = OmegaConf.merge(schema, config, overrides)
    else:
        config = OmegaConf.merge(schema, config)

    return config


def override_config(
    config: Optional[pathlib.Path] = None,
    override_config_path: Optional[pathlib.Path] = None,
    additional_overrides: Optional[List[str]] = None,
) -> OmegaConf:
    schema = OmegaConf.structured(Config)
    config_objects = [schema, config]
    if override_config_path is not None:
        override_config = OmegaConf.load(override_config_path)
        config_objects.append(override_config)

    if additional_overrides is not None:
        if isinstance(additional_overrides, list):

            additional_overrides = OmegaConf.from_dotlist(additional_overrides)
        elif isinstance(additional_overrides, dict):
            additional_overrides = OmegaConf.create(additional_overrides)
        else:
            ValueError("overrides should be dotlist or dict")
        config_objects.append(additional_overrides)

    config = OmegaConf.merge(*config_objects)
    return config


def save_config(path: pathlib.Path, config: OmegaConf):
    OmegaConf.save(config, path, resolve=True)


def resolver_eval(fn: str, *args):
    params, _, body = fn.partition(":")
    if body == "":
        body = params
        params = ""

    if len(params) == 0:
        arg_names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
        assert len(args) <= len(arg_names), f"Only up to {len(arg_names)} arguments are supported"
        params = ",".join(arg_names[: len(args)])

    if not params.startswith("lambda "):
        params = "lambda " + params

    return eval(f"{params}: {body}")(*args)


OmegaConf.register_new_resolver("eval", resolver_eval)
OmegaConf.register_new_resolver("add", lambda *args: sum(args))
OmegaConf.register_new_resolver("sub", lambda a, b: a - b)
OmegaConf.register_new_resolver("mul", lambda *args: reduce(lambda prod, cur: prod * cur, args, 1))
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("min", lambda *args: min(*args))
OmegaConf.register_new_resolver("max", lambda *args: max(*args))
# should be useful in case of dependencies between config params
OmegaConf.register_new_resolver(
    "config_prop", lambda prop, *keys: get_predefined_property(prop, keys)
)

VIT_PARAMS = {
    "vit_small_patch16_224_dino": {"FEAT_DIM": 384, "NUM_PATCHES": 196},
    "vit_small_patch8_224_dino": {"FEAT_DIM": 384, "NUM_PATCHES": 784},
    "vit_base_patch16_224_dino": {"FEAT_DIM": 768, "NUM_PATCHES": 196},
    "vit_base_patch16_448_dino": {"FEAT_DIM": 768, "NUM_PATCHES": 784},
    "vit_base_patch8_224_dino": {"FEAT_DIM": 768, "NUM_PATCHES": 784},
    "vit_base_patch16_224_mae": {"FEAT_DIM": 768, "NUM_PATCHES": 196},
    "vit_base_patch16_224_mocov3": {"FEAT_DIM": 768, "NUM_PATCHES": 196},
    "vit_base_patch16_224_msn": {"FEAT_DIM": 768, "NUM_PATCHES": 196},
    "vit_base_patch14_dinov2": {"FEAT_DIM": 768, "NUM_PATCHES": 256},
    "vit_small_patch14_dinov2": {"FEAT_DIM": 384, "NUM_PATCHES": 256},
    "vit_large_patch14_dinov2": {"FEAT_DIM": 1024, "NUM_PATCHES": 256},
}


def get_predefined_property(prop, keys):
    value = globals()[prop]
    for key in keys:
        if callable(value):
            value = value(key)
        elif isinstance(value, dict):
            value = value[key]
        else:
            raise ValueError(f"Can not handle type {type(value)}")
    return value
