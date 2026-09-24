import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torchvision.utils import make_grid

from slotcurri import configuration, losses, modules, optimizers, utils, visualizations
from slotcurri.data.transforms import Denormalize
from slotcurri.modules.gate_p_residual import GatePResidualNet
from slotcurri.modules.decoders import algebraic_slot_utility_loss
from slotcurri.modules.usage_redistribute import usage_redistribute_direction
from slotcurri.modules.video import (
    SlotUsageHead,
    ownership_confidence,
    slot_confidence_entropy,
    spectral_cs_impurity,
    spectral_graph_n8_impurity,
    spectral_graph_n8_induced_impurity,
)
import torch.nn.functional as F
import os
import re
import shutil

import matplotlib.pyplot as plt


def feature_curriculum_mix(step: int, anneal_steps: int, schedule: str = "cosine") -> float:
    """Blend factor for the v33 feature curriculum.

    0 = fully affinity-smoothed (object-level) backbone tokens, 1 = raw patch tokens.
    Monotone in step, reaches 1 at `anneal_steps` and stays there; after the anneal the
    encoder is byte-identical to the uncurriculumed model. Module-level so the schedule
    is unit-testable without building a model.
    """
    t = min(max(float(step), 0.0) / float(max(int(anneal_steps), 1)), 1.0)
    if str(schedule).lower() == "linear":
        return t
    return 0.5 * (1.0 - math.cos(math.pi * t))


def feature_curriculum_window(
    step: int,
    anneal_steps: int,
    window_start: int,
    schedule: str = "cosine",
) -> int:
    """Chebyshev radius for the window-anneal curriculum (v34).

    w(t) = round(w0 * (1 - s(t))) with the same s as `feature_curriculum_mix`.
    Returns 0 at/after `anneal_steps` (smoothing off / raw). Half-up rounding so
    the midpoint is not banker's-rounded to even.
    """
    s = feature_curriculum_mix(step, anneal_steps, schedule)
    if s >= 1.0:
        return 0
    w0 = max(int(window_start), 0)
    return max(0, int(math.floor(float(w0) * (1.0 - s) + 0.5)))


def feature_curriculum_bandwidth(
    step: int,
    anneal_steps: int,
    h_start: float,
    schedule: str = "cosine",
    h_min: float = 0.02,
) -> float:
    """Mean-shift bandwidth for the modes curriculum (v35).

    h(t) = h0 * (1 - s(t)) with the same s as `feature_curriculum_mix`.
    Returns 0 at/after `anneal_steps`, and also when h would fall to `h_min`
    (treated as raw / cover off).
    """
    s = feature_curriculum_mix(step, anneal_steps, schedule)
    if s >= 1.0:
        return 0.0
    h = float(h_start) * (1.0 - s)
    if h <= float(h_min):
        return 0.0
    return h


def build(
    model_config: configuration.ModelConfig,
    optimizer_config,
    train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
    val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
):
    optimizer_builder = optimizers.OptimizerBuilder(**optimizer_config)

    initializer = modules.build_initializer(model_config.initializer)
    encoder = modules.build_encoder(model_config.encoder, "FrameEncoder")
    # all_grouper = modules.build_grouper(model_config.grouper)
    grouper = modules.build_grouper(model_config.grouper)
    decoder = modules.build_decoder(model_config.decoder)

    target_encoder = None
    if model_config.target_encoder:
        target_encoder = modules.build_encoder(model_config.target_encoder, "FrameEncoder")
        assert (
            model_config.target_encoder_input is not None
        ), "Please specify `target_encoder_input`."

    uh_cfg = model_config.get("usage_head")
    usage_head = None
    if uh_cfg is not None and bool(uh_cfg.get("enabled", False)):
        n_patches = uh_cfg.get("n_patches")
        if n_patches is None:
            raise ValueError("usage_head.enabled requires n_patches")
        usage_head = SlotUsageHead(
            n_patches=int(n_patches),
            mlp_hidden=int(uh_cfg.get("mlp_hidden", 256)),
            d_model=int(uh_cfg.get("d_model", 64)),
            n_blocks=int(uh_cfg.get("n_blocks", 1)),
            n_heads=int(uh_cfg.get("n_heads", 4)),
            stopgrad_attn=bool(uh_cfg.get("stopgrad_attn", True)),
            dropout=float(uh_cfg.get("dropout", 0.0)),
            normalize=str(uh_cfg.get("normalize", "sigmoid")),
        )

    input_type = model_config.get("input_type", "image")
    if input_type == "image":
        processor = modules.LatentProcessor(
            grouper, predictor=None, usage_head=usage_head
        )
    elif input_type == "video": # default input type
        encoder = modules.MapOverTime(encoder)
        decoder = modules.MapOverTime(decoder)
        if target_encoder: # not in use
            target_encoder = modules.MapOverTime(target_encoder)
        if model_config.predictor is not None: # TransformerEncoder
            predictor = modules.build_module(model_config.predictor)
        else:
            predictor = None
        if model_config.latent_processor:
            processor = modules.build_video(
                model_config.latent_processor,
                "LatentProcessor",
                corrector=grouper,
                predictor=predictor,
                usage_head=usage_head,
            )
        else:
            processor = modules.LatentProcessor(
                grouper, predictor, usage_head=usage_head
            )
        processor = modules.ScanOverTime(processor)
    else:
        raise ValueError(f"Unknown input type {input_type}")

    target_type = model_config.get("target_type", "features")
    if target_type == "input":
        default_target_key = input_type
    elif target_type == "features":
        if model_config.target_encoder_input is not None:
            default_target_key = "target_encoder.backbone_features"
        else:
            default_target_key = "encoder.backbone_features"
    else:
        raise ValueError(f"Unknown target type {target_type}. Should be `input` or `features`.")

    loss_defaults = {
        "pred_key": "decoder.reconstruction",
        "target_key": default_target_key,
        "video_inputs": input_type == "video",
        "patch_inputs": target_type == "features",
    }
    if model_config.losses is None:
        loss_fns = {"mse": losses.build(dict(**loss_defaults, name="MSELoss"))}
    else:
        loss_fns = {
            name: losses.build({**loss_defaults, **loss_config})
            for name, loss_config in model_config.losses.items()
        }

    if model_config.mask_resizers:
        mask_resizers = {
            name: modules.build_utils(resizer_config, "Resizer")
            for name, resizer_config in model_config.mask_resizers.items()
        }
    else:
        mask_resizers = {
            "decoder": modules.build_utils(
                {
                    "name": "Resizer",
                    # When using features as targets, assume patch-shaped outputs. With other
                    # targets, assume spatial outputs.
                    "patch_inputs": target_type == "features",
                    "video_inputs": input_type == "video",
                    "resize_mode": "bilinear",
                }
            ),
            "grouping": modules.build_utils(
                {
                    "name": "Resizer",
                    "patch_inputs": True,
                    "video_inputs": input_type == "video",
                    "resize_mode": "bilinear",
                }
            ),
        }

    if model_config.masks_to_visualize:
        masks_to_visualize = model_config.masks_to_visualize
    else:
        masks_to_visualize = ["decoder", "grouping"]

    model = ObjectCentricModel(
        optimizer_builder,
        initializer,
        encoder,
        processor,
        decoder,
        loss_fns,
        noise_scale=model_config.get("noise_scale", 0.1),
        loss_weights=model_config.get("loss_weights", None),
        target_encoder=target_encoder,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        mask_resizers=mask_resizers,
        input_type=input_type,
        target_encoder_input=model_config.get("target_encoder_input", None),
        visualize=model_config.get("visualize", True),
        visualize_every_n_steps=model_config.get("visualize_every_n_steps", 25000),
        masks_to_visualize=masks_to_visualize,
        max_steps=model_config.get("max_steps", 100000),
        experiment_name=model_config.get("experiment_name", "default_experiment"),
        experiment_group=model_config.get("experiment_group", "default_group"),
        attn_mass_curriculum=model_config.get("attn_mass_curriculum", None),
        predictor_dynamics=model_config.get("predictor_dynamics", None),
        pred_recon=model_config.get("pred_recon", None),
        slot_utility=model_config.get("slot_utility", None),
        cyclic_inference=model_config.get("cyclic_inference", True),
        slot_expansion=model_config.get("slot_expansion", True),
        feature_curriculum=model_config.get("feature_curriculum", None),
        slot_ent_impurity=model_config.get("slot_ent_impurity", None),
        usage_head_config=model_config.get("usage_head", None),
        slot_usage_redistribute=model_config.get("slot_usage_redistribute", None),
    )

    if model_config.load_weights:
        model.load_weights_from_checkpoint(model_config.load_weights, model_config.modules_to_load)

    return model


class ObjectCentricModel(pl.LightningModule):
    def __init__(
        self,
        optimizer_builder: Callable,
        initializer: nn.Module,
        encoder: nn.Module,
        # all_grouper: nn.Module,
        processor: nn.Module,
        decoder: nn.Module,
        loss_fns: Dict[str, losses.Loss],
        noise_scale: float = 0.1,
        loss_weights: Optional[Dict[str, float]] = None,
        target_encoder: Optional[nn.Module] = None,
        train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        mask_resizers: Optional[Dict[str, modules.Resizer]] = None,
        input_type: str = "image",
        target_encoder_input: Optional[str] = None,
        visualize: bool = True,
        visualize_every_n_steps: Optional[int] = None,
        masks_to_visualize: Union[str, List[str]] = ["decoder", "grouping"],
        max_steps: int = 100000,
        experiment_name: str = "default_experiment",
        experiment_group: str = "default_group",
        attn_mass_curriculum: Optional[Dict[str, Any]] = None,
        predictor_dynamics: Optional[Dict[str, Any]] = None,
        pred_recon: Optional[Dict[str, Any]] = None,
        slot_utility: Optional[Dict[str, Any]] = None,
        cyclic_inference: Union[bool, str] = True,
        slot_expansion: bool = True,
        feature_curriculum: Optional[Dict[str, Any]] = None,
        slot_ent_impurity: Optional[Dict[str, Any]] = None,
        usage_head_config: Optional[Dict[str, Any]] = None,
        slot_usage_redistribute: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        # bool: legacy on/off (backward sweep anchored at the last frame). A string
        # selects the anchored re-inference variants in ScanOverTime ("evidence" = EABI
        # with g*c, "evidence_sum" = EABI with sum_s g, "random" = its control); eval
        # scripts set this directly on a loaded model.
        self.cyclic_inference = (
            cyclic_inference if isinstance(cyclic_inference, str) else bool(cyclic_inference)
        )
        self.slot_expansion = bool(slot_expansion)
        self.experiment_name = experiment_name
        self.experiment_group = experiment_group
        self.optimizer_builder = optimizer_builder
        self.initializer = initializer
        self.encoder = encoder
        # self.all_grouper = all_grouper
        self.processor = processor

        self.decoder = decoder
        self.target_encoder = target_encoder
        self.max_steps = max_steps
        if loss_weights is not None:
            # Filter out losses that are not used
            assert (
                loss_weights.keys() == loss_fns.keys()
            ), f"Loss weight keys {loss_weights.keys()} != {loss_fns.keys()}"
            loss_fns_filtered = {k: loss for k, loss in loss_fns.items() if loss_weights[k] != 0.0}
            loss_weights_filtered = {
                k: loss for k, loss in loss_weights.items() if loss_weights[k] != 0.0
            }
            self.loss_fns = nn.ModuleDict(loss_fns_filtered)
            self.loss_weights = loss_weights_filtered
        else:
            self.loss_fns = nn.ModuleDict(loss_fns)
            self.loss_weights = {}
        # v39lam1gu: second featrec on the ungated softmax mix. Decoder
        # computes both mixes in one MLP pass when this loss is present.
        self.featrec_ungated = "loss_featrec_ungated" in self.loss_fns

        self.mask_resizers = mask_resizers if mask_resizers else {}
        self.mask_resizers["segmentation"] = modules.Resizer(
            video_inputs=input_type == "video", resize_mode="nearest-exact"
        )
        self.mask_soft_to_hard = modules.SoftToHardMask()
        self.train_metrics = torch.nn.ModuleDict(train_metrics)
        self.val_metrics = torch.nn.ModuleDict(val_metrics)

        self.visualize = visualize
        if visualize:
            assert visualize_every_n_steps is not None
        self.visualize_every_n_steps = visualize_every_n_steps
        if isinstance(masks_to_visualize, str):
            masks_to_visualize = [masks_to_visualize]
        for key in masks_to_visualize:
            if key not in ("decoder", "grouping"):
                raise ValueError(f"Unknown mask type {key}. Should be `decoder` or `grouping`.")
        self.mask_keys_to_visualize = [f"{key}_masks" for key in masks_to_visualize]

        if input_type == "image":
            self.input_key = "image"
            self.expected_input_dims = 4
        elif input_type == "video":
            self.input_key = "video"
            self.expected_input_dims = 5
        else:
            raise ValueError(f"Unknown input type {input_type}. Should be `image` or `video`.")

        self.target_encoder_input_key = (
            target_encoder_input if target_encoder_input else self.input_key
        )
        self.n_slots = self.initializer.n_slots

        self.hier_steps = [self.max_steps // 10, self.max_steps // 4]
        self.hier_n_slots = [2, self.n_slots // 2, self.n_slots]

        self.slot_loss_scale = torch.zeros(self.n_slots)#.to(decoder.device)
        self.slot_loss_scale.requires_grad_(False)
        # self.noise_scale = self.initializer.initial_std * noise_scale

        # --- direction-only supervision for the predictor ---
        # Nothing else in the loss tree reads `state_predicted`; it is only consumed as the
        # next frame's prior. This term is what actually asks the predictor to model motion,
        # and the velocity input configured on the predictor is its precondition.
        pdyn = predictor_dynamics or {}
        self.dyn_weight = float(pdyn.get("weight", 0.0))
        # Direction is meaningless for a slot that barely moved, so drop the low-displacement
        # tail; the threshold is a quantile of the observed ||d|| rather than an absolute.
        self.dyn_min_disp_q = float(pdyn.get("min_disp_quantile", 0.5))
        # A slot whose gate jumps between frames moves because it started or stopped
        # accepting the observation, which is not motion. Those pairs are excluded.
        self.dyn_gate_jump_max = float(pdyn.get("gate_jump_max", 0.3))
        self.dyn_start_step = int(pdyn.get("start_step", 0))

        # --- predictive feature reconstruction (v27) ---
        # L_pred = MSE(Dec(Pred(x_t)), F_{t+1}): the slots at t, pushed through the
        # predictor, must reconstruct the NEXT frame's (frozen) features. Static featrec
        # has ~zero gradient against holding two same-appearance instances in one slot;
        # predicting their two independent motions with one latent leaves an irreducible
        # error, so this is the term that pays for splitting them. Unlike v23's loss_dyn
        # (cosine on slot latents, failed) this supervises through the decoder in feature
        # space -- it prescribes what the prediction must explain, not how latents move.
        pr = pred_recon or {}
        self.pred_weight = float(pr.get("weight", 0.0))
        # Number of random (t -> t+1) transitions decoded per step. The extra decoder pass
        # needs activations for backward, so this bounds the memory overhead (1 of T-1
        # transitions ~= +1/T decoder memory instead of doubling it).
        self.pred_n_transitions = int(pr.get("n_transitions", 1))
        # "lambda" scales the weight by the coupled-curriculum coefficient lambda_t, so the
        # term is silent during the coarse phase (untrained predictor, intentional merging)
        # and fully on for the consolidation stretch. "none" applies the raw weight.
        self.pred_ramp = str(pr.get("ramp", "lambda")).lower()
        self.pred_start_step = int(pr.get("start_step", 0))
        if self.pred_ramp not in ("lambda", "none"):
            raise ValueError(f"pred_recon.ramp must be 'lambda' or 'none', got {self.pred_ramp!r}")

        # --- counterfactual slot-utility rent (v27) ---
        # Re-decode with one randomly chosen gated slot dropped (no_grad) and measure the
        # per-patch error increase on that slot's own territory, relative to the sample's
        # mean error: rel_delta = E_mask[err_drop - err_full] / E[err_full]. Rent
        # relu(1 - rel_delta/margin) is charged against the slot's (live) gate. A duplicate
        # slot's territory is covered by its twin after the decoder renorm (rel_delta ~ 0,
        # full rent); a slot holding unique content is irreplaceable (rel_delta >> margin,
        # exempt). This is the marginal-utility fix for gate_l1's documented failure: the
        # constant rent taxed every slot equally and evicted small-object slots first
        # (v2: active fell to ~1.7); pricing by counterfactual utility exempts them by
        # construction. Delta is detached; the gradient reaches only the gate.
        su = slot_utility or {}
        self.util_weight = float(su.get("weight", 0.0))
        # rel_delta below which rent starts (1.0 = dropping the slot must raise error on
        # its territory by at least the sample-mean per-patch error to live rent-free)
        self.util_margin = float(su.get("margin", 1.0))
        # candidates: slots the gate actually admits and with enough decoder-mask territory
        # for delta to be measurable. Ignored by mode='algebraic' (all slots).
        self.util_gate_min = float(su.get("gate_min", 0.5))
        self.util_min_mask = float(su.get("min_mask", 0.01))
        self.util_ramp = str(su.get("ramp", "lambda")).lower()
        self.util_start_step = int(su.get("start_step", 0))
        self.util_mode = str(su.get("mode", "drop")).lower()
        if self.util_ramp not in ("lambda", "none"):
            raise ValueError(f"slot_utility.ramp must be 'lambda' or 'none', got {self.util_ramp!r}")
        if self.util_mode not in ("drop", "algebraic"):
            raise ValueError(
                f"slot_utility.mode must be 'drop' or 'algebraic', got {self.util_mode!r}"
            )
        # v44: drop Δ on ungated softmax(α) mix (z-free). Add restores a
        # suppressed slot to its α-share: ŷ^{+s}=σ_s R_s+(1-σ_s)ŷ^{g,-s}
        # on patches with σ_s > m_s. add: insert | off.
        # v45 teacher='ce': L = CE(softmax(([Δ↓]_+ + [Δ↑]_+)/τ), z) instead of ⟨z,c⟩.
        self.util_add_insert = self._util_add_insert_flag(su)
        if "add_eps" in su or "add_below" in su:
            raise ValueError(
                "slot_utility.add_eps / add_below were removed; use add: insert"
            )
        teacher_raw = str(su.get("teacher", "rent")).strip().lower()
        if teacher_raw in ("inner", "dot"):
            teacher_raw = "rent"
        if teacher_raw not in ("rent", "ce"):
            raise ValueError(
                f"slot_utility.teacher must be 'rent' or 'ce', got {su.get('teacher')!r}"
            )
        self.util_teacher = teacher_raw
        self.util_ce_tau = float(su.get("ce_tau", su.get("tau", 0.5)))
        if self.util_teacher == "ce" and self.util_ce_tau <= 0.0:
            raise ValueError(
                f"slot_utility.ce_tau must be > 0, got {self.util_ce_tau}"
            )
        # v44: 0.05 * (1 - ||z||^2). Slot-independent occupancy tax is constant on
        # the simplex; this is the convex term that penalizes the uniform vertex.
        # v45 leaves this at 0: the CE target already pulls z to uniform when Δ=0.
        self.util_psi_weight = float(su.get("psi_weight", 0.0))
        self._util_rel_add = None
        self._util_c_mean = None
        self._util_n_suppressed = None
        self._util_pi_max = None
        self._util_pi_entropy = None
        if self.util_psi_weight < 0.0:
            raise ValueError(
                f"slot_utility.psi_weight must be >= 0, got {self.util_psi_weight}"
            )

        # --- feature curriculum (v33): task-level coarse-to-fine ---
        # The backbone tokens are annealed from affinity-smoothed (object-level: within-
        # object feature variance removed by n_steps of non-parametric self-attention,
        # see modules.FeatureSmoothing) to raw patch features over `anneal_steps`. Early
        # on a part-split of one object earns no reconstruction advantage (no variance
        # left to divide) and cannot hold clean ownership (identical features give every
        # competing slot the same q.k on the whole blob), which converts the purity
        # gate's one blind spot -- clean part-splits -- into the mixed-ownership states
        # it already suppresses. This is a schedule on the TASK, not on the gate: a
        # mistimed clock degrades smoothly instead of gating real objects away (the v24
        # failure class). Validation/eval always run on raw features (train()-only), so
        # after the anneal the model is byte-identical to the uncurriculumed one.
        fc = feature_curriculum or {}
        self.featcur_enabled = bool(fc.get("enabled", False))
        self.featcur_anneal_steps = int(fc.get("anneal_steps", 30000))
        self.featcur_schedule = str(fc.get("schedule", "cosine")).lower()
        if self.featcur_schedule not in ("cosine", "linear"):
            raise ValueError(
                "feature_curriculum.schedule must be 'cosine' or 'linear', "
                f"got {self.featcur_schedule!r}"
            )
        if self.featcur_enabled and self.featcur_anneal_steps <= 0:
            raise ValueError("feature_curriculum.anneal_steps must be positive")
        self.featcur_anneal = str(fc.get("anneal", "mix")).lower()
        if self.featcur_anneal not in (
            "mix",
            "window",
            "modes",
            "ncut",
            "n8",
            "half",
            "rowcenter",
        ):
            raise ValueError(
                "feature_curriculum.anneal must be 'mix', 'window', 'modes', "
                f"'ncut', 'n8', 'half' or 'rowcenter', got {self.featcur_anneal!r}"
            )
        # window-anneal (v34): the clock is the Chebyshev radius, not the raw blend.
        # w0 from window_start, else the static `window` field. 0 is illegal (that
        # would be "always raw"); global affinity (window is null) cannot shrink.
        w_start = fc.get("window_start", fc.get("window", None))
        self.featcur_window_start = 0 if w_start is None else int(w_start)
        if self.featcur_enabled and self.featcur_anneal == "window":
            if self.featcur_window_start <= 0:
                raise ValueError(
                    "feature_curriculum.anneal='window' requires window_start "
                    f"(or window) > 0, got {w_start!r}"
                )
        # modes-anneal (v35): the clock is mean-shift bandwidth, not K and not w.
        # h(t) = h0 * (1-s); h<=h_min is raw. Tokens are covered by density-mode
        # means so #{k} = C(scene, h) and occupancy tracks C, not leftover seats.
        self.featcur_h_start = float(fc.get("h_start", fc.get("bandwidth", 0.5)))
        self.featcur_h_min = float(fc.get("h_min", 0.02))
        if self.featcur_enabled and self.featcur_anneal == "modes":
            if self.featcur_h_start <= 0.0:
                raise ValueError(
                    "feature_curriculum.anneal='modes' requires h_start > 0, "
                    f"got {self.featcur_h_start!r}"
                )
        self._featcur_window = None
        self._featcur_h = 0.0
        self._featcur_s = 0.0
        self.featcur_apply = str(
            fc.get(
                "apply",
                "key"
                if self.featcur_anneal in ("ncut", "n8", "half", "rowcenter")
                else "tokens",
            )
        ).lower()
        if self.featcur_apply not in ("tokens", "key"):
            raise ValueError(
                "feature_curriculum.apply must be 'tokens' or 'key', "
                f"got {self.featcur_apply!r}"
            )
        if self.featcur_enabled:
            inner = self._frame_encoder()
            inner.feature_curriculum_apply = self.featcur_apply
            if self.featcur_anneal == "modes":
                inner.feature_modes = modules.FeatureModeCollapse(
                    n_iter=int(fc.get("n_iter", 8)),
                    delta=float(fc.get("delta", 0.02)),
                    h_min=self.featcur_h_min,
                    probe=int(fc.get("probe", 128)),
                    chunk_size=int(fc.get("chunk_size", 16)),
                )
                inner.feature_modes.bandwidth = self.featcur_h_start
                inner.feature_modes_mix = 0.0
            elif self.featcur_anneal in ("ncut", "n8", "half", "rowcenter"):
                if self.featcur_anneal in ("n8", "half", "rowcenter") and bool(
                    fc.get("barrier", False)
                ):
                    raise ValueError(
                        f"feature_curriculum.anneal={self.featcur_anneal!r} "
                        "requires barrier=false (Fiedler/median cut is defined "
                        "on the dense graph)"
                    )
                inner.feature_ncut = modules.NcutRelationalLeveling(
                    chunk_size=int(fc.get("chunk_size", 8)),
                    eps=float(fc.get("eps", 1e-6)),
                    n_iter=int(fc.get("ncut_iters", 16)),
                    barrier=(
                        bool(fc.get("barrier", True))
                        if self.featcur_anneal == "ncut"
                        else False
                    ),
                    n8=self.featcur_anneal == "n8",
                    glob_n8_mix=(
                        float(fc.get("glob_n8_mix", 0.5))
                        if self.featcur_anneal == "half"
                        else 0.0
                    ),
                    row_center=self.featcur_anneal == "rowcenter",
                )
                inner.feature_ncut_mix = 0.0
            else:
                ref = fc.get("window_ref_grid", 37)
                init_window = (
                    self.featcur_window_start
                    if self.featcur_anneal == "window"
                    else fc.get("window", None)
                )
                smoothing = modules.FeatureSmoothing(
                    tau=float(fc.get("tau", 0.1)),
                    window=init_window,
                    n_steps=int(fc.get("n_steps", 1)),
                    chunk_size=int(fc.get("chunk_size", 16)),
                    window_ref_grid=None if ref is None else int(ref),
                )
                inner.feature_smoothing = smoothing
                # step-0: mix-anneal starts fully smoothed; window-anneal too (w=w0).
                inner.feature_smoothing_mix = 0.0

        # --- attention-mass curriculum (dynamic slot gating) ---
        amc = attn_mass_curriculum or {}
        self.attn_mass_enabled = bool(amc.get("enabled", False))
        # The two curricula are mutually exclusive: attention-mass gating owns the slot
        # budget when enabled, so expansion only runs as the standalone SlotCurri schedule.
        self._expansion_active = self.slot_expansion and not self.attn_mass_enabled
        # default (always-active) slots: `n_default` leading slots, or explicit `default_slot_idx`
        n_default = amc.get("n_default", None)
        if n_default is not None:
            self.amc_default_idx = list(range(int(n_default)))
        else:
            di = amc.get("default_slot_idx", 0)
            self.amc_default_idx = [int(di)] if isinstance(di, int) else [int(x) for x in di]
        # thresholds: prefer multipliers of the uniform mass (1/n_slots), else absolute values
        uniform = 1.0 / max(self.n_slots, 1)
        if amc.get("p_start_mult", None) is not None:
            self.amc_p_start_mult = float(amc.get("p_start_mult"))
            self.amc_p_start = self.amc_p_start_mult * uniform
        else:
            self.amc_p_start = float(amc.get("p_start", 0.05))
            self.amc_p_start_mult = self.amc_p_start / uniform
        if amc.get("p_end_mult", None) is not None:
            self.amc_p_end_mult = float(amc.get("p_end_mult"))
            self.amc_p_end = self.amc_p_end_mult * uniform
        else:
            self.amc_p_end = float(amc.get("p_end", 0.005))
            self.amc_p_end_mult = self.amc_p_end / uniform
        # p_mode:
        #   "absolute" (default): p = annealed absolute mass threshold (v10 and earlier)
        #   "median_ema": p = sg(EMA[median(m)]) * annealed p_mult  (scene-adaptive)
        #   "learnable_residual": p = p_sched * (1 + alpha * tanh(f(sg[m])))
        self.amc_p_mode = str(amc.get("p_mode", "absolute")).lower()
        if self.amc_p_mode not in ("absolute", "median_ema", "learnable_residual"):
            raise ValueError(
                f"attn_mass_curriculum.p_mode must be 'absolute', 'median_ema', or "
                f"'learnable_residual', got {self.amc_p_mode!r}"
            )
        self.amc_median_ema_momentum = float(amc.get("median_ema_momentum", 0.9))
        self.amc_p_residual_alpha = float(amc.get("p_residual_alpha", 0.5))
        self.amc_p_residual_l2 = float(amc.get("p_residual_l2", 0.01))
        self.amc_p_residual_warmup_steps = int(amc.get("p_residual_warmup_steps", 25000))
        p_residual_hidden = int(amc.get("p_residual_hidden", 32))
        if self.attn_mass_enabled and self.amc_p_mode == "learnable_residual":
            self.amc_p_residual_net = GatePResidualNet(
                n_slots=self.n_slots, hidden=p_residual_hidden
            )
        else:
            self.amc_p_residual_net = None
        self.amc_anneal_steps = int(amc.get("anneal_steps", self.max_steps // 4))
        # gating mode: "hard" (binary active/dormant) or "soft" (sigmoid gate in [0, 1]).
        # For soft gating, tau anneals from tau_start (soft) -> tau_end (sharp) so the gate
        # gradually converges to the hard decision (soft-to-hard curriculum).
        self.amc_gate_mode = str(amc.get("gate_mode", "hard"))
        if amc.get("gate_tau_start_mult", None) is not None:
            self.amc_tau_start = float(amc.get("gate_tau_start_mult")) * uniform
        else:
            self.amc_tau_start = float(amc.get("gate_tau_start", 0.05))
        if amc.get("gate_tau_end_mult", None) is not None:
            self.amc_tau_end = float(amc.get("gate_tau_end_mult")) * uniform
        else:
            self.amc_tau_end = float(amc.get("gate_tau_end", 0.01))
        # optional L1 sparsity penalty on the (soft/ste) gate. Charges a constant "rent"
        # lambda per active non-default slot, pushing redundant slots' gates toward 0. This
        # restores the anti-over-fragmentation pressure that decoder renormalization cancels
        # for spatially-disjoint (lone-claimant) fragments in pure soft mode. 0 disables it.
        self.amc_gate_l1 = float(amc.get("gate_l1", 0.0))
        # batch-entropy diversity loss (anti-collapse / load-balancing): encourages different
        # slots to be used across the batch, counteracting rich-get-richer collapse to ~1 slot.
        # 0 disables it. Opposite sign to gate_l1 (which prunes slots).
        self.amc_gate_div = float(amc.get("gate_div", 0.0))
        # Error-weighted coverage: push gated attention onto high featrec-residual patches.
        # L = E[ w_f * (1 - c_f) ] with c_f = sum_s g_s A_s,f (or max_s), w_f ∝ detach(error).
        # 0 disables. Keep small; enable mid/late via gate_cov_start_step to protect coarse phase.
        self.amc_gate_cov = float(amc.get("gate_cov", 0.0))
        self.amc_gate_cov_top_frac = float(amc.get("gate_cov_top_frac", 0.1))
        self.amc_gate_cov_start_step = int(amc.get("gate_cov_start_step", 0))
        self.amc_gate_cov_use_max = bool(amc.get("gate_cov_use_max", True))
        # Attention-centroid repulsion: push gated slots' spatial centers apart so
        # same-appearance nearby instances are less likely to merge into one slot.
        # L = mean_{i<j} g_i g_j * max(0, m - ||c_i - c_j||)^2 on normalized patch coords.
        # 0 disables. Enable mid/late via gate_rep_start_step to protect coarse phase.
        self.amc_gate_rep = float(amc.get("gate_rep", 0.0))
        self.amc_gate_rep_margin = float(amc.get("gate_rep_margin", 0.2))
        self.amc_gate_rep_start_step = int(amc.get("gate_rep_start_step", 0))
        # score shaping: gamma-sharpened mass (1.0 = plain attention mass, unchanged) and an
        # optional size-invariant purity rescue for small-but-cleanly-owned slots. The purity
        # threshold q anneals q_start -> q_end alongside p (q_start > 1 keeps the rescue off
        # early so the strict few-slots curriculum is not bypassed); eval uses q_end.
        self.amc_mass_gamma = float(amc.get("mass_gamma", 1.0))
        pq_end = amc.get("purity_q_end", None)
        self.amc_purity_q_end = float(pq_end) if pq_end is not None else None
        self.amc_purity_q_start = float(amc.get("purity_q_start", 1.5))
        self.amc_purity_tau = float(amc.get("purity_tau", 0.05))
        # Floor on the threshold used by the predictor re-gate only, as a multiple of the
        # uniform mass 1/n_slots (None = share the annealed p with the decoder, the
        # historical behaviour). See _state_gate_threshold for why this is a floor under the
        # schedule rather than a constant.
        #
        # The decoder's p has to anneal below the smallest object's mass or small objects
        # are never representable: at p = 1.5/7 a slot holding 2% of the patches gets
        # g = 0.011 and stays pinned to its init. But once p sits under the whole mass
        # distribution the sigmoid saturates and g goes flat across slots, and a flat gate
        # is an identity operation in the temporal mix -- the selectivity the curriculum is
        # supposed to provide disappears exactly when p reaches its end value. One scalar
        # cannot be both below the smallest object and inside the distribution.
        #
        # So the decoder keeps the annealed p (small objects admitted; the anti-fragmentation
        # effect by then lives in the weights, not in the live gate) and the temporal mix gets
        # a fixed threshold. Interpreted as an absolute mass threshold regardless of p_mode,
        # since a curriculum-free gate has no reason to track scene statistics.
        state_p_mult = amc.get("state_p_mult", None)
        self.amc_state_p_mult = float(state_p_mult) if state_p_mult is not None else None
        # If True, the predictor re-gate uses g / max(g) so the winning slot always advances
        # fully. Decoder still sees the raw gate (renorm-invariant).
        self.amc_state_max_norm = bool(amc.get("state_max_norm", False))
        # Eval-only: after π is built, binarize decoder + temporal + Pred src-gate at
        # π >= eval_hard_thresh. 0 disables (training / reported v39lam1gu). Applied
        # only when train=False; purity_weight ignores gate_mode so this is the hard
        # path for that form.
        self.amc_eval_hard_thresh = float(amc.get("eval_hard_thresh", 0.0))
        # Decoder Perron readout: m_s,i = softmax(α) π_s q̃_{1,s,i}. Mix / Pred
        # keep scalar π. perron_readout: train+eval. eval_perron_readout: eval only.
        self.amc_perron_readout = bool(amc.get("perron_readout", False))
        self.amc_eval_perron_readout = bool(amc.get("eval_perron_readout", False))
        # Optional eval-only Pred source-gate. None = same as train
        # (predictor_src_gate). False = vanilla Pred SA at eval only.
        _eval_src = amc.get("eval_predictor_src_gate", None)
        self.amc_eval_predictor_src_gate = (
            None if _eval_src is None else bool(_eval_src)
        )
        self.amc_predictor_pair_isolate = bool(
            amc.get("predictor_pair_isolate", False)
        )
        self.amc_eval_predictor_pair_isolate = bool(
            amc.get("eval_predictor_pair_isolate", False)
        )
        # Eval-only π^γ after the occupancy is built. 1.0 = identity.
        # Decoder, temporal max-norm mix, Perron, and pair-isolate all see π^γ.
        self.amc_eval_pi_gamma = float(amc.get("eval_pi_gamma", 1.0))
        if self.amc_eval_pi_gamma <= 0.0:
            raise ValueError(
                f"attn_mass_curriculum.eval_pi_gamma must be > 0, "
                f"got {self.amc_eval_pi_gamma}"
            )
        # π hysteresis (occlusion memory, v39h): the confidence statistic becomes the
        # leaky max π̃_t = max(π_t, γ π̃_{t-1}) with γ = gate_hysteresis in [0, 1).
        # A slot that was recently pure keeps its gate alive ~1/(1-γ) frames through a
        # brief occlusion; without it, partial occlusion splits the visible support
        # into disconnected 8-nbr components, the n8 gap reads that as a merge, and π
        # collapses exactly while the object is occluded (prior frozen at the wrong
        # moment, decoder mask suppressed). Ghosts gain nothing (π never was high).
        # Applies wherever conf feeds the gate (purity_weight, and the logratio
        # confidence branch); detached like π. 0 disables (v39 default) -- with γ=0
        # the forward pass is byte-identical to before the knob existed.
        self.amc_gate_hysteresis = float(amc.get("gate_hysteresis", 0.0))
        if not (0.0 <= self.amc_gate_hysteresis < 1.0):
            raise ValueError(
                f"attn_mass_curriculum.gate_hysteresis must be in [0, 1), "
                f"got {self.amc_gate_hysteresis}"
            )
        # Decoder-only leaky max (v39d): π̃_t = max(π_t, γ π̃_{t-1}) is applied to
        # the decoder gate only. Temporal mix keeps instantaneous π so a vanished
        # object freezes its prior (reappearance). γ=0.85 is ~5 frames of mask
        # stickiness for n8 false-dips without the v39h 20-frame overwrite.
        # Mutually exclusive with gate_hysteresis (that one hits both paths).
        self.amc_decoder_gate_hysteresis = float(
            amc.get("decoder_gate_hysteresis", 0.0)
        )
        if not (0.0 <= self.amc_decoder_gate_hysteresis < 1.0):
            raise ValueError(
                f"attn_mass_curriculum.decoder_gate_hysteresis must be in [0, 1), "
                f"got {self.amc_decoder_gate_hysteresis}"
            )
        if self.amc_gate_hysteresis > 0.0 and self.amc_decoder_gate_hysteresis > 0.0:
            raise ValueError(
                "attn_mass_curriculum: gate_hysteresis and decoder_gate_hysteresis "
                "cannot both be > 0 (shared vs decoder-only leaky max)"
            )
        # Temporal identity consistency (v39i): after max-norm, the mix gate is
        #   ρ_s = π̄_s * ReLU(cos(û_{t,s}, u_{t,s}))
        # Decoder still sees π. SlotAttention / Pred unchanged. Default False
        # keeps the v39 forward pass byte-identical.
        self.amc_state_identity_cos = bool(amc.get("state_identity_cos", False))
        # Temporal mix statistic, decoder stays on π (v39). Empty / 'pi' is v39.
        # 'lambda1' uses n8 λ1 (scale); 'mass' uses attention-mass fraction.
        sk = str(amc.get("state_conf_kind", "") or "").lower()
        if sk == "l1":
            sk = "lambda1"
        if sk not in ("", "pi", "lambda1", "mass"):
            raise ValueError(
                "attn_mass_curriculum.state_conf_kind must be '', 'pi', "
                f"'lambda1' or 'mass', got {sk!r}"
            )
        self.amc_state_conf_kind = "" if sk in ("", "pi") else sk
        # If True, stop-gradient the soft gate: decoder / temporal mix / gated losses see
        # sg(g). The curriculum still computes g from (m, c, p), but featrec cannot open
        # or close slots by backprop through g (same anti-gaming idea as sg(c)).
        # Bool (hard) gates are unchanged. Default False keeps a differentiable gate.
        self.amc_gate_detach = bool(amc.get("gate_detach", False))
        # If True, skip the predictor re-gate, so the next prior is always Pred(u). Since the
        # corrector output is not gated either, this takes the gate out of the temporal path
        # entirely and leaves it only reweighting decoder masks. Default False keeps
        #   hat{x}_{t+1} = g*Pred(u) + (1-g)*prior
        # so low-gate slots advance slowly and dormant ones carry an unchanged prior.
        self.amc_predictor_ungated = bool(amc.get("predictor_ungated", False))
        # If True, Pred self-attention is source-gated (v39lam1g):
        #   B_{i,j}^g = g_j exp(b_{i,j}) / sum_k g_k exp(b_{i,k})
        # so low-g slots do not contribute as keys/values. Decoder and the
        # temporal mix are unchanged. Default False keeps Pred as vanilla SA.
        self.amc_predictor_src_gate = bool(amc.get("predictor_src_gate", False))
        # v43: Pred source gate uses g / max(g), matching the temporal mix.
        # Default False keeps v39lam1g (raw detached statistic, not max-normed).
        self.amc_predictor_src_max_norm = bool(amc.get("predictor_src_max_norm", False))
        # Ungate the Pred SA diagonal: A_ii = 1, A_ij = g_j (i≠j). Ghost queries
        # can stay on their own key (1-layer SA typically does). With
        # predictor_src_max_norm, g_j is π̄_j so the winner key matches the
        # mix scale. Default False keeps v39lam1g per-key g_j, including a
        # closed diagonal for ghosts.
        self.amc_predictor_src_self = bool(amc.get("predictor_src_self", False))
        if self.amc_predictor_src_self and not self.amc_predictor_src_gate:
            raise ValueError(
                "attn_mass_curriculum.predictor_src_self requires predictor_src_gate"
            )
        # v39lam1gu_umix: occupancy-mix the corrector output for Pred / next
        # prior only. Decoder still reads u^SA. Default False keeps
        #   û_{t+1} = π̄ Pred(u^SA) + (1-π̄) û_t
        # True is
        #   u_t = π̄ u^SA + (1-π̄) û_t
        #   û_{t+1} = π̄ Pred(u_t) + (1-π̄) u_t
        self.amc_predictor_input_mix = bool(amc.get("predictor_input_mix", False))
        # After Pred(u_t), hold the mixed state (umix) or take Pred as the
        # next prior (umix_pred). Default True. False requires input_mix.
        #   True:  û_{t+1} = π̄ Pred(u_t) + (1-π̄) u_t   = u_t + π̄ D
        #   False: û_{t+1} = Pred(u_t)                   = u_t + D
        self.amc_predictor_mix_hold = bool(amc.get("predictor_mix_hold", True))
        if self.amc_predictor_input_mix and self.amc_predictor_ungated:
            raise ValueError(
                "attn_mass_curriculum.predictor_input_mix cannot be combined "
                "with predictor_ungated"
            )
        if not self.amc_predictor_mix_hold and not self.amc_predictor_input_mix:
            raise ValueError(
                "attn_mass_curriculum.predictor_mix_hold=false requires "
                "predictor_input_mix"
            )
        # Temporal-mix smoothing of the gate, decoder stays on instantaneous π.
        # π̃_t = m π_t + (1-m) π̃_{t-1}, then max(π̃_t, hold * π̃_{t-1}).
        # m=1, hold=0 is a no-op (default). Eval-only or train; no extra weights.
        self.amc_state_gate_ema = float(amc.get("state_gate_ema", 1.0))
        self.amc_state_gate_hold = float(amc.get("state_gate_hold", 0.0))
        # If False, loss_ss ignores active_mask (baseline-style full-slot contrastive).
        # Default True preserves v6/v7/v8 gated-anchor contrastive.
        self.amc_contrastive_gate = bool(amc.get("contrastive_gate", True))
        # v36: after computing detached purity c, map the uniform 1/K baseline to 0
        #   p = clip((K c - 1)/(K - 1), 0, 1)
        # Default False keeps v32/v33 (raw c is the gate).
        self.amc_purity_normalize = bool(amc.get("purity_normalize", False))
        # p schedule shape over anneal_steps: "linear" (default, v6/v7/v8), "log"
        # (geometric: p = p_start * (p_end/p_start)^frac), or "cosine" (coupled activity
        # curriculum: p = (1-lambda) p_start + lambda p_end with lambda = (1-cos(pi q))/2,
        # the same lambda that drives the confidence weight beta below).
        self.amc_p_anneal = str(amc.get("p_anneal", "linear")).lower()
        if self.amc_p_anneal not in ("linear", "log", "cosine"):
            raise ValueError(
                f"attn_mass_curriculum.p_anneal must be 'linear', 'log' or 'cosine', "
                f"got {self.amc_p_anneal!r}"
            )
        # --- evidence-aware gate (final method) ---
        # gate_form:
        #   "linear"   (default): g = sigmoid((m - p) / tau)             -- v6..v25
        #   "logratio": g = sigmoid((log r - log p) / tau_g) with
        #               log r = beta*log(m + eps) + (1-beta)*log(sg(c) + eps),
        #               c the per-slot assignment confidence (1 - H/log F) computed from the
        #               same gamma-sharpened attention as m. Relative (threshold-ratio)
        #               evidence, so the gate stays discriminative across the whole p
        #               schedule instead of saturating once p leaves the mass distribution.
        #   "purity_weight" (v32/v37/v37g/v37r/v38/v39/v39n/v41): g = c itself (usage is live z).
        #               c is ownership purity (conf_kind purity/purity_sharp), v37
        #               feature-Gram π (spectral = (λ1-λ2)/mass), v37g same C_s
        #               without /mass (spectral_gap = λ1-λ2), v37r same C_s
        #               (spectral_ratio = (λ1-λ2)/(λ1+λ2), bounded λ1/λ2),
        #               v38 dense relation-graph π
        #               (spectral_graph), v39 8-neighbor graph π (spectral_graph_n8),
        #               v39n relative n8 π (spectral_graph_n8_rel = v39 gap / λ1),
        #               v39s (spectral_graph_n8_l1imp = [λ1 - λ2⁺/(λ1+ε)]_+),
        #               v39lam1 (spectral_graph_n8_lam1 = max(λ1, 0) from the
        #               Perron power pair; no λ2, no k=8 Ritz),
        #               v39ind induced n8 π (spectral_graph_n8_ind = λ1-λ2 of
        #               S_ind re-normalized on thresholded support),
        #               or v41 learned usage z = sigmoid(head(sg(A))) (conf_kind usage).
        #               Thresholdless: no p schedule, no tau, no beta --
        #               the decoder sees softmax(alpha + log c) and the temporal mix
        #               uses c / max(c). Self-annealing (uniform untrained c is
        #               cancelled by the renorm/max-norm), replacing the curriculum.
        # state_gate_form (optional, v26p): the temporal mix can use a different form
        # than the decoder. p_end_mult=0 opens the decoder mass gate (identity) at the
        # end of the curriculum / at eval, instead of log(eps).
        # state_mul_decoder (v26pg): temporal = state_statistic ⊙ g_dec before max-norm.
        # decoder_mul_state (v26pgd): decoder masks = g_dec ⊙ state_statistic. With both
        # flags the two paths share g ⊙ π; p=0 / eval then uses π on the decoder.
        self.amc_gate_form = str(amc.get("gate_form", "linear")).lower()
        if self.amc_gate_form not in ("linear", "logratio", "purity_weight"):
            raise ValueError(
                f"attn_mass_curriculum.gate_form must be 'linear', 'logratio' or "
                f"'purity_weight', got {self.amc_gate_form!r}"
            )
        # Optional temporal-mix form, independent of the decoder gate (v26p). None / the
        # same string as gate_form keeps the historical shared path (plus gate_p_state
        # as a threshold split of that same form).
        sgf = amc.get("state_gate_form", None)
        if sgf is None or str(sgf).strip() == "":
            self.amc_state_gate_form = None
        else:
            self.amc_state_gate_form = str(sgf).lower()
            if self.amc_state_gate_form not in ("linear", "logratio", "purity_weight"):
                raise ValueError(
                    f"attn_mass_curriculum.state_gate_form must be 'linear', 'logratio' "
                    f"or 'purity_weight', got {self.amc_state_gate_form!r}"
                )
            if self.amc_state_gate_form == self.amc_gate_form:
                self.amc_state_gate_form = None
        if self.amc_state_gate_form is not None and self.amc_state_p_mult is not None:
            raise ValueError(
                "attn_mass_curriculum.state_gate_form and state_p_mult cannot both be "
                "set: the former replaces the temporal mass threshold with a different "
                "gate statistic"
            )
        # v26pg: temporal mix uses (state statistic) ⊙ decoder g, then max-norm.
        # Requires a split state_gate_form — a shared gate would square g on the mix.
        self.amc_state_mul_decoder = bool(amc.get("state_mul_decoder", False))
        if self.amc_state_mul_decoder and self.amc_state_gate_form is None:
            raise ValueError(
                "attn_mass_curriculum.state_mul_decoder requires a distinct "
                "state_gate_form (temporal π ⊙ g_dec); with a shared gate this "
                "would square g on the mix"
            )
        # v26pgd: decoder masks use g_dec ⊙ state statistic (π). Same split-form
        # requirement: a shared gate would square g on the reconstruction mix.
        self.amc_decoder_mul_state = bool(amc.get("decoder_mul_state", False))
        if self.amc_decoder_mul_state and self.amc_state_gate_form is None:
            raise ValueError(
                "attn_mass_curriculum.decoder_mul_state requires a distinct "
                "state_gate_form (decoder g_dec ⊙ π); with a shared gate this "
                "would square g on the decoder"
            )
        # Confidence definition for the logratio gate's second branch (v29):
        #   "entropy": c = 1 - H/log F over the gamma-sharpened attention (v26 default).
        #              Spatial concentration; penalizes large objects by construction
        #              (H grows with log(object size)).
        #   "entropy_max": v26fmax. c_ent * max_f A_{s,f} on RAW softmax-over-slots
        #              attention. Entropy is scale-invariant inside the slot, so
        #              always-second leftovers with the same support as a small
        #              object score like one; the peak restores ownership height
        #              (winner ~1, never-argmax ghost < 0.5). Still size-penalizes
        #              large exclusive objects via c_ent. Detached.
        #   "purity" : c = sum A^2 / sum A over the RAW attention -- the attention-weighted
        #              mean of the slot's own per-patch share. Direct, size-invariant
        #              ownership quality.
        #   "purity_sharp": the same statistic on the gamma-sharpened attention the mass
        #              branch uses (one distribution, two moments). Best ghost-vs-small
        #              separation on v20 @ 100k (event_analysis/conf_vs_purity_probe.py).
        #   "spectral": v37. π_s = (λ1-λ2) / (Σ_i A_{i,s} + eps) from
        #              C_s = Z^T diag(a_s^2) Z on L2-normalized backbone tokens Z.
        #              Ownership of two feature modes raises λ2, so the v32 blind
        #              spot (clean part-split, c~=1) is visible without N-cut.
        #              Raw A (no gamma): a^2 is the within-slot moment. No 1/K
        #              map; detached.
        #              spectral_proj_dim>0: fixed orthonormal D→d before C_s (fast).
        #   "spectral_gap": v37g. Same C_s as v37, π_s = λ1-λ2 (no /mass).
        #              Rank-1 exclusive recovers Σ a^2, not Σ a^2 / Σ A, so a
        #              one-color leftover no longer scores like a full object.
        #   "spectral_ratio": v37r. Same C_s, π_s = (λ1-λ2)/(λ1+λ2+eps) in [0,1].
        #              Bounded map of λ1/λ2: (r-1)/(r+1). Size-free; two equal
        #              modes and isotropic ghosts go to 0. Raw λ1/λ2 is not
        #              used (explodes at λ2=0).
        #   "spectral_graph": v38. π_s = λ1 - max(λ2, 0) of
        #              G_s = diag(a_s) S diag(a_s), S = D^{-1/2} R D^{-1/2} on the
        #              same ReLU-cosine R as the Key curriculum (Z from X^bind).
        #              No /mass, no gamma, no 1/K; detached. Cuts and second
        #              communities in the patch graph drop π.
        #   "spectral_graph_n8": v39. Same G_s construction as v38, but R is
        #              8-neighbor ReLU-cosine only. π_s = λ1 - max(λ2, 0).
        #              Curriculum P stays global.
        #   "spectral_graph_n8_rel": v39n. Same G_s / n8 R as v39, then
        #              π_s = (λ1 - max(λ2, 0)) / max(λ1, eps) in [0, 1].
        #              Scale-invariant: two-community merge still drops π;
        #              G_s scale no longer kills ghosts.
        #   "spectral_graph_n8_l1imp": v39s. Same G_s / n8 R as v39, then
        #              π_s = [λ1 - max(λ2, 0) / (λ1 + eps)]_+.
        #              Keeps G_s scale; subtracts relative impurity.
        #   "spectral_graph_n8_lam1": v39lam1. Same G_s / n8 R as v39, then
        #              π_s = max(λ1, 0) from the +ones Perron power pair
        #              (q1^T G q1, q1). Occupancy/scale only; no λ2 / k=8 Ritz.
        #              Decoder and temporal mix both see λ1 (unlike v39l1).
        #   "spectral_graph_n8_ind": v39ind. Induced n8 graph on support
        #              Ω = {a ≥ support_rel * max a}, S re-normalized there.
        #              π_s = λ1 - max(λ2, 0). Two n8-components → π=0.
        #   "usage": v41. z_s = sigmoid(MLP(sg(A_s)) -> cross-slot SA). Live; the
        #              head stop-grads A. No 1/K map (sigmoid is already in (0, 1)).
        # Detached in every case except usage (anti-gaming). Not to be confused with
        # the purity_q OR-rescue, which can only OPEN gates and bypasses the evidence
        # score entirely; this is the multiplicative evidence branch.
        self.amc_conf_kind = str(amc.get("conf_kind", "entropy")).lower()
        if self.amc_conf_kind not in (
            "entropy",
            "entropy_max",
            "purity",
            "purity_sharp",
            "spectral",
            "spectral_gap",
            "spectral_ratio",
            "spectral_graph",
            "spectral_graph_n8",
            "spectral_graph_n8_rel",
            "spectral_graph_n8_l1imp",
            "spectral_graph_n8_lam1",
            "spectral_graph_n8_ind",
            "usage",
        ):
            raise ValueError(
                f"attn_mass_curriculum.conf_kind must be 'entropy', 'entropy_max', "
                f"'purity', 'purity_sharp', 'spectral', 'spectral_gap', "
                f"'spectral_ratio', "
                f"'spectral_graph', 'spectral_graph_n8', 'spectral_graph_n8_rel', "
                f"'spectral_graph_n8_l1imp', 'spectral_graph_n8_lam1', "
                f"'spectral_graph_n8_ind' or 'usage', "
                f"got {self.amc_conf_kind!r}"
            )
        # purity_weight is DEFINED as ownership / spectral / usage weighting; the
        # conf_kind default ("entropy") would silently weight by spatial concentration
        # instead, so an explicit purity/usage choice is required. Same check when
        # only the temporal mix uses purity_weight (v26p).
        if (
            self.amc_gate_form == "purity_weight"
            or self.amc_state_gate_form == "purity_weight"
        ) and self.amc_conf_kind not in (
            "purity",
            "purity_sharp",
            "spectral",
            "spectral_gap",
            "spectral_ratio",
            "spectral_graph",
            "spectral_graph_n8",
            "spectral_graph_n8_rel",
            "spectral_graph_n8_l1imp",
            "spectral_graph_n8_lam1",
            "spectral_graph_n8_ind",
            "usage",
        ):
            raise ValueError(
                "attn_mass_curriculum gate_form/state_gate_form='purity_weight' "
                "requires conf_kind 'purity', 'purity_sharp', 'spectral', "
                f"'spectral_gap', 'spectral_ratio', 'spectral_graph', 'spectral_graph_n8', "
                f"'spectral_graph_n8_rel', 'spectral_graph_n8_l1imp', "
                f"'spectral_graph_n8_lam1', "
                f"'spectral_graph_n8_ind' or 'usage', "
                f"got {self.amc_conf_kind!r}"
            )
        # Coupled confidence weight: beta_t = 1 - lambda_t (1 - beta_final), i.e. the
        # confidence contribution 1-beta_t ramps 0 -> 1-beta_final with the SAME lambda that
        # relaxes the threshold. As the activity criterion is relaxed to accommodate smaller
        # objects, assignment confidence is simultaneously introduced to suppress the diffuse
        # inactive slots that the lower threshold would otherwise admit. beta_final=1.0
        # keeps the gate mass-only (confidence path never engages).
        self.amc_beta_final = float(amc.get("beta_final", 1.0))
        if not (0.0 <= self.amc_beta_final <= 1.0):
            raise ValueError(
                f"attn_mass_curriculum.beta_final must be in [0, 1], got {self.amc_beta_final}"
            )
        # beta_start decouples the confidence ramp from the curriculum for ablations:
        # beta_t = beta_start - lambda_t (beta_start - beta_final). Default 1.0 recovers the
        # coupled schedule above; beta_start == beta_final holds beta fixed from step 0
        # (uncoupled ablation, e.g. v26f).
        self.amc_beta_start = float(amc.get("beta_start", 1.0))
        if not (0.0 <= self.amc_beta_start <= 1.0):
            raise ValueError(
                f"attn_mass_curriculum.beta_start must be in [0, 1], got {self.amc_beta_start}"
            )
        # Log-domain gate temperature tau_g (dimensionless, units of log evidence-ratio):
        # g = 0.5 at r = p, and e.g. r = 2p gives sigmoid(log 2 / tau_g) ~= 0.80 at 0.5.
        self.amc_gate_tau_log = float(amc.get("gate_tau_log", 0.5))
        # tau schedule shape over anneal_steps: "linear" (default) or "log"
        # (geometric: tau = tau_start * (tau_end/tau_start)^frac). Defaults to linear so
        # v6–v10 configs are unchanged; v11 uses log for a faster early soft->sharp drop.
        self.amc_tau_anneal = str(amc.get("tau_anneal", "linear")).lower()
        if self.amc_tau_anneal not in ("linear", "log"):
            raise ValueError(
                f"attn_mass_curriculum.tau_anneal must be 'linear' or 'log', got {self.amc_tau_anneal!r}"
            )
        self._active_mask = None  # stashed per forward for loss/logging
        # Val scalars for loggers. Lightning 1.9 + max_epochs=-1 keeps one eval
        # ResultCollection for the whole run; self.log(..., on_epoch=True) of live
        # CUDA tensors (v41 usage z in featrec) can pin the first val's _computed
        # and reprint it every val_check_interval. Accumulate python floats here
        # and write loggers in on_validation_end.
        self._val_loss_acc: Dict[str, float] = {}
        self._val_loss_weight: float = 0.0
        self._pending_val_scalars: Optional[Dict[str, float]] = None
        self._gate_p_eff_mean = None
        self._state_gate_mean = None
        self._gate_delta = None  # stashed for residual L2 (may be graph-connected)
        self._gate_conf_mean = None  # mean assignment confidence (logratio gate, logging)
        self._identity_cos_mean = None  # mean ReLU-cosine û vs u (v39i, logging)
        self._util_rel_delta = None  # mean rel_delta of sampled slots (slot_utility, logging)
        self._util_rel_add = None
        self._util_c_mean = None
        self._util_n_suppressed = None
        self._util_pi_max = None
        self._util_pi_entropy = None

        # --- v40/v41 coupled slot-entropy / impurity (train-only aux) ---
        # L_ent on the gate scalar (v40: ownership c from live A; v41/v42: usage z).
        # entropy_normalize True (default): H(q)/log K in [0, 1]. False (v42): raw
        # H(q) nats so the same object count is K-invariant. L_imp = scalar-weighted
        # mean of a two-mode impurity: v40 uses λ2/λ1 on G_s (impurity_kind=g_s);
        # v39ind uses exp(-(λ1-λ2)/τ) of the induced n8 S (induced_n8); v41 uses
        # λ2/λ1 of the v37 Gram C_s = Z^T diag(a^2) Z (impurity_kind=c_s).
        # C_s top-2 is 16 subspace iters + 2D Ritz (same solver as the v37
        # gate); proj_dim is the JL width, not an eigensolver switch.
        # w_ent = (1-λ) w_ent_start + λ w_ent_end, same for w_imp.
        # v40 (only `weight` w0):  w_ent w0→0, w_imp 0→w0.
        # v41 endpoints: w_ent 0.3→0.1, w_imp 0.1→0.3 (neither dies).
        sei = slot_ent_impurity or {}
        self.sei_enabled = bool(sei.get("enabled", False))
        self.sei_weight = float(sei.get("weight", 0.2))
        self.sei_w_ent_start = float(sei.get("w_ent_start", self.sei_weight))
        self.sei_w_ent_end = float(sei.get("w_ent_end", 0.0))
        self.sei_w_imp_start = float(sei.get("w_imp_start", 0.0))
        self.sei_w_imp_end = float(sei.get("w_imp_end", self.sei_weight))
        self.sei_anneal_steps = int(sei.get("anneal_steps", 50000))
        self.sei_eps = float(sei.get("eps", 1e-6))
        self.sei_chunk_size = int(sei.get("chunk_size", 8))
        self.sei_n_iter = int(sei.get("n_iter", 16))
        self.sei_entropy_normalize = bool(sei.get("entropy_normalize", True))
        self.sei_target = str(sei.get("target", "c")).lower()
        ik = str(sei.get("impurity_kind", "g_s") or "g_s").lower()
        if ik in ("", "gs", "g_s", "spectral_graph_n8"):
            ik = "g_s"
        elif ik in ("induced_n8", "spectral_graph_n8_ind"):
            ik = "induced_n8"
        elif ik in ("c_s", "cs", "spectral", "feature_gram"):
            ik = "c_s"
        else:
            raise ValueError(
                "slot_ent_impurity.impurity_kind must be 'g_s', 'induced_n8' "
                f"or 'c_s', got {ik!r}"
            )
        self.sei_impurity_kind = ik
        self.sei_proj_dim = int(sei.get("proj_dim", 0))
        if self.sei_proj_dim < 0:
            raise ValueError(
                f"slot_ent_impurity.proj_dim must be >= 0, got {self.sei_proj_dim}"
            )
        self.sei_support_rel = float(sei.get("support_rel", 0.25))
        self.sei_fiedler_tau = float(sei.get("fiedler_tau", 0.05))
        self.amc_n8_support_rel = float(amc.get("n8_support_rel", 0.25))
        self.amc_spectral_proj_dim = int(amc.get("spectral_proj_dim", 0))
        if self.sei_target not in ("c", "z"):
            raise ValueError(
                f"slot_ent_impurity.target must be 'c' or 'z', got {self.sei_target!r}"
            )
        uh_cfg = usage_head_config or {}
        self.usage_head_enabled = bool(uh_cfg.get("enabled", False))
        if self.usage_head_enabled and self.amc_conf_kind != "usage":
            raise ValueError(
                "usage_head.enabled requires attn_mass_curriculum.conf_kind='usage'"
            )
        if self.amc_conf_kind == "usage" and not self.usage_head_enabled:
            raise ValueError(
                "attn_mass_curriculum.conf_kind='usage' requires usage_head.enabled"
            )
        if self.sei_enabled and self.sei_target == "z" and not self.usage_head_enabled:
            raise ValueError(
                "slot_ent_impurity.target='z' requires usage_head.enabled"
            )
        if self.usage_head_enabled and not bool(uh_cfg.get("live_gate", True)):
            self.amc_gate_detach = True
        # --- v43 reconstruction-based usage redistribution ---
        # L_gate trains usage-head logits on simplex-projected ∇J. Decoder /
        # temporal see sg(z). λ is concentration; 0.3 keeps it below typical |d|.
        ur = slot_usage_redistribute or {}
        self.usage_redist_enabled = bool(ur.get("enabled", False))
        self.usage_redist_lambda = float(ur.get("lambda", ur.get("lam", 0.3)))
        self.usage_redist_eps = float(ur.get("eps", 1e-8))
        self.usage_redist_z_eps = float(ur.get("z_eps", 0.0))
        self._usage_d_rms = None
        self._usage_v_rms = None
        self._usage_n_support = None
        if self.usage_redist_enabled:
            if not self.usage_head_enabled:
                raise ValueError(
                    "slot_usage_redistribute.enabled requires usage_head.enabled"
                )
            if self.usage_redist_lambda < 0.0:
                raise ValueError(
                    f"slot_usage_redistribute.lambda must be >= 0, got "
                    f"{self.usage_redist_lambda}"
                )
        if self.sei_enabled:
            for name, val in (
                ("weight", self.sei_weight),
                ("w_ent_start", self.sei_w_ent_start),
                ("w_ent_end", self.sei_w_ent_end),
                ("w_imp_start", self.sei_w_imp_start),
                ("w_imp_end", self.sei_w_imp_end),
            ):
                if val < 0.0:
                    raise ValueError(
                        f"slot_ent_impurity.{name} must be >= 0, got {val}"
                    )
            if self.sei_anneal_steps <= 0:
                raise ValueError(
                    "slot_ent_impurity.anneal_steps must be positive, "
                    f"got {self.sei_anneal_steps}"
                )
            if self.sei_impurity_kind == "induced_n8":
                if self.sei_fiedler_tau <= 0.0:
                    raise ValueError(
                        "slot_ent_impurity.fiedler_tau must be > 0, "
                        f"got {self.sei_fiedler_tau}"
                    )
                if not (0.0 <= self.sei_support_rel <= 1.0):
                    raise ValueError(
                        "slot_ent_impurity.support_rel must be in [0, 1], "
                        f"got {self.sei_support_rel}"
                    )

    def _p_residual_alpha_eff(self, train: bool) -> float:
        """Warm up residual strength so early coarse curriculum stays near v10."""
        alpha = self.amc_p_residual_alpha
        if not train or self.amc_p_residual_warmup_steps <= 0:
            return alpha
        step = self.trainer.global_step
        return alpha * min(1.0, float(step) / float(self.amc_p_residual_warmup_steps))

    def _util_add_insert_flag(self, su: Dict[str, Any]) -> bool:
        """slot_utility.add: insert/sigma/true enables σ-m restore; off disables."""
        raw = su.get("add", su.get("add_insert", False))
        if raw in (True, 1):
            return True
        if raw in (False, None, 0):
            return False
        if isinstance(raw, str):
            key = raw.strip().lower()
            if key in ("insert", "sigma", "sigma_m", "on", "true", "yes"):
                return True
            if key in ("none", "off", "false", "drop", "0"):
                return False
            raise ValueError(
                f"slot_utility.add must be 'insert' or 'off', got {raw!r}"
            )
        return bool(raw)

    def _live_usage_z(self, outputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Live softmax z from the usage head. Decoder/temporal may hold sg(z)."""
        proc = outputs.get("processor") or {}
        z = proc.get("gate_conf")
        if z is None:
            z = self._active_mask
        if z is None or not torch.is_floating_point(z):
            return None
        return z

    def _curriculum_lambda(self, train: bool) -> float:
        """Coupled curriculum coefficient lambda_t = (1 - cos(pi q)) / 2, q = step/anneal.

        One coefficient drives both the threshold relaxation (p_anneal='cosine') and the
        confidence weight beta_t, so the two cannot drift apart. Evaluation uses the end
        of the curriculum (lambda = 1), consistent with the p/tau eval convention.
        """
        if not train:
            return 1.0
        step = self.trainer.global_step
        frac = min(max(step / max(self.amc_anneal_steps, 1), 0.0), 1.0)
        return 0.5 * (1.0 - float(np.cos(np.pi * frac)))

    def _sei_lambda(self, train: bool) -> float:
        """v40 cosine coefficient on slot_ent_impurity.anneal_steps.

        Same shape as `_curriculum_lambda`, but its own clock so the gate does
        not have to carry a p/β schedule. Eval / t>=T uses λ=1.
        """
        if not train:
            return 1.0
        step = self.trainer.global_step
        frac = min(max(step / max(self.sei_anneal_steps, 1), 0.0), 1.0)
        return 0.5 * (1.0 - float(np.cos(np.pi * frac)))

    def _sei_weights(self, train: bool) -> Tuple[float, float, float]:
        """(w_ent, w_imp, lambda_t) linear in λ on the configured endpoints."""
        lam = self._sei_lambda(train)
        w_ent = (1.0 - lam) * self.sei_w_ent_start + lam * self.sei_w_ent_end
        w_imp = (1.0 - lam) * self.sei_w_imp_start + lam * self.sei_w_imp_end
        return w_ent, w_imp, lam

    def _aux_ramp(self, ramp: str) -> float:
        """Weight multiplier for the v27 auxiliary losses.

        'lambda' reuses the coupled-curriculum coefficient lambda_t (0 during the coarse
        phase, 1 from the end of the curriculum on), so no new schedule is introduced;
        'none' returns 1.0.
        """
        if ramp == "lambda":
            return self._curriculum_lambda(True)
        return 1.0

    def _gate_beta(self, train: bool) -> Optional[float]:
        """Coverage weight beta_t = beta_start - lambda_t (beta_start - beta_final).

        beta_start defaults to 1.0 (coupled ramp of the confidence weight). None when the
        gate is the legacy linear form (no confidence branch).
        """
        if not self.attn_mass_enabled or self.amc_gate_form != "logratio":
            return None
        lam = self._curriculum_lambda(train)
        return self.amc_beta_start - lam * (self.amc_beta_start - self.amc_beta_final)

    def _annealed_p_mult(self, train: bool) -> float:
        """Annealed p multiplier (p_start_mult -> p_end_mult)."""
        if not train:
            return self.amc_p_end_mult
        step = self.trainer.global_step
        frac = min(max(step / max(self.amc_anneal_steps, 1), 0.0), 1.0)
        if self.amc_p_anneal == "log":
            if self.amc_p_start_mult <= 0.0 or self.amc_p_end_mult <= 0.0:
                raise ValueError("p_anneal='log' requires positive p_start_mult and p_end_mult")
            return self.amc_p_start_mult * (
                (self.amc_p_end_mult / self.amc_p_start_mult) ** frac
            )
        if self.amc_p_anneal == "cosine":
            lam = self._curriculum_lambda(train)
            return self.amc_p_start_mult + (self.amc_p_end_mult - self.amc_p_start_mult) * lam
        return self.amc_p_start_mult + (self.amc_p_end_mult - self.amc_p_start_mult) * frac

    def _gate_threshold(self, train: bool) -> Optional[float]:
        """Value passed as `gate_p` into the processor for the current step.

        absolute / learnable_residual: annealed absolute mass threshold (= p_mult / n_slots).
        median_ema: annealed p_mult (processor sets p = sg(EMA[median])*p_mult).
        purity_weight: None -- the form is thresholdless (the statistic is the gate).

        Training anneals over `amc_anneal_steps` (strict->loose). Evaluation uses end value.
        """
        if not self.attn_mass_enabled:
            return None
        if self.amc_gate_form == "purity_weight":
            return None
        p_mult = self._annealed_p_mult(train)
        if self.amc_p_mode == "median_ema":
            return p_mult
        # absolute and learnable_residual both receive p_sched as absolute threshold
        return p_mult / max(self.n_slots, 1)

    def _state_gate_threshold(self, train: bool) -> Optional[float]:
        """Mass threshold for the predictor re-gate, or None to share the decoder's.

        A floor under the annealed schedule, `max(p_sched, state_p_mult / n_slots)`, not a
        constant. A constant would be *looser* than the schedule for most of the traverse
        -- 0.5 only crosses 1.5 -> 0.1 linear at step 47k of 66k -- and would therefore
        delete the early capacity restriction, which is the curriculum's actual mechanism
        and acts precisely here in the temporal mix. As a floor the state gate tracks the
        decoder gate until the schedule drops past it and then holds, so the coarse-to-fine
        traverse is untouched and only the flat tail changes.

        Read off `_annealed_p_mult` rather than `_gate_threshold` so this stays an absolute
        mass threshold under every p_mode: a curriculum-free gate has no reason to track
        scene statistics. Evaluation uses p_end_mult, so it returns the floor.
        """
        if (
            not self.attn_mass_enabled
            or self.amc_state_p_mult is None
            or self.amc_state_gate_form is not None
        ):
            return None
        floor = self.amc_state_p_mult / max(self.n_slots, 1)
        scheduled = self._annealed_p_mult(train) / max(self.n_slots, 1)
        return max(scheduled, floor)

    def _gate_tau(self, train: bool) -> Optional[float]:
        """Sigmoid gate temperature for soft gating at the current step.

        Training anneals tau_start -> tau_end over `amc_anneal_steps` (soft->sharp).
        Shape is controlled by `tau_anneal` (linear or log). Evaluation always uses
        the final (sharpest) tau_end. Returns None when soft gating is disabled.
        """
        if not self.attn_mass_enabled or self.amc_gate_mode not in ("soft", "ste"):
            return None
        if not train:
            return self.amc_tau_end
        step = self.trainer.global_step
        frac = min(max(step / max(self.amc_anneal_steps, 1), 0.0), 1.0)
        if self.amc_tau_anneal == "log":
            if self.amc_tau_start <= 0.0 or self.amc_tau_end <= 0.0:
                raise ValueError("tau_anneal='log' requires positive tau_start and tau_end")
            return self.amc_tau_start * ((self.amc_tau_end / self.amc_tau_start) ** frac)
        return self.amc_tau_start + (self.amc_tau_end - self.amc_tau_start) * frac

    def _purity_q(self, train: bool) -> Optional[float]:
        """Purity-rescue threshold q for the current step (None = rescue disabled).

        Training linearly anneals q_start -> q_end over `amc_anneal_steps`. With
        q_start > 1 the rescue is effectively off early on (purity <= 1), so the strict
        few-slots phase of the curriculum is preserved; the rescue phases in together
        with the loosening mass threshold. Evaluation always uses q_end.
        """
        if not self.attn_mass_enabled or self.amc_purity_q_end is None:
            return None
        if not train:
            return self.amc_purity_q_end
        step = self.trainer.global_step
        frac = min(max(step / max(self.amc_anneal_steps, 1), 0.0), 1.0)
        return self.amc_purity_q_start + (self.amc_purity_q_end - self.amc_purity_q_start) * frac

    def configure_optimizers(self):
        modules = {
            "initializer": self.initializer,
            "encoder": self.encoder,
            "processor": self.processor,
            "decoder": self.decoder,
        }
        return self.optimizer_builder(modules)

    def forward(self, inputs: Dict[str, Any], train=True, cycle=False) -> Dict[str, Any]:
        encoder_input = inputs[self.input_key]  # batch [x n_frames] x n_channels x height x width
        assert encoder_input.ndim == self.expected_input_dims
        batch_size = len(encoder_input)

        encoder_output = self.encoder(encoder_input)
        features = encoder_output["features"]
        B, T, HW, D = features.size()
        H = W = int(HW**0.5) if HW > 0 else 1

        ### --- slot expansion schedule --- ###
        if train and self._expansion_active:
            for hi in range(len(self.hier_steps)):
                if self.trainer.global_step == self.hier_steps[hi]: #
                    half = self.hier_n_slots[hi+1] # Next slot num
                    num_new = half - self.hier_n_slots[hi] # New slots to be added
                    losses = self.slot_loss_scale[:self.hier_n_slots[hi]]  # (2,)
                    weights = losses / losses.sum()  # (2,)

                    ### Determine the split number for each slot
                    # integer part
                    splits_star = weights * num_new  # (2,)
                    k = torch.floor(splits_star).long()

                    # fractional part
                    r = int(num_new - k.sum().item())
                    if r > 0:
                        frac = splits_star - k.float()
                        _, idxs = torch.sort(frac, descending=True)
                        for idx in idxs[:r]:
                            k[idx] += 1

                    with torch.no_grad():
                        prev = self.hier_n_slots[hi]

                        ### For measuring the noise scale
                        # cosine distance between slots
                        slots_raw = self.initializer.slots.data
                        slots_norm = F.normalize(slots_raw, dim=-1)
                        cos_all = torch.matmul(slots_norm[0], slots_norm[0].T)  # (N, N)

                        for slot_idx, cnt in enumerate(k.tolist()):
                            if cnt > 0:
                                cos_row = cos_all[slot_idx].clone()  # (N,)
                                cos_row[slot_idx] = -1.0
                                nn_idx = cos_row.argmax()
                                diff = slots_raw[0, slot_idx] - slots_raw[0, nn_idx]
                                d_l2 = diff.norm()  # norm(s_i - s_nn)
                                alpha = 0.2
                                norm_i = slots_raw[0, slot_idx].norm()
                                mean_norm = slots_raw[0].norm(dim=-1).mean() + 1e-8
                                noise_scale = alpha * d_l2 * (norm_i / mean_norm)
                                rand_dir = F.normalize(torch.randn(cnt, diff.numel(), device=diff.device), dim=-1)  # (cnt,D)
                                dir_vec = rand_dir
                                noise = dir_vec * noise_scale
                                base = slots_raw[:, slot_idx:slot_idx + 1].expand(-1, cnt, -1)  # (1,cnt,D)
                                self.initializer.slots[:, prev:prev + cnt] = base + noise.unsqueeze(0)  # (1,cnt,D)
                                prev += cnt

        slots_initial = self.initializer(batch_size=batch_size) # batch x n_slots x slot_dim
        if train and self._expansion_active:
            for hi in range(len(self.hier_steps)):
                if self.trainer.global_step <= self.hier_steps[hi]:
                    slots_initial = slots_initial[:, :self.hier_n_slots[hi], :]
                    break

        processor_kwargs: Dict[str, Any] = {"cycle": cycle}
        key_features = encoder_output.get("features_key")
        if key_features is not None:
            # ScanOverTime takes the full (B, T, ...) tensor as key_inputs;
            # LatentProcessor (image) takes a single frame as key_features.
            if hasattr(self.processor, "next_state_key"):
                processor_kwargs["key_inputs"] = key_features
            else:
                processor_kwargs["key_features"] = key_features
        if self.attn_mass_enabled and self.amc_conf_kind in (
            "spectral",
            "spectral_gap",
            "spectral_ratio",
            "spectral_graph",
            "spectral_graph_n8",
            "spectral_graph_n8_rel",
            "spectral_graph_n8_l1imp",
            "spectral_graph_n8_lam1",
            "spectral_graph_n8_ind",
        ):
            # Z / X^bind for v37 / v37g / v37r C_s, v38 dense S, and v39 / v39n 8-neighbor S.
            # Key-only curriculum exposes that as backbone_key; otherwise
            # backbone_features is X^bind (raw X when the curriculum is off / mix=1).
            bind = encoder_output.get("backbone_key")
            if bind is None:
                bind = encoder_output.get("backbone_features")
            if bind is None:
                raise ValueError(
                    f"attn_mass_curriculum.conf_kind={self.amc_conf_kind!r} "
                    "requires encoder backbone_features"
                )
            if hasattr(self.processor, "next_state_key"):
                processor_kwargs["bind_inputs"] = bind
            else:
                processor_kwargs["bind_features"] = bind

        if self.attn_mass_enabled:
            gate_p = self._gate_threshold(train)
            gate_tau = self._gate_tau(train)
            purity_q = self._purity_q(train)
            processor_output = self.processor(
                slots_initial, features, cycle=cycle,
                gate_p=gate_p, default_idx=self.amc_default_idx,
                gate_mode=self.amc_gate_mode, gate_tau=gate_tau,
                mass_gamma=self.amc_mass_gamma,
                purity_q=purity_q, purity_tau=self.amc_purity_tau,
                gate_p_state=self._state_gate_threshold(train),
                state_max_norm=self.amc_state_max_norm,
                predictor_ungated=self.amc_predictor_ungated,
                predictor_src_gate=(
                    bool(self.amc_predictor_src_gate)
                    if train or self.amc_eval_predictor_src_gate is None
                    else bool(self.amc_eval_predictor_src_gate)
                ),
                predictor_src_max_norm=self.amc_predictor_src_max_norm,
                predictor_src_self=self.amc_predictor_src_self,
                predictor_pair_isolate=(
                    bool(self.amc_predictor_pair_isolate)
                    or (not train and bool(self.amc_eval_predictor_pair_isolate))
                ),
                predictor_input_mix=self.amc_predictor_input_mix,
                predictor_mix_hold=self.amc_predictor_mix_hold,
                state_gate_ema=self.amc_state_gate_ema,
                state_gate_hold=self.amc_state_gate_hold,
                p_mode=self.amc_p_mode,
                median_ema_momentum=self.amc_median_ema_momentum,
                p_residual_net=self.amc_p_residual_net,
                p_residual_alpha=self._p_residual_alpha_eff(train),
                gate_form=self.amc_gate_form,
                gate_beta=self._gate_beta(train),
                gate_tau_log=self.amc_gate_tau_log,
                conf_kind=self.amc_conf_kind,
                gate_detach=self.amc_gate_detach,
                purity_normalize=self.amc_purity_normalize,
                state_gate_form=self.amc_state_gate_form,
                state_mul_decoder=self.amc_state_mul_decoder,
                decoder_mul_state=self.amc_decoder_mul_state,
                gate_hysteresis=self.amc_gate_hysteresis,
                decoder_gate_hysteresis=self.amc_decoder_gate_hysteresis,
                state_identity_cos=self.amc_state_identity_cos,
                state_conf_kind=self.amc_state_conf_kind,
                n8_support_rel=self.amc_n8_support_rel,
                spectral_proj_dim=self.amc_spectral_proj_dim,
                eval_hard_thresh=(self.amc_eval_hard_thresh if not train else 0.0),
                eval_perron_readout=(
                    bool(self.amc_perron_readout)
                    or (not train and bool(self.amc_eval_perron_readout))
                ),
                pi_gamma=(
                    1.0 if train else float(self.amc_eval_pi_gamma)
                ),
                **{k: v for k, v in processor_kwargs.items() if k != "cycle"},
            )
            slots = processor_output["state"]
            active_mask = processor_output.get("active_mask")
            self._active_mask = active_mask
            gate_conf = processor_output.get("gate_conf")
            self._gate_conf_mean = (
                float(gate_conf.float().mean().detach()) if gate_conf is not None else None
            )
            ident = processor_output.get("identity_cos")
            self._identity_cos_mean = (
                float(ident.float().mean().detach()) if ident is not None else None
            )
            state_gate = processor_output.get("state_gate")
            if state_gate is not None and state_gate.dtype != torch.bool:
                self._state_gate_mean = float(state_gate.float().mean().detach())
            else:
                self._state_gate_mean = None
            p_eff = processor_output.get("gate_p_eff")
            if p_eff is not None and torch.is_tensor(p_eff):
                self._gate_p_eff_mean = float(p_eff.float().mean().detach())
            else:
                self._gate_p_eff_mean = None
            gate_delta = processor_output.get("gate_delta")
            # keep tensor for residual L2 (needs grad); logging uses detach mean
            self._gate_delta = gate_delta if (gate_delta is not None and torch.is_tensor(gate_delta)) else None
            need_slot_recons = self.training and (
                (self.util_weight > 0.0 and self.util_mode == "algebraic")
                or self.usage_redist_enabled
            )
            need_ungated = self.featrec_ungated or (
                self.training
                and (
                    self.usage_redist_enabled
                    or (self.util_weight > 0.0 and self.util_add_insert)
                )
            )
            decoder_output = self.decoder(
                slots,
                processor_output.get("decoder_gate", active_mask),
                return_slot_recons=need_slot_recons,
                return_ungated=need_ungated,
            )
        else:
            self._active_mask = None
            self._gate_p_eff_mean = None
            self._gate_delta = None
            self._state_gate_mean = None
            self._gate_conf_mean = None
            self._identity_cos_mean = None
            processor_output = self.processor(
                slots_initial, features, **processor_kwargs
            )
            slots = processor_output["state"]
            decoder_output = self.decoder(
                slots,
                return_ungated=self.featrec_ungated or (
                    self.training and self.usage_redist_enabled
                ),
            )
        if self.featrec_ungated and "reconstruction_ungated" not in decoder_output:
            decoder_output["reconstruction_ungated"] = self.decoder(slots)["reconstruction"]
        # feat_orig, feat_recon: (B, C, H, W)
        # 1) compute sobel gradients
        feat_recon = decoder_output['reconstruction'] #
        feat_orig = encoder_output['backbone_features']

        B, T, HW, D = feat_recon.shape
        H = W = int(HW ** 0.5) if HW > 0 else 1
        feat_recon = feat_recon.reshape(B, T, H, W, D).permute(0, 4, 1, 2, 3)  # (B, D, T, H, W)
        feat_orig = feat_orig.reshape(B, T, H, W, D).permute(0, 4, 1, 2, 3)  # (B, D, T, H, W)

        encoder_output['SSIM_orig'] = feat_orig
        encoder_output['SSIM_recon'] = feat_recon

        outputs = {
            "batch_size": batch_size,
            "encoder": encoder_output,
            "processor": processor_output,
            "decoder": decoder_output,
        }
        outputs["targets"] = self.get_targets(inputs, outputs)

        return outputs

    def process_masks(
        self,
        masks: torch.Tensor,
        inputs: Dict[str, Any],
        resizer: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if masks is None:
            return None, None, None

        if resizer is None:
            masks_for_vis = masks
            masks_for_vis_hard = self.mask_soft_to_hard(masks)
            masks_for_metrics_hard = masks_for_vis_hard
        else:
            masks_for_vis = resizer(masks, inputs[self.input_key])
            masks_for_vis_hard = self.mask_soft_to_hard(masks_for_vis)
            target_masks = inputs.get("segmentations")
            if target_masks is not None and masks_for_vis.shape[-2:] != target_masks.shape[-2:]:
                masks_for_metrics = resizer(masks, target_masks)
                masks_for_metrics_hard = self.mask_soft_to_hard(masks_for_metrics)
            else:
                masks_for_metrics_hard = masks_for_vis_hard

        return masks_for_vis, masks_for_vis_hard, masks_for_metrics_hard

    @torch.no_grad()
    def aux_forward(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute auxilliary outputs only needed for metrics and visualisations."""
        decoder_masks = outputs["decoder"].get("masks")
        decoder_masks, decoder_masks_hard, decoder_masks_metrics_hard = self.process_masks(
            decoder_masks, inputs, self.mask_resizers.get("decoder")
        )

        grouping_masks = outputs["processor"]["corrector"].get("masks")
        grouping_masks, grouping_masks_hard, grouping_masks_metrics_hard = self.process_masks(
            grouping_masks, inputs, self.mask_resizers.get("grouping")
        )

        aux_outputs = {}
        if decoder_masks is not None:
            aux_outputs["decoder_masks"] = decoder_masks
        if decoder_masks_hard is not None:
            aux_outputs["decoder_masks_vis_hard"] = decoder_masks_hard
        if decoder_masks_metrics_hard is not None:
            aux_outputs["decoder_masks_hard"] = decoder_masks_metrics_hard
        if grouping_masks is not None:
            aux_outputs["grouping_masks"] = grouping_masks
        if grouping_masks_hard is not None:
            aux_outputs["grouping_masks_vis_hard"] = grouping_masks_hard
        if grouping_masks_metrics_hard is not None:
            aux_outputs["grouping_masks_hard"] = grouping_masks_metrics_hard

        return aux_outputs

    def get_targets(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        if self.target_encoder:
            target_encoder_input = inputs[self.target_encoder_input_key]
            assert target_encoder_input.ndim == self.expected_input_dims

            with torch.no_grad():
                encoder_output = self.target_encoder(target_encoder_input)

            outputs["target_encoder"] = encoder_output

        targets = {}
        for name, loss_fn in self.loss_fns.items():
            targets[name] = loss_fn.get_target(inputs, outputs)

        return targets

    def compute_loss(self, outputs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        for name, loss_fn in self.loss_fns.items():
            prediction = loss_fn.get_prediction(outputs)
            target = outputs["targets"][name]
            if name == 'loss_featrec' and self._expansion_active:
                if self.trainer.global_step + 1 == self.hier_steps[0] or self.trainer.global_step + 1 == self.hier_steps[1]:
                    loss_all = loss_fn(prediction, target, True).mean(-1, keepdim=True) # 64 2304 384 -> 64 2304 1
                    decoding_masks = outputs["decoder"]["masks"].permute(0, 1, 3, 2).flatten(1, 2) # 64 2304 2
                    masked_loss = loss_all * decoding_masks # 64 2304 2
                    slot_loss_per_slot = masked_loss.mean(dim=(0, 1)) # num_slot

                    self.slot_loss_scale[:len(slot_loss_per_slot)] = slot_loss_per_slot.detach() # 2

            if name == "loss_cc" and self.attn_mass_enabled and self._active_mask is not None:
                losses[name] = loss_fn(
                    prediction, target, occupancy=self._active_mask
                )
            elif name == "loss_ss" and self.attn_mass_enabled and self._active_mask is not None:
                ss_kwargs = {}
                if self.amc_contrastive_gate:
                    # restrict L_ss to live anchors; loss max-norms to π̄ = π / max_s(π)
                    ss_kwargs["active_mask"] = self._active_mask
                if bool(getattr(loss_fn, "cand_neg", False)) or bool(
                    getattr(loss_fn, "occ_kernel", False)
                ):
                    # occupancy → c=1-π̄ for cand_neg extras and/or occ_kernel blocks
                    ss_kwargs["occupancy"] = self._active_mask
                if ss_kwargs:
                    losses[name] = loss_fn(prediction, target, **ss_kwargs)
                else:
                    losses[name] = loss_fn(prediction, target)
            else:
                losses[name] = loss_fn(prediction, target)

        losses_weighted = [loss * self.loss_weights.get(name, 1.0) for name, loss in losses.items()]
        total_loss = torch.stack(losses_weighted).sum()

        # --- L1 gate sparsity penalty (soft/ste gating only, training only) ---
        # A constant per-slot "rent" (amc_gate_l1) pushes redundant slots' gates toward 0,
        # restoring the consolidation pressure that decoder renormalization removes for
        # lone-claimant (spatially-disjoint) fragments. Default slots are always-active and
        # therefore exempt. Logged raw as `loss_gate_sparsity`; the weighted term is summed
        # into the total. Skipped for hard gating (bool gate carries no gradient).
        if (
            self.training
            and self.attn_mass_enabled
            and self.amc_gate_l1 > 0.0
            and self.amc_gate_mode in ("soft", "ste")
            and self._active_mask is not None
            and self._active_mask.dtype != torch.bool
        ):
            gate = self._active_mask.float()  # (B, T, S)
            keep = torch.ones(gate.shape[-1], dtype=torch.bool, device=gate.device)
            for di in self.amc_default_idx:
                if 0 <= int(di) < keep.shape[0]:
                    keep[int(di)] = False
            if keep.any():
                gate_sparsity = gate[..., keep].mean()
                losses["loss_gate_sparsity"] = gate_sparsity
                total_loss = total_loss + self.amc_gate_l1 * gate_sparsity

        # --- batch-entropy diversity loss (soft/ste, training only) ---
        # Maximize the entropy of the batch-mean per-slot usage distribution so different
        # slots get used across the batch (load balancing), counteracting the rich-get-richer
        # collapse to a single slot. Logged term is (log S - H) >= 0 (0 = uniform usage,
        # log S = fully collapsed); its gradient pushes usage toward uniform.
        if (
            self.training
            and self.attn_mass_enabled
            and self.amc_gate_div > 0.0
            and self.amc_gate_mode in ("soft", "ste")
            and self._active_mask is not None
            and self._active_mask.dtype != torch.bool
        ):
            gate = self._active_mask.float()  # (B, T, S)
            gbar = gate.mean(dim=tuple(range(gate.dim() - 1)))  # (S,) mean usage per slot
            q = gbar / gbar.sum().clamp_min(1e-8)               # distribution over slots
            neg_entropy = (q * q.clamp_min(1e-8).log()).sum()   # = -H(q)
            log_s = float(np.log(gate.shape[-1]))
            div_term = neg_entropy + log_s                       # log S - H(q) >= 0
            losses["loss_gate_div"] = div_term
            total_loss = total_loss + self.amc_gate_div * div_term

        # --- error-weighted coverage (soft/ste, training only) ---
        # Encourage gated attention to cover high-residual patches (small/ignored regions).
        # Error weights are detached so the model cannot game the loss by inflating residuals.
        if (
            self.training
            and self.attn_mass_enabled
            and self.amc_gate_cov > 0.0
            and self.amc_gate_mode in ("soft", "ste")
            and self._active_mask is not None
            and self._active_mask.dtype != torch.bool
            and self.trainer.global_step >= self.amc_gate_cov_start_step
        ):
            cov_term = self._error_weighted_coverage(outputs, self._active_mask.float())
            if cov_term is not None:
                losses["loss_gate_cov"] = cov_term
                total_loss = total_loss + self.amc_gate_cov * cov_term

        # --- attention centroid repulsion (soft/ste, training only) ---
        # Penalize gated slots whose attention centroids fall inside a spatial margin.
        # Targets same-class nearby merges (hikers/cubs) without requiring residual gaps.
        if (
            self.training
            and self.attn_mass_enabled
            and self.amc_gate_rep > 0.0
            and self.amc_gate_mode in ("soft", "ste")
            and self._active_mask is not None
            and self._active_mask.dtype != torch.bool
            and self.trainer.global_step >= self.amc_gate_rep_start_step
        ):
            rep_term = self._attention_centroid_repulsion(outputs, self._active_mask.float())
            if rep_term is not None:
                losses["loss_gate_rep"] = rep_term
                total_loss = total_loss + self.amc_gate_rep * rep_term

        # --- direction-only dynamics supervision for the predictor (training only) ---
        if (
            self.training
            and self.dyn_weight > 0.0
            and self.trainer.global_step >= self.dyn_start_step
        ):
            dyn_term = self._dynamics_direction(outputs)
            if dyn_term is not None:
                losses["loss_dyn"] = dyn_term
                total_loss = total_loss + self.dyn_weight * dyn_term

        # --- learnable p residual L2 (keep corrections near zero / v10 default) ---
        if (
            self.training
            and self.attn_mass_enabled
            and self.amc_p_mode == "learnable_residual"
            and self.amc_p_residual_l2 > 0.0
            and self._gate_delta is not None
        ):
            delta_term = self._gate_delta.float().pow(2).mean()
            losses["loss_p_residual"] = delta_term
            total_loss = total_loss + self.amc_p_residual_l2 * delta_term

        # --- predictive feature reconstruction (v27) ---
        if (
            self.training
            and self.pred_weight > 0.0
            and self.trainer.global_step >= self.pred_start_step
        ):
            pred_term = self._predictive_recon_loss(outputs)
            if pred_term is not None:
                losses["loss_pred"] = pred_term
                total_loss = total_loss + (
                    self.pred_weight * self._aux_ramp(self.pred_ramp) * pred_term
                )

        # --- counterfactual slot-utility rent (v27) ---
        if (
            self.training
            and self.util_weight > 0.0
            and self.trainer.global_step >= self.util_start_step
        ):
            util_term = self._slot_utility_loss(outputs)
            if util_term is not None:
                losses["loss_util"] = util_term
                total_loss = total_loss + (
                    self.util_weight * self._aux_ramp(self.util_ramp) * util_term
                )

        # --- v44 simplex concentration: 1 - ||z||^2 (train-only) ---
        if self.training and self.util_psi_weight > 0.0:
            z_psi = self._live_usage_z(outputs)
            if z_psi is not None:
                zf = z_psi.float()
                psi = (1.0 - zf.pow(2).sum(dim=-1)).mean()
                losses["loss_psi"] = psi
                total_loss = total_loss + self.util_psi_weight * psi

        # --- v43 reconstruction-based usage redistribution (train-only) ---
        if self.training and self.usage_redist_enabled:
            gate_term = self._usage_redistribute_loss(outputs)
            if gate_term is not None:
                losses["loss_gate"] = gate_term
                total_loss = total_loss + gate_term

        # --- v40/v41 coupled slot-entropy / impurity (train-only) ---
        if self.training and self.sei_enabled and (
            self.sei_w_ent_start > 0.0
            or self.sei_w_ent_end > 0.0
            or self.sei_w_imp_start > 0.0
            or self.sei_w_imp_end > 0.0
        ):
            w_ent, w_imp, _lam = self._sei_weights(True)
            att, bind = self._sei_att_and_bind(outputs)
            if att is not None:
                if self.sei_target == "z":
                    # Prefer live gate_conf so L_ent still trains the head if the
                    # decoder/temporal gate was stop-grad (live_gate false).
                    proc = outputs.get("processor") or {}
                    z = proc.get("gate_conf")
                    if z is None:
                        z = self._active_mask
                    if z is None or not torch.is_floating_point(z):
                        raise ValueError(
                            "slot_ent_impurity.target='z' requires a float usage gate"
                        )
                    conf = z.reshape(-1, z.shape[-1]) if z.ndim == 3 else z
                    if conf.shape[0] != att.shape[0] or conf.shape[-1] != att.shape[1]:
                        raise ValueError(
                            f"usage gate {tuple(conf.shape)} vs attention {tuple(att.shape)}"
                        )
                else:
                    conf = ownership_confidence(att, eps=self.sei_eps)
                ent_raw = slot_confidence_entropy(
                    conf, eps=self.sei_eps, normalize=False
                )
                ent_norm = ent_raw / math.log(max(conf.shape[-1], 2))
                ent = ent_norm if self.sei_entropy_normalize else ent_raw
                ent = ent.mean()
                losses["loss_ent"] = ent
                if not self.sei_entropy_normalize:
                    losses["loss_ent_norm"] = ent_norm.mean()
                if w_ent > 0.0:
                    total_loss = total_loss + w_ent * ent
                if w_imp > 0.0:
                    if bind is None:
                        raise ValueError(
                            "slot_ent_impurity requires encoder backbone_features"
                        )
                    if self.sei_impurity_kind == "induced_n8":
                        rho = spectral_graph_n8_induced_impurity(
                            att,
                            bind,
                            eps=self.sei_eps,
                            chunk_size=self.sei_chunk_size,
                            n_iter=self.sei_n_iter,
                            support_rel=self.sei_support_rel,
                            fiedler_tau=self.sei_fiedler_tau,
                        )
                    elif self.sei_impurity_kind == "c_s":
                        rho = spectral_cs_impurity(
                            att,
                            bind,
                            eps=self.sei_eps,
                            chunk_size=self.sei_chunk_size,
                            n_iter=self.sei_n_iter,
                            proj_dim=self.sei_proj_dim,
                        )
                    else:
                        rho = spectral_graph_n8_impurity(
                            att,
                            bind,
                            eps=self.sei_eps,
                            chunk_size=self.sei_chunk_size,
                            n_iter=self.sei_n_iter,
                        )
                    c_sg = conf.detach()
                    imp = (c_sg * rho).sum(dim=-1) / (
                        c_sg.sum(dim=-1) + self.sei_eps
                    )
                    imp = imp.mean()
                    losses["loss_imp"] = imp
                    total_loss = total_loss + w_imp * imp

        return total_loss, losses

    def _sei_att_and_bind(
        self, outputs: Dict[str, Any]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Last-iter attention (B*T, S, N) and backbone tokens (B*T, N, D) for v40.

        Attention stays live. Bind tokens are detached here; the impurity
        helper also detaches Z after L2-norm.
        """
        proc = outputs.get("processor") or {}
        att = proc.get("state_attn_mask")
        if att is None:
            return None, None
        if att.ndim == 4:
            att = att.reshape(-1, att.shape[-2], att.shape[-1])
        elif att.ndim != 3:
            raise ValueError(
                f"slot_ent_impurity expected attention (B, S, N) or (B, T, S, N), "
                f"got {tuple(att.shape)}"
            )
        enc = outputs.get("encoder") or {}
        bind = enc.get("backbone_features")
        if bind is None:
            return att, None
        if bind.ndim == 4:
            bind = bind.reshape(-1, bind.shape[-2], bind.shape[-1])
        elif bind.ndim != 3:
            raise ValueError(
                f"slot_ent_impurity expected backbone (B, N, D) or (B, T, N, D), "
                f"got {tuple(bind.shape)}"
            )
        return att, bind.detach()

    def _predictive_recon_loss(self, outputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Predictive feature reconstruction: MSE(Dec(Pred(x_t)), F_{t+1}).

        Uses `state_predicted_pregate` for the same reason loss_dyn does: the term must
        supervise the predictor module itself. Going through the gated temporal mix would
        scale the predictor's gradient by g and let the loss lean on the gate instead of
        on motion. The gate handed to the decoder is detached likewise -- this loss trains
        the predictor/decoder/slots, never the gate.

        Only `n_transitions` random (t -> t+1) pairs are decoded per step to bound the
        extra decoder memory; the estimate stays unbiased over training.
        """
        proc = outputs.get("processor") or {}
        pred = proc.get("state_predicted_pregate")
        target = outputs.get("encoder", {}).get("backbone_features")
        if pred is None or target is None or pred.ndim != 4 or pred.shape[1] < 2:
            return None

        t_total = pred.shape[1]
        n = max(1, min(self.pred_n_transitions, t_total - 1))
        idx = torch.randperm(t_total - 1, device=pred.device)[:n]
        pred_slice = pred[:, idx]  # predictions made at t = idx, for frames idx + 1

        gate = self._active_mask
        if gate is not None and gate.ndim == 3 and gate.shape[:2] == pred.shape[:2]:
            decoder_output = self.decoder(pred_slice, gate[:, idx].detach())
        else:
            decoder_output = self.decoder(pred_slice)
        recon = decoder_output["reconstruction"]  # (B, n, F, D)
        tgt = target[:, idx + 1].detach()
        return F.mse_loss(recon, tgt)

    def _algebraic_slot_utility_loss(
        self,
        outputs: Dict[str, Any],
        gate: torch.Tensor,
        dec: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        mix = dec.get("reconstruction")
        if self.usage_redist_enabled:
            slot_recons = dec.get("slot_recons")
        else:
            slot_recons = dec.pop("slot_recons", None)
        masks = dec.get("masks")
        masks_ungated = dec.get("masks_ungated")
        mix_ungated = dec.get("reconstruction_ungated")
        target = outputs.get("encoder", {}).get("backbone_features")
        if mix is None or slot_recons is None or masks is None or target is None:
            return None
        if gate.ndim == 2:
            gate = gate.unsqueeze(1)
        if gate.ndim != 3:
            return None
        want_add = self.util_add_insert
        if want_add and masks_ungated is None:
            raise ValueError(
                "slot_utility.add=insert requires decoder masks_ungated "
                "(enable loss_featrec_ungated)"
            )
        # v44: drop on softmax(α). v42 has neither ungated tensor.
        use_ungated_drop = masks_ungated is not None and (
            want_add or mix_ungated is not None
        )
        proc = outputs.get("processor") or {}
        logits = proc.get("gate_logits")
        result = algebraic_slot_utility_loss(
            mix,
            slot_recons,
            masks,
            target,
            gate,
            margin=self.util_margin,
            min_mask=self.util_min_mask,
            masks_ungated=masks_ungated if use_ungated_drop else None,
            mix_ungated=mix_ungated if use_ungated_drop else None,
            add_insert=want_add,
            return_aux=True,
            teacher=self.util_teacher,
            ce_tau=self.util_ce_tau,
            logits=logits if self.util_teacher == "ce" else None,
        )
        loss, rel_delta, aux = result
        self._util_rel_delta = float(rel_delta.mean())
        self._util_c_mean = float(aux["cost"].mean()) if "cost" in aux else None
        pi = aux.get("pi")
        if pi is not None:
            q = pi.clamp_min(1e-8)
            self._util_pi_max = float(pi.amax(dim=-1).mean())
            self._util_pi_entropy = float(-(q * q.log()).sum(dim=-1).mean())
        else:
            self._util_pi_max = None
            self._util_pi_entropy = None
        if "rel_add" in aux:
            rel_add = aux["rel_add"]
            suppressed = aux.get("suppressed")
            if suppressed is not None and float(suppressed.sum()) > 0.0:
                self._util_rel_add = float((rel_add * suppressed).sum() / suppressed.sum())
                self._util_n_suppressed = float(suppressed.sum(dim=-1).mean())
            else:
                self._util_rel_add = float(rel_add.mean())
                self._util_n_suppressed = 0.0
        else:
            self._util_rel_add = None
            self._util_n_suppressed = None
        return loss

    def _slot_utility_loss(self, outputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Counterfactual slot-utility rent (marginal-utility replacement for gate_l1).

        mode='drop' (v27): per sample, one random gated slot is dropped and the batch is
        re-decoded under no_grad. The error increase is measured on that slot's decoder
        mask and charged as gate * sg(rent).

        mode='algebraic' (v42/v44/v45): all slots, no extra decode, per frame.
        MLPDecoder α/recon_s do not depend on z. v44 drop uses ungated mix_{-s};
        add restores σ-share on [σ-m]_+. teacher='rent': ⟨z, sg(c)⟩. teacher='ce'
        (v45): CE(softmax(([Δ↓]_+ + [Δ↑]_+)/τ), z).
        """
        gate = self._live_usage_z(outputs)
        if gate is None:
            gate = self._active_mask
        if gate is None or gate.dtype == torch.bool:
            return None
        proc = outputs.get("processor") or {}
        dec = outputs.get("decoder") or {}
        if self.util_mode == "algebraic":
            return self._algebraic_slot_utility_loss(outputs, gate, dec)

        if gate.ndim != 3:
            return None
        slots = proc.get("state")
        masks = dec.get("masks")
        recon = dec.get("reconstruction")
        target = outputs.get("encoder", {}).get("backbone_features")
        if slots is None or masks is None or recon is None or target is None:
            return None
        if slots.ndim != 4 or masks.ndim != 4:
            return None
        b, t, s, _ = slots.shape

        with torch.no_grad():
            err_full = (recon - target).pow(2).mean(-1)  # (B, T, F)
            g_bar = gate.float().mean(dim=1)  # (B, S)
            mask_share = masks.float().mean(dim=(1, 3))  # (B, S)
            cand = (g_bar > self.util_gate_min) & (mask_share > self.util_min_mask)

            drop_idx = torch.full((b,), -1, dtype=torch.long, device=slots.device)
            for bi in range(b):
                c = cand[bi].nonzero(as_tuple=False).flatten()
                if c.numel() > 0:
                    drop_idx[bi] = c[torch.randint(c.numel(), (1,), device=c.device)]
            valid = drop_idx >= 0
            if not bool(valid.any()):
                return None
            safe_idx = drop_idx.clamp_min(0)

            g_drop = gate.float().clone()
            g_drop[torch.arange(b, device=slots.device), :, safe_idx] = 0.0
            recon_drop = self.decoder(slots, g_drop)["reconstruction"]
            err_drop = (recon_drop - target).pow(2).mean(-1)  # (B, T, F)

            # territory of the dropped slot = its decoder mask (soft ownership weights)
            w = masks.float()[torch.arange(b, device=slots.device), :, safe_idx]  # (B, T, F)
            delta = ((err_drop - err_full) * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(1e-8)
            rel_delta = delta / err_full.mean(dim=(1, 2)).clamp_min(1e-8)  # (B,)
            rent = (1.0 - rel_delta / max(self.util_margin, 1e-8)).clamp(min=0.0, max=1.0)
            self._util_rel_delta = float(rel_delta[valid].mean())

        g_live = gate.float().mean(dim=1)  # (B, S), graph-connected to the mass branch
        g_sel = g_live[torch.arange(b, device=slots.device), safe_idx]
        return (g_sel * rent)[valid].mean()

    def _usage_redistribute_loss(self, outputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """v43 L_gate: train usage-head logits toward simplex-projected ∇J."""
        proc = outputs.get("processor") or {}
        dec = outputs.get("decoder") or {}
        mix = dec.get("reconstruction")
        slot_recons = dec.get("slot_recons")
        masks_ungated = dec.get("masks_ungated")
        target = outputs.get("encoder", {}).get("backbone_features")
        logits = proc.get("gate_logits")
        z = proc.get("gate_conf")
        if z is None:
            z = self._active_mask
        if (
            mix is None
            or slot_recons is None
            or masks_ungated is None
            or target is None
            or logits is None
            or z is None
            or not torch.is_floating_point(z)
        ):
            return None
        loss, d, _, v = usage_redistribute_direction(
            mix,
            slot_recons,
            masks_ungated,
            z,
            target,
            logits,
            lam=self.usage_redist_lambda,
            eps=self.usage_redist_eps,
            z_eps=self.usage_redist_z_eps,
        )
        self._usage_d_rms = float(d.float().pow(2).mean().sqrt())
        self._usage_v_rms = float(v.float().pow(2).mean().sqrt())
        self._usage_n_support = float((z.detach() > self.usage_redist_z_eps).float().sum(-1).mean())
        return loss

    def _dynamics_direction(self, outputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Cosine alignment between the predictor's step and the next-frame displacement.

        With x_t the corrector output the predictor is fed, Delta_t = Pred(x_t) - x_t is the
        predictor's residual step and d_t = x_{t+1} - x_t is the displacement it should have
        produced -- the same quantity the predictor receives as velocity one step earlier,

            L = sum_{t,s} w_{t,s} (1 - cos(Delta_{t,s}, d_{t,s})) / sum_{t,s} w_{t,s}

        Direction only, for two reasons. Cosine is scale free, so the predictor cannot lower
        the loss by shrinking Delta toward zero, which is exactly what an MSE against the
        small d_t would reward. And the target is detached, so the corrector cannot lower it
        by moving x_{t+1} toward whatever was predicted, which would collapse the
        representation into whichever states are easiest to predict rather than teach motion.
        """
        proc = outputs.get("processor") or {}
        x = proc.get("state")
        pre = proc.get("state_predicted_pregate")
        if x is None or pre is None:
            return None
        if x.ndim != 4 or x.shape[1] < 2 or pre.shape != x.shape:
            return None

        delta = (pre - x)[:, :-1]                       # (B, T-1, S, D)
        d = (x[:, 1:] - x[:, :-1]).detach()             # (B, T-1, S, D)
        cos = torch.nn.functional.cosine_similarity(delta, d, dim=-1, eps=1e-8)  # (B, T-1, S)

        w = torch.ones_like(cos)
        gate = self._active_mask
        if gate is not None and gate.dtype != torch.bool and gate.shape == x.shape[:3]:
            g = gate.float().detach()
            w = w * g[:, :-1]
            w = w * ((g[:, 1:] - g[:, :-1]).abs() < self.dyn_gate_jump_max).to(w.dtype)

        if 0.0 < self.dyn_min_disp_q < 1.0:
            nd = d.norm(dim=-1)
            thresh = torch.quantile(nd.flatten().float(), self.dyn_min_disp_q)
            w = w * (nd > thresh).to(w.dtype)

        denom = w.sum()
        if float(denom) <= 0.0:
            return None
        return ((1.0 - cos) * w).sum() / denom.clamp_min(1e-8)

    def _error_weighted_coverage(
        self, outputs: Dict[str, Any], gate: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Error-weighted uncovered mass under gated attention.

        c_f = max_s (g_s A_s,f)  [or sum_s],  w_f ∝ detach(||recon-target||^2) on top residual
        patches,  L = sum_f w_f (1 - c_f) averaged over batch/time.
        """
        proc = outputs.get("processor") or {}
        att = proc.get("state_attn_mask")
        if att is None:
            corrector = proc.get("corrector") or {}
            att = corrector.get("masks") if isinstance(corrector, Mapping) else None
        if att is None:
            return None

        # att: (B, S, F) or (B, T, S, F); gate: (B, S) or (B, T, S)
        if att.ndim == 3:
            att = att.unsqueeze(1)
        if gate.ndim == 2:
            gate = gate.unsqueeze(1)
        if att.ndim != 4 or gate.ndim != 3:
            return None
        if att.shape[:3] != gate.shape:
            # allow (B, T, F, S) -> (B, T, S, F)
            if att.shape[0] == gate.shape[0] and att.shape[1] == gate.shape[1] and att.shape[-1] == gate.shape[-1]:
                att = att.transpose(-1, -2)
            else:
                return None

        recon = outputs.get("decoder", {}).get("reconstruction")
        target = outputs.get("encoder", {}).get("backbone_features")
        if recon is None or target is None:
            return None
        # recon/target: (B, T, F, D) for video features
        if recon.ndim != 4 or target.ndim != 4:
            return None
        if recon.shape[:3] != att.shape[:2] + (att.shape[-1],):
            return None

        # per-patch MSE, detached weights
        err = (recon - target.detach()).pow(2).mean(dim=-1)  # (B, T, F)
        err = err.detach()

        top_frac = self.amc_gate_cov_top_frac
        if 0.0 < top_frac < 1.0:
            # keep only top residual patches per (B, T)
            f = err.shape[-1]
            k = max(1, int(round(top_frac * f)))
            # threshold = k-th largest
            topk_vals = torch.topk(err, k=k, dim=-1, largest=True).values
            thresh = topk_vals[..., -1:].detach()
            w = err * (err >= thresh).to(err.dtype)
        else:
            w = err
        w_sum = w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        w = w / w_sum

        g = gate.unsqueeze(-1)  # (B, T, S, 1)
        owned = g * att  # (B, T, S, F)
        if self.amc_gate_cov_use_max:
            c = owned.amax(dim=2)  # (B, T, F) — prefer a single owner
        else:
            c = owned.sum(dim=2).clamp(max=1.0)
        cov = (w * (1.0 - c)).sum(dim=-1).mean()
        return cov

    def _attention_centroid_repulsion(
        self, outputs: Dict[str, Any], gate: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Gated attention-centroid hinge repulsion.

        For each frame, build per-slot centroids c_s from soft attention over the patch
        grid (coords in [0, 1]^2), then
          L = mean_{i<j} [ g_i g_j * max(0, m - ||c_i-c_j||_2)^2 ]
        normalized by the sum of pair weights g_i g_j so scale stays gate-invariant.
        """
        proc = outputs.get("processor") or {}
        att = proc.get("state_attn_mask")
        if att is None:
            corrector = proc.get("corrector") or {}
            att = corrector.get("masks") if isinstance(corrector, Mapping) else None
        if att is None:
            return None

        # att: (B, S, F) or (B, T, S, F); gate: (B, S) or (B, T, S)
        if att.ndim == 3:
            att = att.unsqueeze(1)
        if gate.ndim == 2:
            gate = gate.unsqueeze(1)
        if att.ndim != 4 or gate.ndim != 3:
            return None
        if att.shape[:3] != gate.shape:
            if (
                att.shape[0] == gate.shape[0]
                and att.shape[1] == gate.shape[1]
                and att.shape[-1] == gate.shape[-1]
            ):
                att = att.transpose(-1, -2)
            else:
                return None

        b, t, s, f = att.shape
        h = int(f**0.5)
        w = h
        if h * w == f:
            # square patch grid -> (y, x) in [0, 1]
            ys = torch.linspace(0.0, 1.0, h, device=att.device, dtype=att.dtype)
            xs = torch.linspace(0.0, 1.0, w, device=att.device, dtype=att.dtype)
            try:
                gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            except TypeError:
                gy, gx = torch.meshgrid(ys, xs)
            pos = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1)  # (F, 2)
        else:
            # fallback: 1D index as a single spatial axis, pad second coord with 0.5
            idx = torch.linspace(0.0, 1.0, f, device=att.device, dtype=att.dtype)
            pos = torch.stack([idx, torch.full_like(idx, 0.5)], dim=-1)

        mass = att.sum(dim=-1).clamp_min(1e-8)  # (B, T, S)
        centroids = torch.einsum("btsf,fd->btsd", att, pos) / mass.unsqueeze(-1)

        # pairwise distances (B, T, S, S)
        diff = centroids.unsqueeze(3) - centroids.unsqueeze(2)
        dist = diff.pow(2).sum(dim=-1).clamp_min(0.0).sqrt()
        margin = float(self.amc_gate_rep_margin)
        hinge = (margin - dist).clamp_min(0.0).pow(2)

        pair_gate = gate.unsqueeze(3) * gate.unsqueeze(2)  # (B, T, S, S)
        tri = torch.triu(
            torch.ones(s, s, device=att.device, dtype=att.dtype), diagonal=1
        )  # i < j
        weighted = pair_gate * hinge * tri
        denom = (pair_gate * tri).sum(dim=(-1, -2)).clamp_min(1e-8)
        per_bt = weighted.sum(dim=(-1, -2)) / denom
        return per_bt.mean()

    def _frame_encoder(self) -> nn.Module:
        """The inner FrameEncoder (unwraps MapOverTime for video input)."""
        return self.encoder.module if hasattr(self.encoder, "module") else self.encoder

    def _apply_feature_curriculum(self) -> None:
        """Write this step's mix / window / bandwidth onto the frame encoder (train only)."""
        inner = self._frame_encoder()
        step = self.trainer.global_step
        self._featcur_s = feature_curriculum_mix(
            step, self.featcur_anneal_steps, self.featcur_schedule
        )
        if self.featcur_anneal == "window":
            w = feature_curriculum_window(
                step,
                self.featcur_anneal_steps,
                self.featcur_window_start,
                self.featcur_schedule,
            )
            if inner.feature_smoothing is not None:
                inner.feature_smoothing.window = w
            # w<=0 is raw: skip the module. Otherwise fully smoothed at the current radius
            # (no blend with raw -- the clock is w, not the v33 mix).
            inner.feature_smoothing_mix = 1.0 if w <= 0 else 0.0
            self._featcur_window = w
            self._featcur_h = 0.0
        elif self.featcur_anneal == "modes":
            h = feature_curriculum_bandwidth(
                step,
                self.featcur_anneal_steps,
                self.featcur_h_start,
                self.featcur_schedule,
                self.featcur_h_min,
            )
            if inner.feature_modes is not None:
                inner.feature_modes.bandwidth = h
            # h==0 is raw. Otherwise fully covered at the current bandwidth
            # (no blend with raw -- the clock is h, not the v33 mix).
            inner.feature_modes_mix = 1.0 if h <= 0.0 else 0.0
            self._featcur_h = h
            self._featcur_window = None
        elif self.featcur_anneal in ("ncut", "n8", "half", "rowcenter"):
            inner.feature_ncut_mix = self._featcur_s
            self._featcur_window = None
            self._featcur_h = 0.0
        else:
            inner.feature_smoothing_mix = self._featcur_s
            sm = inner.feature_smoothing
            self._featcur_window = None if sm is None else sm.window
            self._featcur_h = 0.0

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        if self.featcur_enabled:
            self._apply_feature_curriculum()
        outputs = self.forward(batch)
        if self.train_metrics or (
            self.visualize and self.trainer.global_step % self.visualize_every_n_steps == 0
        ):
            aux_outputs = self.aux_forward(batch, outputs)

        total_loss, losses = self.compute_loss(outputs)
        if len(losses) == 1:
            to_log = {"train/loss": total_loss}  # Log only total loss if only one loss configured
        else:
            to_log = {f"train/{name}": loss for name, loss in losses.items()}
            to_log["train/loss"] = total_loss

        if self.featcur_enabled:
            inner = self._frame_encoder()
            if self.featcur_anneal == "modes":
                to_log["train/featcur_mix"] = float(inner.feature_modes_mix)
                to_log["train/featcur_h"] = float(self._featcur_h)
                modes = inner.feature_modes
                if modes is not None and modes.last_c is not None:
                    to_log["train/featcur_C"] = modes.last_c.float().mean()
                    to_log["train/featcur_C_std"] = modes.last_c.float().std(unbiased=False)
            elif self.featcur_anneal in ("ncut", "n8", "half", "rowcenter"):
                to_log["train/featcur_mix"] = float(inner.feature_ncut_mix)
                to_log["train/featcur_beta"] = 1.0 - float(inner.feature_ncut_mix)
            else:
                to_log["train/featcur_mix"] = float(inner.feature_smoothing_mix)
            to_log["train/featcur_s"] = float(self._featcur_s)
            if self._featcur_window is not None:
                to_log["train/featcur_window"] = int(self._featcur_window)

        if self.attn_mass_enabled and self._active_mask is not None:
            # for soft gating this is the effective (summed-gate) active slot count
            gate = self._active_mask.float()  # (B, T, S) or (B, S)
            to_log["train/active_slots"] = (
                gate.mean(dim=-1).sum(dim=-1).mean()
                if gate.ndim >= 4
                else gate.sum(-1).mean()
            )
            gate_p_now = self._gate_threshold(True)
            if gate_p_now is not None:  # purity_weight is thresholdless
                to_log["train/gate_p"] = float(gate_p_now)
                to_log["train/gate_p_mult"] = float(self._annealed_p_mult(True))
            if getattr(self, "_gate_p_eff_mean", None) is not None:
                to_log["train/gate_p_eff"] = self._gate_p_eff_mean
            if self.amc_p_mode == "learnable_residual" and gate_p_now is not None:
                to_log["train/gate_p_sched"] = float(gate_p_now)
                to_log["train/gate_p_alpha"] = float(self._p_residual_alpha_eff(True))
                if self._gate_delta is not None:
                    to_log["train/gate_delta"] = self._gate_delta.float().detach().mean()
            if self.amc_gate_form == "logratio":
                to_log["train/gate_lambda"] = float(self._curriculum_lambda(True))
                to_log["train/gate_beta"] = float(self._gate_beta(True))
                to_log["train/gate_tau_log"] = float(self.amc_gate_tau_log)
            if self.amc_gate_form in ("logratio", "purity_weight") or (
                self.amc_state_gate_form in ("logratio", "purity_weight")
            ):
                if getattr(self, "_gate_conf_mean", None) is not None:
                    to_log["train/gate_conf"] = self._gate_conf_mean
            if getattr(self, "_identity_cos_mean", None) is not None:
                to_log["train/identity_cos"] = self._identity_cos_mean
            if self.amc_gate_mode in ("soft", "ste") or self.amc_gate_form == "purity_weight":
                if self.amc_gate_form == "linear":
                    # tau in mass units only parameterizes the linear gate
                    to_log["train/gate_tau"] = float(self._gate_tau(True))
                # Distinguish "one strong winner" vs "all slots weakly on":
                # sum(g) alone cannot tell these apart.
                g_max = gate.amax(dim=-1)  # (...,)
                g_sorted = gate.sort(dim=-1, descending=True).values
                g_top2 = g_sorted[..., : min(2, g_sorted.shape[-1])].sum(-1)
                # entropy of per-example gate distribution (normalized over slots)
                q = gate / gate.sum(-1, keepdim=True).clamp_min(1e-8)
                ent = -(q * q.clamp_min(1e-8).log()).sum(-1)  # (...,)
                to_log["train/gate_max"] = g_max.mean()
                to_log["train/gate_top2"] = g_top2.mean()
                to_log["train/gate_entropy"] = ent.mean()
                # hard-ish fraction: slots with g > 0.5
                to_log["train/gate_n_half"] = (gate > 0.5).float().sum(-1).mean()
            if self.amc_purity_q_end is not None:
                to_log["train/purity_q"] = float(self._purity_q(True))
            state_p = self._state_gate_threshold(True)
            if state_p is not None:
                # gate_state_slots vs active_slots is the whole point of the split: the
                # first should stay well below the second once p has annealed.
                to_log["train/gate_state_p"] = float(state_p)
            if (
                state_p is not None or self.amc_state_gate_form is not None
            ) and getattr(self, "_state_gate_mean", None) is not None:
                to_log["train/gate_state_slots"] = self._state_gate_mean * self.n_slots

        if self.pred_weight > 0.0:
            to_log["train/pred_w_eff"] = self.pred_weight * self._aux_ramp(self.pred_ramp)
        if self.util_weight > 0.0:
            to_log["train/util_w_eff"] = self.util_weight * self._aux_ramp(self.util_ramp)
            if self._util_rel_delta is not None:
                to_log["train/util_rel_delta"] = self._util_rel_delta
            if self._util_rel_add is not None:
                to_log["train/util_rel_add"] = self._util_rel_add
            if self._util_c_mean is not None:
                to_log["train/util_c_mean"] = self._util_c_mean
            if self._util_n_suppressed is not None:
                to_log["train/util_n_suppressed"] = self._util_n_suppressed
            if self.util_add_insert:
                to_log["train/util_add_insert"] = 1.0
            if self.util_teacher == "ce":
                to_log["train/util_ce_tau"] = float(self.util_ce_tau)
                if self._util_pi_max is not None:
                    to_log["train/util_pi_max"] = self._util_pi_max
                if self._util_pi_entropy is not None:
                    to_log["train/util_pi_entropy"] = self._util_pi_entropy
        if self.util_psi_weight > 0.0:
            to_log["train/util_psi_w"] = float(self.util_psi_weight)
        if getattr(self, "usage_redist_enabled", False):
            to_log["train/usage_lambda"] = float(self.usage_redist_lambda)
            if self._usage_d_rms is not None:
                to_log["train/usage_d_rms"] = self._usage_d_rms
            if self._usage_v_rms is not None:
                to_log["train/usage_v_rms"] = self._usage_v_rms
            if self._usage_n_support is not None:
                to_log["train/usage_n_support"] = self._usage_n_support
        if self.sei_enabled:
            w_ent, w_imp, lam = self._sei_weights(True)
            to_log["train/sei_lambda"] = float(lam)
            to_log["train/sei_w_ent"] = float(w_ent)
            to_log["train/sei_w_imp"] = float(w_imp)

        if self.train_metrics:
            for key, metric in self.train_metrics.items():
                values = metric(**batch, **outputs, **aux_outputs)
                self._add_metric_to_log(to_log, f"train/{key}", values)
                metric.reset()
        self.log_dict(to_log, on_step=True, on_epoch=False, batch_size=outputs["batch_size"])

        del outputs  # Explicitly delete to save memory

        if (
            self.visualize
            and self.trainer.global_step % self.visualize_every_n_steps == 0
            and self.global_rank == 0
        ):
            self._log_inputs(
                batch[self.input_key],
                {key: aux_outputs[f"{key}_hard"] for key in self.mask_keys_to_visualize},
                mode="train",
            )
            self._log_masks(aux_outputs, self.mask_keys_to_visualize, mode="train")

        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        if "batch_padding_mask" in batch:
            batch = self._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                return

        cycle = self.cyclic_inference

        outputs = self.forward(batch, train=False, cycle=cycle)
        aux_outputs = self.aux_forward(batch, outputs)


        total_loss, losses = self.compute_loss(outputs)
        if len(losses) == 1:
            to_log = {"val/loss": total_loss}  # Log only total loss if only one loss configured
        else:
            to_log = {f"val/{name}": loss for name, loss in losses.items()}
            to_log["val/loss"] = total_loss
        to_log = {k: self._as_log_float(v) for k, v in to_log.items()}
        self._accumulate_val_losses(to_log, int(outputs["batch_size"]))

        if self.val_metrics:
            for metric in self.val_metrics.values():
                metric.update(**batch, **outputs, **aux_outputs)

        # Progress bar only. Logger output is written in on_validation_end from
        # python floats so Lightning cannot reprint the first val's _computed.
        self.log_dict(
            to_log,
            on_step=False,
            on_epoch=True,
            batch_size=outputs["batch_size"],
            prog_bar=True,
            logger=False,
        )

        if self.visualize and batch_idx == 0 and self.global_rank == 0:
            masks_to_vis = {
                key: aux_outputs[f"{key}_vis_hard"] for key in self.mask_keys_to_visualize
            }
            if batch["segmentations"].shape[-2:] != batch[self.input_key].shape[-2:]:
                masks_to_vis["segmentations"] = self.mask_resizers["segmentation"](
                    batch["segmentations"], batch[self.input_key]
                )
            else:
                masks_to_vis["segmentations"] = batch["segmentations"]
            self._log_inputs(
                batch[self.input_key],
                masks_to_vis,
                mode="val",
            )
            self._log_masks(aux_outputs, self.mask_keys_to_visualize, mode="val")

    def on_validation_epoch_start(self):
        self._val_loss_acc = {}
        self._val_loss_weight = 0.0
        self._pending_val_scalars = None

    def validation_epoch_end(self, outputs):
        to_log: Dict[str, float] = {}
        if self._val_loss_weight > 0.0:
            w = self._val_loss_weight
            to_log = {k: v / w for k, v in self._val_loss_acc.items()}
        if self.val_metrics:
            for key, metric in self.val_metrics.items():
                self._add_metric_to_log(to_log, f"val/{key}", metric.compute())
                metric.reset()
        to_log = {k: self._as_log_float(v) for k, v in to_log.items()}
        self._pending_val_scalars = to_log
        if to_log:
            self.log_dict(to_log, prog_bar=True, logger=False, on_step=False, on_epoch=True)

    def on_validation_end(self):
        scalars = self._pending_val_scalars
        self._pending_val_scalars = None
        self._val_loss_acc = {}
        self._val_loss_weight = 0.0
        if not scalars or self.trainer.sanity_checking:
            return
        if self.trainer.is_global_zero:
            for logger in self.trainer.loggers:
                logger.log_metrics(dict(scalars), step=int(self.trainer.global_step))
                logger.save()

    def _accumulate_val_losses(self, to_log: Dict[str, float], batch_size: int) -> None:
        bs = max(int(batch_size), 1)
        for name, value in to_log.items():
            self._val_loss_acc[name] = self._val_loss_acc.get(name, 0.0) + float(value) * bs
        self._val_loss_weight += bs

    @staticmethod
    def _as_log_float(value: Any) -> float:
        if torch.is_tensor(value):
            return float(value.detach().float().mean().cpu())
        return float(value)

    @staticmethod
    def _add_metric_to_log(
        log_dict: Dict[str, Any], name: str, values: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ):
        if isinstance(values, dict):
            for k, v in values.items():
                log_dict[f"{name}/{k}"] = v
        else:
            log_dict[name] = values


    def _log_inputs(
        self,
        inputs: torch.Tensor,
        masks_by_name: Dict[str, torch.Tensor],
        mode: str,
        step: Optional[int] = None,
    ):
        denorm = Denormalize(input_type=self.input_key)
        if step is None:
            step = self.trainer.global_step

        if self.input_key == "video":
            video = torch.stack([denorm(video) for video in inputs])
            self._log_video(f"{mode}/{self.input_key}", video, global_step=step)
            for mask_name, masks in masks_by_name.items():
                video_with_masks = visualizations.mix_videos_with_masks(video, masks)
                self._log_video(
                    f"{mode}/video_with_{mask_name}",
                    video_with_masks,
                    global_step=step,
                )
        elif self.input_key == "image":
            image = denorm(inputs)
            self._log_images(f"{mode}/{self.input_key}", image, global_step=step)
            for mask_name, masks in masks_by_name.items():
                image_with_masks = visualizations.mix_images_with_masks(image, masks)
                self._log_images(
                    f"{mode}/image_with_{mask_name}",
                    image_with_masks,
                    global_step=step,
                )
        else:
            raise ValueError(f"input_type should be 'image' or 'video', but got '{self.input_key}'")

    def _log_masks(
        self,
        aux_outputs,
        mask_keys=("decoder_masks",),
        mode="val",
        types: tuple = ("frames",),
        step: Optional[int] = None,
    ):
        if step is None:
            step = self.trainer.global_step
        for mask_key in mask_keys:
            if mask_key in aux_outputs:
                masks = aux_outputs[mask_key]
                if self.input_key == "video":
                    _, f, n_obj, H, W = masks.shape
                    first_masks = masks[0].permute(1, 0, 2, 3)
                    first_masks_inverted = 1 - first_masks.reshape(n_obj, f, 1, H, W)
                    self._log_video(
                        f"{mode}/{mask_key}",
                        first_masks_inverted,
                        global_step=step,
                        n_examples=n_obj,
                        types=types,
                    )
                elif self.input_key == "image":
                    _, n_obj, H, W = masks.shape
                    first_masks_inverted = 1 - masks[0].reshape(n_obj, 1, H, W)
                    self._log_images(
                        f"{mode}/{mask_key}",
                        first_masks_inverted,
                        global_step=step,
                        n_examples=n_obj,
                    )
                else:
                    raise ValueError(
                        f"input_type should be 'image' or 'video', but got '{self.input_key}'"
                    )

    def _log_video(
        self,
        name: str,
        data: torch.Tensor,
        global_step: int,
        n_examples: int = 8,
        max_frames: int = 8,
        types: tuple = ("frames",),
    ):
        data = data[:n_examples]
        logger = self._get_tensorboard_logger()

        if logger is not None:
            if "video" in types:
                logger.experiment.add_video(f"{name}/video", data, global_step=global_step)
            if "frames" in types:
                _, num_frames, _, _, _ = data.shape
                num_frames = min(max_frames, num_frames)
                data = data[:, :num_frames]
                data = data.flatten(0, 1)
                logger.experiment.add_image(
                    f"{name}/frames", make_grid(data, nrow=num_frames), global_step=global_step
                )

    def _save_video(self, name: str, data: torch.Tensor, global_step: int):
        assert (
            data.shape[0] == 1
        ), f"Only single videos saving are supported, but shape is: {data.shape}"
        data = data.cpu().numpy()[0].transpose(0, 2, 3, 1)
        data_dir = self.save_data_dir / name
        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(data_dir / f"{global_step}.npy", data)

    def _log_images(
        self,
        name: str,
        data: torch.Tensor,
        global_step: int,
        n_examples: int = 8,
    ):
        n_examples = min(n_examples, data.shape[0])
        data = data[:n_examples]
        logger = self._get_tensorboard_logger()

        if logger is not None:
            logger.experiment.add_image(
                f"{name}/images", make_grid(data, nrow=n_examples), global_step=global_step
            )

    @staticmethod
    def _remove_padding(
        batch: Dict[str, Any], padding_mask: torch.Tensor
    ) -> Optional[Dict[str, Any]]:
        if torch.all(padding_mask):
            # Batch consists only of padding
            return None

        mask = ~padding_mask
        mask_as_idxs = torch.arange(len(mask))[mask.cpu()]

        output = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                output[key] = value[mask]
            elif isinstance(value, list):
                output[key] = [value[idx] for idx in mask_as_idxs]

        return output

    def _get_tensorboard_logger(self):
        if self.loggers is not None:
            for logger in self.loggers:
                if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
                    return logger
        else:
            if isinstance(self.logger, pl.loggers.tensorboard.TensorBoardLogger):
                return self.logger

    def on_load_checkpoint(self, checkpoint):
        # Reset timer during loading of the checkpoint
        # as timer is used to track time from the start
        # of the current run.
        if "callbacks" in checkpoint and "Timer" in checkpoint["callbacks"]:
            checkpoint["callbacks"]["Timer"]["time_elapsed"] = {
                "train": 0.0,
                "sanity_check": 0.0,
                "validate": 0.0,
                "test": 0.0,
                "predict": 0.0,
            }

    def load_weights_from_checkpoint(
        self, checkpoint_path: str, module_mapping: Optional[Dict[str, str]] = None
    ):
        """Load weights from a checkpoint into the specified modules."""
        checkpoint = torch.load(checkpoint_path)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        if module_mapping is None:
            module_mapping = {
                key.split(".")[0]: key.split(".")[0]
                for key in checkpoint
                if hasattr(self, key.split(".")[0])
            }

        for dest_module, source_module in module_mapping.items():
            try:
                module = utils.read_path(self, dest_module)
            except ValueError:
                raise ValueError(f"Module {dest_module} could not be retrieved from model") from None

            state_dict = {}
            for key, weights in checkpoint.items():
                if key.startswith(source_module):
                    if key != source_module:
                        key = key[len(source_module + ".") :]  # Remove prefix
                    state_dict[key] = weights
            if len(state_dict) == 0:
                raise ValueError(
                    f"No weights for module {source_module} found in checkpoint {checkpoint_path}."
                )

            module.load_state_dict(state_dict)
