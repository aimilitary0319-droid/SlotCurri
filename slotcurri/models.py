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

    input_type = model_config.get("input_type", "image")
    if input_type == "image":
        processor = modules.LatentProcessor(grouper, predictor=None)
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
            )
        else:
            processor = modules.LatentProcessor(grouper, predictor)
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
    ):
        super().__init__()
        # bool: legacy on/off (backward sweep anchored at the last frame). A string
        # selects the anchored re-inference variants in ScanOverTime ("evidence" = EABI,
        # "random" = its control); eval scripts set this directly on a loaded model.
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
        # for delta to be measurable
        self.util_gate_min = float(su.get("gate_min", 0.5))
        self.util_min_mask = float(su.get("min_mask", 0.01))
        self.util_ramp = str(su.get("ramp", "lambda")).lower()
        self.util_start_step = int(su.get("start_step", 0))
        if self.util_ramp not in ("lambda", "none"):
            raise ValueError(f"slot_utility.ramp must be 'lambda' or 'none', got {self.util_ramp!r}")

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
        if self.featcur_anneal not in ("mix", "window", "modes", "ncut"):
            raise ValueError(
                "feature_curriculum.anneal must be 'mix', 'window', 'modes' or 'ncut', "
                f"got {self.featcur_anneal!r}"
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
        self.featcur_apply = str(fc.get("apply", "key" if self.featcur_anneal == "ncut" else "tokens")).lower()
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
            elif self.featcur_anneal == "ncut":
                inner.feature_ncut = modules.NcutRelationalLeveling(
                    chunk_size=int(fc.get("chunk_size", 8)),
                    eps=float(fc.get("eps", 1e-6)),
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
        #   "purity_weight" (v32): g = sg(c) itself, with c the ownership purity
        #               (conf_kind purity/purity_sharp). Thresholdless: no p schedule, no
        #               tau, no beta -- the decoder sees softmax(alpha + log c) and the
        #               temporal mix uses c / max(c). Self-annealing (uniform untrained c
        #               is cancelled by the renorm/max-norm), replacing the curriculum.
        self.amc_gate_form = str(amc.get("gate_form", "linear")).lower()
        if self.amc_gate_form not in ("linear", "logratio", "purity_weight"):
            raise ValueError(
                f"attn_mass_curriculum.gate_form must be 'linear', 'logratio' or "
                f"'purity_weight', got {self.amc_gate_form!r}"
            )
        # Confidence definition for the logratio gate's second branch (v29):
        #   "entropy": c = 1 - H/log F over the gamma-sharpened attention (v26 default).
        #              Spatial concentration; penalizes large objects by construction
        #              (H grows with log(object size)).
        #   "purity" : c = sum A^2 / sum A over the RAW attention -- the attention-weighted
        #              mean of the slot's own per-patch share. Direct, size-invariant
        #              ownership quality.
        #   "purity_sharp": the same statistic on the gamma-sharpened attention the mass
        #              branch uses (one distribution, two moments). Best ghost-vs-small
        #              separation on v20 @ 100k (event_analysis/conf_vs_purity_probe.py).
        # Detached in every case (anti-gaming). Not to be confused with the purity_q
        # OR-rescue, which can only OPEN gates and bypasses the evidence score entirely;
        # this is the multiplicative evidence branch.
        self.amc_conf_kind = str(amc.get("conf_kind", "entropy")).lower()
        if self.amc_conf_kind not in ("entropy", "purity", "purity_sharp"):
            raise ValueError(
                f"attn_mass_curriculum.conf_kind must be 'entropy', 'purity' or "
                f"'purity_sharp', got {self.amc_conf_kind!r}"
            )
        # purity_weight is DEFINED as ownership weighting; the conf_kind default
        # ("entropy") would silently weight by spatial concentration instead, so an
        # explicit purity choice is required.
        if self.amc_gate_form == "purity_weight" and self.amc_conf_kind not in (
            "purity",
            "purity_sharp",
        ):
            raise ValueError(
                "attn_mass_curriculum.gate_form='purity_weight' requires conf_kind "
                f"'purity' or 'purity_sharp', got {self.amc_conf_kind!r}"
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
        self._gate_p_eff_mean = None
        self._state_gate_mean = None
        self._gate_delta = None  # stashed for residual L2 (may be graph-connected)
        self._gate_conf_mean = None  # mean assignment confidence (logratio gate, logging)
        self._util_rel_delta = None  # mean rel_delta of sampled slots (slot_utility, logging)

    def _p_residual_alpha_eff(self, train: bool) -> float:
        """Warm up residual strength so early coarse curriculum stays near v10."""
        alpha = self.amc_p_residual_alpha
        if not train or self.amc_p_residual_warmup_steps <= 0:
            return alpha
        step = self.trainer.global_step
        return alpha * min(1.0, float(step) / float(self.amc_p_residual_warmup_steps))

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
        if not self.attn_mass_enabled or self.amc_state_p_mult is None:
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
                **{k: v for k, v in processor_kwargs.items() if k != "cycle"},
            )
            slots = processor_output["state"]
            active_mask = processor_output.get("active_mask")
            self._active_mask = active_mask
            gate_conf = processor_output.get("gate_conf")
            self._gate_conf_mean = (
                float(gate_conf.float().mean().detach()) if gate_conf is not None else None
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
            decoder_output = self.decoder(slots, active_mask)
        else:
            self._active_mask = None
            self._gate_p_eff_mean = None
            self._gate_delta = None
            self._state_gate_mean = None
            self._gate_conf_mean = None
            processor_output = self.processor(
                slots_initial, features, **processor_kwargs
            )
            slots = processor_output["state"]
            decoder_output = self.decoder(slots)
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

            if (
                name == 'loss_ss'
                and self.attn_mass_enabled
                and self.amc_contrastive_gate
                and self._active_mask is not None
            ):
                # restrict the slot-slot contrastive loss to active slots
                losses[name] = loss_fn(prediction, target, active_mask=self._active_mask)
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

        return total_loss, losses

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

    def _slot_utility_loss(self, outputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Counterfactual slot-utility rent (marginal-utility replacement for gate_l1).

        Per sample, one random gated slot is dropped and the batch is re-decoded under
        no_grad (the counterfactual carries no activations, so this pass is cheap). The
        error increase is measured ON THE DROPPED SLOT'S OWN TERRITORY (decoder-mask
        weighted) and normalized by the sample's mean per-patch error, which removes the
        size bias that made the constant rent evict small-object slots:

            rel_delta = E_mask[err_drop - err_full] / E[err_full]
            rent      = clamp(1 - rel_delta / margin, 0, 1)
            loss      = mean_b [ gate(dropped slot) * sg(rent) ]

        Only the live gate carries gradient (through the sigmoid into the mass branch),
        matching the detach discipline of the confidence branch and gate_cov's error
        weights. Bool (hard) gates carry no gradient, so the term is skipped then.
        """
        gate = self._active_mask
        if gate is None or gate.dtype == torch.bool or gate.ndim != 3:
            return None
        proc = outputs.get("processor") or {}
        dec = outputs.get("decoder") or {}
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
        elif self.featcur_anneal == "ncut":
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
            elif self.featcur_anneal == "ncut":
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
            to_log["train/active_slots"] = gate.sum(-1).mean()
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
            if self.amc_gate_form in ("logratio", "purity_weight"):
                if getattr(self, "_gate_conf_mean", None) is not None:
                    to_log["train/gate_conf"] = self._gate_conf_mean
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
                if getattr(self, "_state_gate_mean", None) is not None:
                    to_log["train/gate_state_slots"] = self._state_gate_mean * self.n_slots

        if self.pred_weight > 0.0:
            to_log["train/pred_w_eff"] = self.pred_weight * self._aux_ramp(self.pred_ramp)
        if self.util_weight > 0.0:
            to_log["train/util_w_eff"] = self.util_weight * self._aux_ramp(self.util_ramp)
            if self._util_rel_delta is not None:
                to_log["train/util_rel_delta"] = self._util_rel_delta

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

        if self.val_metrics:
            for metric in self.val_metrics.values():
                metric.update(**batch, **outputs, **aux_outputs)

        self.log_dict(
            to_log, on_step=False, on_epoch=True, batch_size=outputs["batch_size"], prog_bar=True
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

    def validation_epoch_end(self, outputs):
        if self.val_metrics:
            to_log = {}
            for key, metric in self.val_metrics.items():
                self._add_metric_to_log(to_log, f"val/{key}", metric.compute())
                metric.reset()
            self.log_dict(to_log, prog_bar=True)

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
