import math
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import nn

from slotcurri.utils import make_build_fn


@make_build_fn(__name__, "video module")
def build(config, name: str, **kwargs):
    pass  # No special module building needed


def _max_norm(gate: Optional[torch.Tensor], enabled: bool) -> Optional[torch.Tensor]:
    """g / max_s(g), so the winning slot always advances fully in the temporal mix.

    Without this the early curriculum can put p above every slot's mass, leaving no
    slot with a gate large enough to advance at all. Bool (hard) gates and the
    disabled case pass through untouched.
    """
    if not enabled or gate is None or gate.dtype == torch.bool:
        return gate
    return gate / gate.amax(dim=-1, keepdim=True).clamp_min(1e-8)


class LatentProcessor(nn.Module):
    """Updates latent state based on inputs and state and predicts next state."""

    def __init__(
        self,
        corrector: nn.Module,
        predictor: Optional[nn.Module] = None,
        state_key: str = "slots",
        first_step_corrector_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.corrector = corrector # slot attention
        self.predictor = predictor # transformer encoder
        self.state_key = state_key
        if first_step_corrector_args is not None:
            self.first_step_corrector_args = first_step_corrector_args
        else:
            self.first_step_corrector_args = None
        # Velocity conditioning turns itself on when the predictor was built with vel_dim,
        # so a single config knob on the predictor controls the whole path.
        self.predictor_takes_vel = predictor is not None and any(
            getattr(block, "vel_proj", None) is not None
            for block in getattr(predictor, "blocks", [])
        )

    def forward(
        self, state: torch.Tensor, inputs: Optional[torch.Tensor], time_step: Optional[int] = None,
        onetoone: bool = False, gate_p: Optional[float] = None, default_idx: Any = 0,
        gate_mode: str = "hard", gate_tau: Optional[float] = None,
        mass_gamma: float = 1.0, purity_q: Optional[float] = None,
        purity_tau: Optional[float] = None,
        gate_p_state: Optional[float] = None,
        state_max_norm: bool = False,
        predictor_ungated: bool = False,
        prev_state: Optional[torch.Tensor] = None,
        p_mode: str = "absolute",
        median_ema_prev: Optional[torch.Tensor] = None,
        median_ema_momentum: float = 0.9,
        p_residual_net: Optional[nn.Module] = None,
        p_residual_alpha: float = 0.0,
        gate_form: str = "linear",
        gate_beta: Optional[float] = None,
        gate_tau_log: Optional[float] = None,
        conf_kind: str = "entropy",
        gate_detach: bool = False,
        key_features: Optional[torch.Tensor] = None,
        purity_normalize: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # state: batch x n_slots x slot_dim (1 7 64)
        if onetoone:
            # state: batch x n_slots x slot_dim (1 30 1369 64)
            assert state.ndim == 3
        else:
            # state: batch x n_slots x slot_dim (1 30 1369)
            assert state.ndim == 3
        # inputs: batch x n_inputs x input_dim (1 30 1369 64)
        assert inputs.ndim == 3
        if inputs is not None:
            corr_kwargs: Dict[str, Any] = {}
            if time_step == 0 and self.first_step_corrector_args:
                corr_kwargs.update(self.first_step_corrector_args)
            if key_features is not None:
                corr_kwargs["key_features"] = key_features
            corrector_output = self.corrector(state, inputs, **corr_kwargs)
            updated_state = corrector_output[self.state_key]
            state_attn_mask = corrector_output['masks'] if 'masks' in corrector_output else None
        else:
            # Run predictor without updating on current inputs
            corrector_output = None
            updated_state = state
            state_attn_mask = None

        # --- attention-mass gating: hold non-active slots' priors at their incoming state ---
        # `state_attn_mask` is the last-iteration softmax-over-slots attention (B, S, F).
        # Per-slot attention mass = fraction of patches claimed by that slot.
        # `gate_mode`:
        #   "hard" -> binary active/dormant (active_mask is bool); dormant slots removed.
        #   "soft" -> continuous gate g = sigmoid((mass - p) / tau) in [0, 1] (float);
        #            a small tau (annealed over training) sharpens toward the hard decision.
        #   "ste"  -> straight-through: forward uses the hard 0/1 gate (true dormancy) while
        #            the backward pass uses the sigmoid surrogate gradient (differentiable).
        active_mask = None
        state_mask = None
        mass_median_ema = None
        gate_p_eff = None
        gate_delta = None
        gate_conf = None
        # gate_form="purity_weight" has no threshold, so it activates on the form alone
        # (gate_p arrives as None from the model, every other form still requires it).
        gating_on = gate_p is not None or gate_form == "purity_weight"
        if gating_on and state_attn_mask is not None:
            att = state_attn_mask  # (B, S, F), per-patch softmax over slots
            if mass_gamma is not None and float(mass_gamma) != 1.0:
                # gamma-sharpened mass: per-patch attention raised to gamma and renormalized.
                # gamma=1 -> plain attention mass (unchanged); gamma -> inf -> per-patch argmax
                # winner share. Sharpening suppresses "always-second" ghost slots that gather
                # mass without ever winning a patch, while keeping the score differentiable
                # and normalized (sums to 1 over slots, mean 1/S -> p_mult semantics hold).
                att_sharp = att.pow(float(mass_gamma))
                att_sharp = att_sharp / att_sharp.sum(dim=1, keepdim=True).clamp_min(1e-8)
            else:
                att_sharp = att
            mass_frac = att_sharp.sum(dim=-1) / att_sharp.shape[-1]  # (B, S)

            # --- assignment confidence (evidence-aware gate, gate_form="logratio") ---
            # A low mass does not by itself mean "inactive": a small object is low-mass too.
            # What separates a small active object from a diffuse ghost slot is how the
            # (sharpened) attention is distributed over the features the slot does claim:
            #   P_{s,f} = A~_{s,f} / sum_f A~_{s,f}   (per-slot distribution over features)
            #   H_s     = -sum_f P log P,   c_s = 1 - H_s / log F   in [0, 1]
            # small+peaked -> c high, diffuse -> c low. Detached (sg) so the model cannot
            # open its gate by artificially sharpening attention; the mass branch keeps its
            # gradient, which is the path featrec is supposed to shape.
            ck = str(conf_kind).lower()
            if gate_form in ("logratio", "purity_weight"):
                if ck in ("purity", "purity_sharp"):
                    # Ownership quality instead of spatial concentration (v29): the
                    # attention-weighted mean of the slot's own per-patch share,
                    #   c_s = sum_f A_{s,f}^2 / sum_f A_{s,f}  in (0, 1].
                    # Size-invariant (a fully-owned object scores ~1 whether it covers 20
                    # patches or 800, where the entropy form gives the large one c ~ 0.1).
                    # "purity" computes it on the raw attention, "purity_sharp" on the same
                    # gamma-sharpened tensor as the mass branch (one distribution, two
                    # moments). On v20 @ 100k the sharp form separates ghosts from small
                    # objects best (conf-only AUC 0.999 vs 0.997 raw vs 0.647 entropy; see
                    # event_analysis/conf_vs_purity_probe.py). Detached like the entropy
                    # form so the model cannot open its gate by sharpening.
                    src = att if ck == "purity" else att_sharp
                    gate_conf = (
                        (src * src).sum(dim=-1) / src.sum(dim=-1).clamp_min(1e-8)
                    ).clamp(min=0.0, max=1.0).detach()
                else:
                    p_feat = att_sharp / att_sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    ent = -(p_feat * p_feat.clamp_min(1e-8).log()).sum(dim=-1)  # (B, S)
                    log_f = math.log(max(att_sharp.shape[-1], 2))
                    gate_conf = (1.0 - ent / log_f).clamp(min=0.0, max=1.0).detach()

            # Threshold p:
            #   absolute (default): gate_p is the annealed absolute mass threshold
            #   median_ema: gate_p is annealed p_mult; p = sg(EMA[median(m)]) * p_mult
            #               (recovers absolute p≈p_mult/S when masses are near-uniform)
            #   learnable_residual: gate_p is absolute p_sched;
            #               p = p_sched * (1 + alpha * tanh(f(sg[m])))
            p_mode_l = str(p_mode).lower()
            if gate_form == "purity_weight":
                p_use = None  # thresholdless form: p_mode machinery is inert
            elif p_mode_l == "median_ema":
                med = mass_frac.median(dim=-1).values.detach()  # (B,)
                mom = float(median_ema_momentum)
                mom = min(max(mom, 0.0), 1.0)
                if median_ema_prev is None:
                    mass_median_ema = med
                else:
                    mass_median_ema = mom * median_ema_prev.detach() + (1.0 - mom) * med
                # (B, 1) so it broadcasts over slots; detached scene statistic
                p_use = (mass_median_ema * float(gate_p)).unsqueeze(-1)
            elif p_mode_l == "learnable_residual" and p_residual_net is not None:
                p_sched = float(gate_p)
                # features detached; delta still receives loss grad via L_delta and via g(m-p)
                gate_delta = p_residual_net(mass_frac.detach())  # (B, 1), in (-1, 1)
                alpha = float(p_residual_alpha)
                p_use = p_sched * (1.0 + alpha * gate_delta)
                p_use = p_use.clamp(min=1e-6, max=0.999)
            else:
                p_use = gate_p  # scalar absolute threshold
            gate_p_eff = p_use

            purity = None
            if purity_q is not None:
                # size-invariant "ownership quality": attention-weighted mean of the slot's own
                # per-patch attention (sum A^2 / sum A, in (0, 1]). High for slots that dominantly
                # own the (few) patches they claim; low for diffuse ghost slots. Used as an OR
                # rescue so small-but-cleanly-owned objects survive the mass threshold.
                purity = (att * att).sum(dim=-1) / att.sum(dim=-1).clamp_min(1e-8)  # (B, S)
            default_list = [default_idx] if isinstance(default_idx, int) else list(default_idx)

            # --- evidence score for the log-ratio gate ---
            # log r = beta * log(m + eps) + (1 - beta) * log(sg(c) + eps). Independent of the
            # threshold, so computed once and shared by the decoder gate and the state gate.
            # beta=1 (or gate_beta None) degenerates to pure coverage in log domain.
            log_r = None
            if gate_form == "logratio":
                log_eps = 1e-6
                beta = 1.0 if gate_beta is None else float(gate_beta)
                log_r = beta * (mass_frac + log_eps).log()
                if gate_conf is not None and beta < 1.0:
                    log_r = log_r + (1.0 - beta) * (gate_conf + log_eps).log()

            def build_gate(p_thresh):
                """Gate from the evidence score, thresholded at `p_thresh`.

                Factored out so the decoder gate and the state gate below differ only in
                their threshold and cannot drift apart.

                gate_form:
                  "linear"  : score = (m - p) / tau            (mass units; v6..v25)
                  "logratio": score = (log r - log p) / tau_g  (relative evidence; scale-
                              invariant, so r=0.02,p=0.01 and r=0.2,p=0.1 gate identically,
                              and the gate stays discriminative wherever p sits w.r.t. the
                              mass distribution instead of saturating like the linear form)
                """
                if gate_form == "logratio":
                    if torch.is_tensor(p_thresh):
                        log_p = (p_thresh + 1e-6).log()
                    else:
                        log_p = math.log(float(p_thresh) + 1e-6)
                    tau_g = gate_tau_log if (gate_tau_log is not None and gate_tau_log > 0) else 0.5
                    score = (log_r - log_p) / tau_g  # (B, S)
                    exceeds = log_r >= log_p
                else:
                    tau = gate_tau if (gate_tau is not None and gate_tau > 0) else 1e-2
                    score = (mass_frac - p_thresh) / tau  # (B, S)
                    exceeds = mass_frac >= p_thresh
                if gate_mode in ("soft", "ste"):
                    soft = torch.sigmoid(score)  # (B, S) in (0, 1)
                    if purity is not None:
                        ptau = purity_tau if (purity_tau is not None and purity_tau > 0) else 5e-2
                        # OR-combination: active if big enough (mass) OR cleanly owned (purity)
                        soft = torch.maximum(soft, torch.sigmoid((purity - purity_q) / ptau))
                    # default slots are always fully active (value 1, no gradient there)
                    default_vec = torch.zeros_like(soft[:1])  # (1, S)
                    for di in default_list:
                        if 0 <= int(di) < default_vec.shape[1]:
                            default_vec[0, int(di)] = 1.0
                    soft = torch.maximum(soft, default_vec)
                    if gate_mode == "ste":
                        # forward = hard 0/1 (defaults forced on); backward = soft gradient
                        hard_bool = exceeds
                        if purity is not None:
                            hard_bool = hard_bool | (purity >= purity_q)
                        hard = torch.maximum(hard_bool.float(), default_vec)
                        return hard + (soft - soft.detach())
                    return soft  # float gate weights in [0, 1]
                active = exceeds.clone()  # (B, S)
                if purity is not None:
                    active = active | (purity >= purity_q)
                # default slots are always active (exempt from the threshold)
                for di in default_list:
                    if 0 <= int(di) < active.shape[1]:
                        active[:, int(di)] = True
                return active  # bool

            if gate_form == "purity_weight":
                # v32 (purity gating): the ownership statistic IS the gate. No threshold,
                # no temperature, no schedule -- the whole curriculum machine (p, tau,
                # beta, build_gate above) is bypassed and c is used directly:
                #   decoder : masks * c, renormalized  ==  softmax_s(alpha + log c)
                #   temporal: alpha = c / max_j(c) via state_max_norm further down
                # Untrained attention gives near-uniform c (~1/S), and a uniform gate is
                # cancelled exactly by both application points (decoder renorm, max-norm),
                # so early training is the ungated baseline; the gate phases itself in as
                # attention sharpens (self-annealing, no schedule to mistune). c stays
                # detached (anti-gaming), so the gate is pure forward modulation.
                # gate_p_state has no meaning here (there is no threshold to split).
                #
                # v36: purity_normalize maps the uniform baseline c=1/K onto 0
                #   p = clip((K c - 1)/(K - 1), 0, 1)
                # so ghosts go to 0 while exclusive owners stay at 1. Decoder adds
                # a floor on g so all-zero p recovers softmax(alpha) (ungated).
                active_mask = gate_conf
                if purity_normalize and active_mask is not None:
                    n_slots_g = active_mask.shape[-1]
                    if n_slots_g > 1:
                        active_mask = (
                            (n_slots_g * active_mask - 1.0) / (n_slots_g - 1.0)
                        ).clamp(0.0, 1.0)
                if default_list:
                    default_vec = torch.zeros_like(active_mask[:1])  # (1, S)
                    for di in default_list:
                        if 0 <= int(di) < default_vec.shape[1]:
                            default_vec[0, int(di)] = 1.0
                    active_mask = torch.maximum(active_mask, default_vec)
                state_mask = active_mask
            else:
                active_mask = build_gate(p_use)
                # The decoder's threshold has to anneal below the smallest object's mass or
                # small objects are never representable, but at that p the gate saturates and
                # goes flat across slots, which is exactly where the temporal mix loses its
                # selectivity. `gate_p_state` decouples the two: the returned active_mask
                # (decoder, logging, contrastive, aux losses) keeps the annealed p, while the
                # predictor re-gate further down uses a threshold of its own.
                state_mask = (
                    active_mask if gate_p_state is None else build_gate(float(gate_p_state))
                )
            if gate_detach:
                # sg(g): keep the forward gate, drop the Jacobian into (m, c).
                same = state_mask is active_mask
                if active_mask is not None and torch.is_floating_point(active_mask):
                    active_mask = active_mask.detach()
                if same:
                    state_mask = active_mask
                elif state_mask is not None and torch.is_floating_point(state_mask):
                    state_mask = state_mask.detach()
            # The corrector output is deliberately NOT gated here. Gating it as well as the
            # predictor output puts the gate twice on the same path, and because the predictor
            # is a residual block (Pred(x) = x + D) the observation then lands at g^2 while the
            # dynamics arrive at g -- a slot at g = 0.3 would admit 9% of what it sees while
            # taking 30% of its predicted motion, which is not a ratio anything asked for:
            #   both gated:  hat{x}_{t+1} = hat{x}_t + g^2 (u_t - hat{x}_t) + g D
            #   here:        hat{x}_{t+1} = hat{x}_t + g   (u_t - hat{x}_t) + g D
            # The remaining application is the predictor re-gate, which is the temporal
            # propagation step the gate is meant to control.
            #
            # Consequence to keep in mind: `state` (what the decoder, loss_ss and the velocity
            # signal all read) now carries the full correction for every slot, so a low-gate
            # slot no longer hands stale content to the decoder. The gate restricts what that
            # slot's prior -- and hence the query it corrects from -- can become, not what it
            # reports this frame.

        # Velocity of the corrector output, one step behind the displacement the predictor is
        # asked to produce. Input and target are the same quantity, which makes this a plain
        # autoregressive second-order model d_t = f(x_t, d_{t-1}) rather than one that also
        # has to learn a change of coordinates. `prev_state` is None on the first frame, which
        # leaves the predictor exactly in its velocity-free configuration there.
        vel = None
        if self.predictor_takes_vel and prev_state is not None:
            vel = updated_state - prev_state

        if self.predictor:
            if vel is not None:
                predicted_state = self.predictor(updated_state, vel=vel)
            else:
                predicted_state = self.predictor(updated_state)
        else:
            # Just pass updated_state along as prediction
            predicted_state = updated_state
        # Kept for the dynamics loss, which must supervise the predictor module itself rather
        # than the gated mix: reading the mix would scale the predictor's gradient by g, so
        # the throttled slots that most need a dynamics prior would learn one the slowest.
        predicted_pregate = predicted_state

        # The one place the mass gate enters the temporal path. Low-gate slots keep their
        # incoming prior instead of advancing, so the gate sets how fast a slot's memory is
        # allowed to track what it sees:
        #   hat{x}_{t+1} = g * Pred(u_t) + (1-g) * hat{x}_t  =  hat{x}_t + g (u_t - hat{x}_t) + g D
        # `state_max_norm` normalizes by max_s(g) here so the winning slot always advances
        # fully; the decoder keeps the raw gate, being scale-invariant after its renorm.
        #
        # predictor_ungated=True removes this mix, and since the corrector output is no longer
        # gated either that leaves the gate out of the temporal path entirely -- it would then
        # only reweight decoder masks. Only v18 sets it.
        if state_mask is not None and not predictor_ungated:
            a = _max_norm(state_mask, state_max_norm).unsqueeze(-1).to(predicted_state.dtype)
            predicted_state = a * predicted_state + (1.0 - a) * state

        if active_mask is None:
            # keep a consistent output tree; all slots are "active" when gating is off
            active_mask = torch.ones(
                updated_state.shape[:2], dtype=torch.bool, device=updated_state.device
            )
        if state_mask is None:
            state_mask = active_mask

        out = {
            "state": updated_state,
            "state_predicted": predicted_state,
            "state_predicted_pregate": predicted_pregate,
            "corrector": corrector_output,
            "state_attn_mask": state_attn_mask,
            "active_mask": active_mask,
            # Always present so the output tree shape is constant; equals active_mask
            # unless gate_p_state gave the temporal mix its own threshold.
            "state_gate": state_mask,
        }
        if gate_conf is not None:
            # detached (B, S) assignment confidence, exposed for logging only
            out["gate_conf"] = gate_conf
        if mass_median_ema is not None:
            out["mass_median_ema"] = mass_median_ema
        if gate_p_eff is not None and torch.is_tensor(gate_p_eff):
            out["gate_p_eff"] = gate_p_eff.squeeze(-1)  # (B,)
        if gate_delta is not None and torch.is_tensor(gate_delta):
            out["gate_delta"] = gate_delta.squeeze(-1)  # (B,)
        return out


class MapOverTime(nn.Module):
    """Wrapper applying wrapped module independently to each time step.

    Assumes batch is first dimension, time is second dimension.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args):
        batch_size = None
        seq_len = None
        flattened_args = []
        for idx, arg in enumerate(args):
            B, T = arg.shape[:2]
            if not batch_size:
                batch_size = B
            elif batch_size != B:
                raise ValueError(
                    f"Inconsistent batch size of {B} of argument {idx}, was {batch_size} before."
                )

            if not seq_len:
                seq_len = T
            elif seq_len != T:
                raise ValueError(
                    f"Inconsistent sequence length of {T} of argument {idx}, was {seq_len} before."
                )

            flattened_args.append(arg.flatten(0, 1))

        outputs = self.module(*flattened_args)

        if isinstance(outputs, Mapping):
            unflattened_outputs = {
                k: v.unflatten(0, (batch_size, seq_len)) for k, v in outputs.items()
            }
        else:
            unflattened_outputs = outputs.unflatten(0, (batch_size, seq_len))

        return unflattened_outputs


class ScanOverTime(nn.Module):
    """Wrapper applying wrapped module recurrently over time steps"""

    def __init__(
        self, module: nn.Module, next_state_key: str = "state_predicted", pass_step: bool = True
    ) -> None:
        super().__init__()
        self.module = module
        self.next_state_key = next_state_key
        self.pass_step = pass_step
        # Anchor frame chosen per sample by the last "evidence"/"random" cycle (diagnostics
        # for eval scripts; None when the last forward used another mode).
        self.last_anchor_frames: Optional[torch.Tensor] = None

    def forward(
        self,
        initial_state: torch.Tensor,
        inputs: torch.Tensor,
        cycle: Any = False,
        gate_p: Optional[float] = None,
        default_idx: Any = 0,
        gate_mode: str = "hard",
        gate_tau: Optional[float] = None,
        mass_gamma: float = 1.0,
        purity_q: Optional[float] = None,
        purity_tau: Optional[float] = None,
        gate_p_state: Optional[float] = None,
        state_max_norm: bool = False,
        predictor_ungated: bool = False,
        p_mode: str = "absolute",
        median_ema_momentum: float = 0.9,
        p_residual_net: Optional[nn.Module] = None,
        p_residual_alpha: float = 0.0,
        gate_form: str = "linear",
        gate_beta: Optional[float] = None,
        gate_tau_log: Optional[float] = None,
        conf_kind: str = "entropy",
        gate_detach: bool = False,
        key_inputs: Optional[torch.Tensor] = None,
        purity_normalize: bool = False,
    ):
        # initial_state: batch x ...
        # inputs: batch x n_frames x ...
        seq_len = inputs.shape[1]

        gate_kwargs = dict(
            gate_p=gate_p, default_idx=default_idx, gate_mode=gate_mode, gate_tau=gate_tau,
            mass_gamma=mass_gamma, purity_q=purity_q, purity_tau=purity_tau,
            gate_p_state=gate_p_state,
            state_max_norm=state_max_norm,
            predictor_ungated=predictor_ungated,
            p_mode=p_mode, median_ema_momentum=median_ema_momentum,
            p_residual_net=p_residual_net, p_residual_alpha=p_residual_alpha,
            gate_form=gate_form, gate_beta=gate_beta, gate_tau_log=gate_tau_log,
            conf_kind=conf_kind, gate_detach=gate_detach,
            purity_normalize=purity_normalize,
        )

        state = initial_state
        median_ema = None
        # Previous posterior, used by the predictor as a velocity reference. Kept in sweep
        # order, so during the backward pass below it is the temporally later frame and the
        # velocity correctly points the way that sweep is predicting.
        prev_state = None
        outputs = []
        for t in range(seq_len):
            kwargs = dict(gate_kwargs)
            kwargs["median_ema_prev"] = median_ema
            kwargs["prev_state"] = prev_state
            if key_inputs is not None:
                kwargs["key_features"] = key_inputs[:, t]
            if self.pass_step:
                output = self.module(state, inputs[:, t], t, **kwargs)
            else:
                output = self.module(state, inputs[:, t], **kwargs)
            outputs.append(output)
            prev_state = output["state"]
            state = output[self.next_state_key]
            if "mass_median_ema" in output:
                median_ema = output["mass_median_ema"]

        self.last_anchor_frames = None
        if cycle:
            # `cycle` selects the re-inference protocol after the forward sweep:
            #   True / "last": legacy cyclic inference -- backward sweep anchored at the
            #       LAST frame's state, all frames replaced by the backward outputs.
            #   "evidence": evidence-anchored bidirectional inference (EABI). The anchor
            #       is the frame where the most objects are intactly bound (soft active
            #       count weighted by ownership purity), the backward sweep runs
            #       anchor -> 1, and only pre-anchor frames are replaced (re-running
            #       forward from the anchor state would reproduce the forward sweep
            #       exactly, so those frames are kept as-is).
            #   "evidence_mass": EABI with the coverage-weighted statistic sum_s g*m
            #       (mass share owned by trusted slots) -- kept as an A/B variant; it
            #       ignores object COUNT, so a frame where one background slot owns
            #       everything can outscore a frame with five cleanly-bound objects.
            #   "random": EABI with a uniformly random anchor -- control run isolating
            #       the value of evidence-based anchor selection.
            mode = cycle.strip().lower() if isinstance(cycle, str) else "last"
            if mode in ("evidence", "evidence_mass", "random"):
                if mode == "random":
                    anchors = torch.randint(
                        seq_len, (inputs.shape[0],), device=inputs.device
                    )
                else:
                    anchors = _evidence_anchors(
                        outputs,
                        gate_kwargs.get("mass_gamma") or 1.0,
                        stat="mass" if mode == "evidence_mass" else "count",
                    )
                self.last_anchor_frames = anchors.detach().to("cpu")
                return self._cycle_from_anchors(
                    outputs, inputs, gate_kwargs, median_ema, anchors, key_inputs
                )
            if mode != "last":
                raise ValueError(f"unknown cycle mode {cycle!r}")
            # backward pass
            ### not last frame
            new_outputs = []
            new_outputs.append(outputs[-1])
            state = outputs[-1][self.next_state_key]
            prev_state = outputs[-1]["state"]
            # continue EMA through the backward sweep (same clip statistics)
            for t in range(seq_len - 1):
                back_t = seq_len - t - 2
                kwargs = dict(gate_kwargs)
                kwargs["median_ema_prev"] = median_ema
                kwargs["prev_state"] = prev_state
                if key_inputs is not None:
                    kwargs["key_features"] = key_inputs[:, back_t]
                out = self.module(state, inputs[:, back_t], **kwargs)
                new_outputs.append(out)
                prev_state = out["state"]
                state = out[self.next_state_key]
                if "mass_median_ema" in out:
                    median_ema = out["mass_median_ema"]
            new_outputs = new_outputs[::-1]  # reverse the order of outputs
            return merge_dict_trees(new_outputs, axis=1)

        return merge_dict_trees(outputs, axis=1)

    def _cycle_from_anchors(
        self,
        outputs: List[Dict[str, Any]],
        inputs: torch.Tensor,
        gate_kwargs: Dict[str, Any],
        median_ema: Optional[torch.Tensor],
        anchors: torch.Tensor,
        key_inputs: Optional[torch.Tensor] = None,
    ):
        """Backward sweep from a per-sample anchor frame, stitched with the forward sweep.

        Semantics per sample b with anchor a_b: frames >= a_b keep the forward outputs,
        frames < a_b come from a backward sweep warm-started at the anchor's predicted
        state (mirroring the legacy cycle, which is the special case a_b = T-1). Both
        sweeps share the anchor state, so slot identities stay consistent across the
        stitch -- required for video metrics.

        Batching: the sweep iterates t = max(a)-1 .. 0 for the whole batch at once; a
        sample's rows are (re-)injected with its anchor states at its own start step
        t = a_b - 1. Rows computed before a sample's sweep begins are finite but
        meaningless and are discarded by the per-sample stitch, so mixed anchors in one
        batch cost only wasted compute, never wrong outputs.
        """
        seq_len = len(outputs)
        b = inputs.shape[0]
        forward_tree = merge_dict_trees(outputs, axis=1)
        max_anchor = int(anchors.max().item())
        if max_anchor <= 0:
            return forward_tree

        # (B, S, D) anchor states gathered per sample. The backward step out of the
        # anchor consumes the anchor's *predicted* state, exactly like the legacy cycle
        # consumes outputs[-1][next_state_key]; the posterior becomes prev_state so the
        # predictor's velocity reference stays in sweep order.
        pred_states = torch.stack([o[self.next_state_key] for o in outputs], dim=1)
        post_states = torch.stack([o["state"] for o in outputs], dim=1)
        idx = anchors.view(b, 1, 1, 1).expand(-1, 1, *pred_states.shape[2:])
        anchor_pred = pred_states.gather(1, idx).squeeze(1)
        anchor_post = post_states.gather(1, idx).squeeze(1)

        state = anchor_pred
        prev_state = anchor_post
        new_outputs = list(outputs)  # frames >= anchor keep the forward outputs
        for t in range(max_anchor - 1, -1, -1):
            starts = (anchors == t + 1).view(b, *([1] * (state.ndim - 1)))
            state = torch.where(starts, anchor_pred, state)
            prev_state = torch.where(starts, anchor_post, prev_state)
            kwargs = dict(gate_kwargs)
            kwargs["median_ema_prev"] = median_ema
            kwargs["prev_state"] = prev_state
            if key_inputs is not None:
                kwargs["key_features"] = key_inputs[:, t]
            # no time_step: the first-step corrector args never apply on re-inference
            # sweeps (matches the legacy cycle)
            out = self.module(state, inputs[:, t], **kwargs)
            new_outputs[t] = out
            prev_state = out["state"]
            state = out[self.next_state_key]
            if "mass_median_ema" in out:
                median_ema = out["mass_median_ema"]

        backward_tree = merge_dict_trees(new_outputs, axis=1)
        use_backward = (
            torch.arange(seq_len, device=anchors.device).view(1, seq_len)
            < anchors.view(b, 1)
        )
        return _stitch_trees(forward_tree, backward_tree, use_backward)


def _evidence_anchors(
    outputs: List[Dict[str, Any]], mass_gamma: float, window: int = 5, stat: str = "count"
) -> torch.Tensor:
    """Per-sample anchor frame: argmax of the gate's own per-frame evidence.

    stat="count" (default): E_t = sum_s g_{t,s} * c_{t,s}, the soft number of active
    slots weighted by ownership purity c = sum_f A~^2 / sum_f A~. This implements "the
    frame where the most objects are INTACTLY present": sum_s g counts trusted slots,
    purity discounts objects that are only partially visible / entering / occluded
    (mixed ownership at their boundary lowers c; a fully-owned object scores ~1 at any
    size). The anchor exists to hand the backward sweep a state that has bound as many
    of the video's objects as possible, which is a count, not a mass share.

    stat="mass": E_t = sum_s g_{t,s} * m_{t,s}, the fraction of (gamma-sharpened)
    attention mass owned by trusted slots. Since sum_s m_s = 1 per frame this measures
    how confidently the frame is EXPLAINED, not how many objects are held -- one
    trusted background slot owning everything can outscore five cleanly-bound objects.
    Kept as an A/B variant for the eval.

    Both signals come from the forward sweep's outputs, so anchor selection costs no
    extra model evaluation. E_t is mean-smoothed over a `window`-frame neighborhood
    (replicate-padded) before the argmax so a single-frame noise spike cannot become
    the anchor. With gating disabled active_mask is all-ones and gamma=1 purity/mass
    are frame-independent constants only in degenerate cases; anchor quality then just
    falls back to whatever the statistic sees.
    """
    gates = torch.stack([o["active_mask"].float() for o in outputs], dim=1)  # (B, T, S)
    att = torch.stack([o["state_attn_mask"].float() for o in outputs], dim=1)  # (B,T,S,F)
    gamma = float(mass_gamma or 1.0)
    if gamma != 1.0:
        att = att.pow(gamma)
        att = att / att.sum(dim=2, keepdim=True).clamp_min(1e-8)
    if stat == "mass":
        per_slot = att.sum(dim=-1) / att.shape[-1]  # coverage m, (B, T, S)
    else:
        per_slot = (att * att).sum(dim=-1) / att.sum(dim=-1).clamp_min(1e-8)  # purity c
    evidence = (gates * per_slot).sum(dim=-1)  # (B, T)

    t_len = evidence.shape[1]
    k = min(window, t_len)
    if k % 2 == 0:
        k = max(k - 1, 1)
    if k > 1:
        kernel = torch.ones(1, 1, k, device=evidence.device, dtype=evidence.dtype) / k
        padded = torch.nn.functional.pad(
            evidence.unsqueeze(1), (k // 2, k // 2), mode="replicate"
        )
        evidence = torch.nn.functional.conv1d(padded, kernel).squeeze(1)
    return evidence.argmax(dim=1)  # (B,)


def _stitch_trees(
    forward_tree: Mapping, backward_tree: Mapping, use_backward: torch.Tensor
):
    """Per-(sample, frame) select between two stacked output trees.

    `use_backward` is (B, T) bool; leaves shaped (B, T, ...) are selected elementwise,
    anything else (non-tensors, oddly shaped leaves) keeps the forward version.
    """
    out = {}
    for key, fwd in forward_tree.items():
        bwd = backward_tree[key]
        if isinstance(fwd, Mapping):
            out[key] = _stitch_trees(fwd, bwd, use_backward)
        elif (
            isinstance(fwd, torch.Tensor)
            and fwd.ndim >= 2
            and fwd.shape[:2] == use_backward.shape
        ):
            mask = use_backward.view(*use_backward.shape, *([1] * (fwd.ndim - 2)))
            out[key] = torch.where(mask, bwd, fwd)
        else:
            out[key] = fwd
    return out


def merge_dict_trees(trees: List[Mapping], axis: int = 0):
    """Stack all leafs given a list of dictionaries trees.

    Example:
    x = merge_dict_trees([
        {
            "a": torch.ones(2, 1),
            "b": {"x": torch.ones(2, 2)}
        },
        {
            "a": torch.ones(3, 1),
            "b": {"x": torch.ones(1, 2)}
        }
    ])

    x == {
        "a": torch.ones(5, 1),
        "b": {"x": torch.ones(3, 2)}
    }
    """
    out = {}
    if len(trees) > 0:
        ref_tree = trees[0]
        for key, value in ref_tree.items():
            values = [tree[key] for tree in trees]
            if isinstance(value, torch.Tensor):
                out[key] = torch.stack(values, axis)
            elif isinstance(value, Mapping):
                out[key] = merge_dict_trees(values, axis)
            else:
                out[key] = values

    return out
