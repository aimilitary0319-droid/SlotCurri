from typing import Any, Dict, Optional, Tuple

import einops
import torch
from torch import nn

from slotcurri import modules, utils
from pytorch_msssim import ssim

@utils.make_build_fn(__name__, "loss")
def build(config, name: str):
    target_transform = None
    if config.get("target_transform"):
        target_transform = modules.build_module(config.get("target_transform"))

    cls = utils.get_class_by_name(__name__, name)
    if cls is not None:
        return cls(
            target_transform=target_transform,
            **utils.config_as_kwargs(config, ("target_transform",)),
        )
    else:
        raise ValueError(f"Unknown loss `{name}`")


class Loss(nn.Module):
    """Base class for loss functions.

    Args:
        video_inputs: If true, assume inputs contain a time dimension.
        patch_inputs: If true, assume inputs have a one-dimensional patch dimension. If false,
            assume inputs have height, width dimensions.
        pred_dims: Dimensions [from, to) of prediction tensor to slice. Useful if only a
            subset of the predictions should be used in the loss, i.e. because the other dimensions
            are used in other losses.
        remove_last_n_frames: Number of frames to remove from the prediction before computing the
            loss. Only valid with video inputs. Useful if the last frame does not have a
            correspoding target.
        target_transform: Transform that can optionally be applied to the target.
    """

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        video_inputs: bool = False,
        patch_inputs: bool = True,
        keep_input_dim: bool = False,
        pred_dims: Optional[Tuple[int, int]] = None,
        remove_last_n_frames: int = 0,
        target_transform: Optional[nn.Module] = None,
        input_key: Optional[str] = None,
    ):
        super().__init__()
        self.pred_path = pred_key.split(".")
        self.target_path = target_key.split(".")
        self.video_inputs = video_inputs
        self.patch_inputs = patch_inputs
        self.keep_input_dim = keep_input_dim
        self.input_key = input_key
        self.n_expected_dims = (
            2 + (1 if patch_inputs or keep_input_dim else 2) + (1 if video_inputs else 0)
        )

        if pred_dims is not None:
            assert len(pred_dims) == 2
            self.pred_dims = slice(pred_dims[0], pred_dims[1])
        else:
            self.pred_dims = None

        self.remove_last_n_frames = remove_last_n_frames
        if remove_last_n_frames > 0 and not video_inputs:
            raise ValueError("`remove_last_n_frames > 0` only valid with `video_inputs==True`")

        self.target_transform = target_transform
        self.to_canonical_dims = self.get_dimension_canonicalizer()

    def get_dimension_canonicalizer(self) -> torch.nn.Module:
        """Return a module which reshapes tensor dimensions to (batch, n_positions, n_dims)."""
        if self.video_inputs:
            if self.patch_inputs:
                pattern = "B F P D -> B (F P) D"
            elif self.keep_input_dim:
                return torch.nn.Identity()
            else:
                pattern = "B F D H W -> B (F H W) D"
        else:
            if self.patch_inputs:
                return torch.nn.Identity()
            else:
                pattern = "B D H W -> B (H W) D"

        return einops.layers.torch.Rearrange(pattern)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        target = utils.read_path(outputs, elements=self.target_path, error=False)
        if target is None:
            target = utils.read_path(inputs, elements=self.target_path)

        target = target.detach()

        if self.target_transform:
            with torch.no_grad():
                if self.input_key is not None:
                    target = self.target_transform(target, inputs[self.input_key])
                else:
                    target = self.target_transform(target)

        # Convert to dimension order (batch, positions, dims)
        target = self.to_canonical_dims(target)

        return target

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        prediction = utils.read_path(outputs, elements=self.pred_path)
        if prediction.ndim != self.n_expected_dims:
            raise ValueError(
                f"Prediction has {prediction.ndim} dimensions (and shape {prediction.shape}), but "
                f"expected it to have {self.n_expected_dims} dimensions."
            )

        if self.video_inputs and self.remove_last_n_frames > 0:
            prediction = prediction[:, : -self.remove_last_n_frames]

        # Convert to dimension order (batch, positions, dims)
        prediction = self.to_canonical_dims(prediction)

        if self.pred_dims:
            prediction = prediction[..., self.pred_dims]

        return prediction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in subclasses")


class TorchLoss(Loss):
    """Wrapper around PyTorch loss functions."""

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        loss: str,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        loss_kwargs = loss_kwargs if loss_kwargs is not None else {}
        if hasattr(torch.nn, loss):
            self.loss_fn = getattr(torch.nn, loss)(reduction="mean", **loss_kwargs)
            self.loss_fn_none = getattr(torch.nn, loss)(reduction="none", **loss_kwargs)
        else:
            raise ValueError(f"Loss function torch.nn.{loss} not found")

        # Cross entropy loss wants dimension order (batch, classes, positions)
        self.positions_last = loss == "CrossEntropyLoss"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.positions_last:
            prediction = prediction.transpose(-2, -1)
            target = target.transpose(-2, -1)

        return self.loss_fn(prediction, target)


class MSELoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, loss="MSELoss", **kwargs)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, return_none=False) -> torch.Tensor:
        if self.positions_last:
            prediction = prediction.transpose(-2, -1)
            target = target.transpose(-2, -1)
        if return_none:
            return self.loss_fn_none(prediction, target)
        return self.loss_fn(prediction, target)

class SSIMLoss(Loss):
    def __init__(
        self,
        pred_key: str,
        target_key: str,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        target = utils.read_path(outputs, elements=self.target_path, error=False)
        if target is None:
            target = utils.read_path(inputs, elements=self.target_path)

        target = target.detach()

        if self.target_transform:
            with torch.no_grad():
                if self.input_key is not None:
                    target = self.target_transform(target, inputs[self.input_key])
                else:
                    target = self.target_transform(target)

        # print('SSIM3D loss target shape : ', target.shape)
        return target

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        prediction = utils.read_path(outputs, elements=self.pred_path)

        if self.video_inputs and self.remove_last_n_frames > 0:
            prediction = prediction[:, : -self.remove_last_n_frames]

        # print('SSIM3D loss prediction shape : ', prediction.shape)
        # Convert to dimension order (batch, positions, dims)
        # prediction = self.to_canonical_dims(prediction)
        #
        # if self.pred_dims:
        #     prediction = prediction[..., self.pred_dims]


        return prediction

    def forward(self, feat_recon, feat_orig):
        # feat_recon, feat_orig: (B, C, T, H, W)
        B, C, T, H, W = feat_recon.shape
        eps = 1e-6

        # orig_flat = feat_orig.reshape(B, -1)
        # min_o     = orig_flat.min(dim=1)[0].view(B, 1, 1, 1, 1)
        # max_o     = orig_flat.max(dim=1)[0].view(B, 1, 1, 1, 1)
        # recon_flat= feat_recon.reshape(B, -1)
        # min_r     = recon_flat.min(dim=1)[0].view(B, 1, 1, 1, 1)
        # max_r     = recon_flat.max(dim=1)[0].view(B, 1, 1, 1, 1)
        # feat_orig_norm  = (feat_orig  - min_o) / (max_o - min_o + eps)
        # feat_recon_norm = (feat_recon - min_r) / (max_r - min_r + eps)

        orig_flat = feat_orig.reshape(B, -1)
        recon_flat = feat_recon.reshape(B, -1)
        all_flat = torch.cat([orig_flat, recon_flat], dim=1)
        min_val = all_flat.min(dim=1)[0].view(B, 1, 1, 1, 1).detach()
        max_val = all_flat.max(dim=1)[0].view(B, 1, 1, 1, 1).detach()

        feat_orig_norm = (feat_orig - min_val) / (max_val - min_val + eps)
        feat_recon_norm = (feat_recon - min_val) / (max_val - min_val + eps)

        loss_ssim3d = 1.0 - ssim(
            feat_recon_norm, feat_orig_norm.detach(),
            data_range=1.0,   # [0,1]
            size_average=True,
            win_size=3
        )
        return loss_ssim3d

        # ssim3d = SSIM3D(window_size=3, reduction='mean')
        #
        # loss_ssim3d = 1.0 - ssim3d(x, y)
        # return loss_ssim3d

class CrossEntropyLoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, loss="CrossEntropyLoss", **kwargs)

def _batch_cat_slots(tensor: torch.Tensor) -> torch.Tensor:
    """(B, T, K, ...) -> (1, T, B*K, ...); matches Slot_Slot batch_contrast layout."""
    if tensor.ndim == 3:
        return einops.rearrange(tensor, "b t k -> t (b k)").unsqueeze(0)
    if tensor.ndim == 4:
        return einops.rearrange(tensor, "b t k d -> t (b k) d").unsqueeze(0)
    raise ValueError(
        f"batch_contrast layout expects 3D or 4D (B, T, K, [D]), got {tuple(tensor.shape)}"
    )


class Slot_Slot_Contrastive_Loss(Loss):
    def __init__(
        self,
        pred_key: str,
        target_key: str,
        temperature: float = 0.1,
        batch_contrast: bool = True,
        gate_negatives: bool = False,
        cand_neg: bool = False,
        occ_kernel: bool = False,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.batch_contrast = batch_contrast
        # If True, weight each candidate (negative) in the softmax denominator by its gate,
        # so dormant/partially-gated slots contribute proportionally as negatives instead of
        # at full strength. Only affects the active-mask path; default False keeps the
        # original (baseline-aligned) behavior where all slots are full-strength negatives.
        self.gate_negatives = gate_negatives
        # Same-frame extras: other *candidate* slots (c=1-π̄) as extra CE classes.
        # Identity positive u_i^(t) ↔ u_i^(t+1) is unchanged. Live columns get
        # -inf extras so their softmax stays the original S-way. Default False.
        self.cand_neg = bool(cand_neg)
        # Identity CE unchanged. Logit bias log(π̄_i π̄_j + c_i c_j) splits the
        # softmax into a live pool and a candidate pool (c=1-π̄). Cross terms
        # drop out, so candidates may match live (challenge) while remaining
        # unique among themselves. Default False is original S-way CE.
        self.occ_kernel = bool(occ_kernel)

    def forward(self, slots, _, active_mask=None, occupancy=None):
        # slots: (B, T, S, D); active_mask (optional): (B, T, S) bool.
        slots = nn.functional.normalize(slots, p=2.0, dim=-1)
        # Live weights are π̄ = π / max_s(π), same as L_cc / occ_kernel.
        # Must happen before batch_contrast so the max is per (B, T), not B*K.
        if active_mask is not None and active_mask.dtype != torch.bool:
            g = active_mask.float().detach()
            active_mask = g / g.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        pi_bar = None
        if occupancy is not None and occupancy.dtype != torch.bool:
            if self.cand_neg or self.occ_kernel:
                g = occupancy.float().detach()
                pi_bar = g / g.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        cand = None
        if self.cand_neg and pi_bar is not None:
            cand = (1.0 - pi_bar).clamp(0.0, 1.0)
        if self.batch_contrast:
            slots = _batch_cat_slots(slots)
            if active_mask is not None:
                # match the (batch-concatenated) slot layout: (1, T, B*K)
                active_mask = _batch_cat_slots(active_mask)
            if cand is not None:
                cand = _batch_cat_slots(cand)
            if pi_bar is not None:
                pi_bar = _batch_cat_slots(pi_bar)
        s1 = slots[:, :-1, :, :]
        s2 = slots[:, 1:, :, :]
        ss = torch.matmul(s1, s2.transpose(-2, -1)) / self.temperature
        B, T, S, D = ss.shape
        ss = ss.reshape(B * T, S, S)
        if self.occ_kernel and pi_bar is not None:
            # w_ij = π̄_i^t π̄_j^{t+1} + c_i^t c_j^{t+1}; diagonal forced to 1
            # so identity stays in the softmax if occupancy flips across t.
            p1 = pi_bar[:, :-1].reshape(B * T, S)
            p2 = pi_bar[:, 1:].reshape(B * T, S)
            c1 = (1.0 - p1).clamp(0.0, 1.0)
            c2 = (1.0 - p2).clamp(0.0, 1.0)
            w = p1.unsqueeze(-1) * p2.unsqueeze(-2) + c1.unsqueeze(-1) * c2.unsqueeze(-2)
            w = w.clone()
            w.diagonal(dim1=-2, dim2=-1).fill_(1.0)
            ss = ss + w.clamp_min(1e-8).log()
        if cand is not None:
            # Extra CE classes (concat on dim=1, the original softmax axis):
            #   extra_{k,j} = <u_k^(t), u_j^(t)>/τ  if both k,j are candidates, k≠j
            # Live columns (c_j=0) keep all extras at -inf → original S-way CE.
            c1 = cand[:, :-1].reshape(B * T, S)
            ff = torch.matmul(s1, s1.transpose(-2, -1)) / self.temperature
            ff = ff.reshape(B * T, S, S)
            pair_c = c1.unsqueeze(-1) * c1.unsqueeze(-2)
            pair_c = pair_c * (
                1.0 - torch.eye(S, device=ss.device, dtype=pair_c.dtype)
            )
            extra = ff + pair_c.clamp_min(1e-8).log()
            extra = extra.masked_fill(pair_c <= 0, float("-inf"))
            ss = torch.cat([ss, extra], dim=1)  # (B*T, 2S, S)
        if active_mask is None:
            if ss.shape[1] == S:
                target = torch.eye(S).expand(B * T, S, S).to(ss.device)
                return self.criterion(ss, target)
            # Identity class is still row i of the time block (first S rows).
            target = torch.arange(S, device=ss.device).unsqueeze(0).expand(B * T, S)
            return self.criterion(ss, target)

        # active-only: only slots active at both frame t and t+1 count as anchors.
        # Works for hard (bool) and soft (float gate in [0, 1]) masks: the product is a
        # per-anchor weight, so partially-gated slots contribute proportionally.
        a1 = active_mask[:, :-1].float()
        a2 = active_mask[:, 1:].float()
        # Detach the gate weights so the contrastive loss uses them ONLY as fixed per-anchor
        # importance masks, never as a learnable lever. Without detach, loss = -sum(diag*pair)/
        # sum(pair) is a gate-weighted average of alignment quality, which is minimized by
        # concentrating gate weight on already-well-aligned (high-mass) slots and driving the
        # gates of poorly-aligned (small-object, low-mass) slots toward 0 -> mass concentration
        # / slot collapse. Detaching removes that pro-collapse gradient: gates are then shaped
        # only by featrec (anti-collapse) + gate_div, and loss_ss trains representations only.
        pair = (a1.detach() * a2.detach()).reshape(B * T, S)  # (B*T, S)
        if self.gate_negatives:
            # Weight each candidate (frame-t slot i, the softmax axis dim=1) by its gate:
            # adding log(gate_i) to the logits makes logsumexp weight negatives by their
            # gate. Hard/STE gates in {0, 1} -> log(1)=0 keep, log(0)=-inf drop (mask dormant
            # negatives); soft gates in (0, 1) -> continuous down-weighting. Unifies all modes.
            log_a1 = torch.log(a1.reshape(B * T, S).clamp_min(1e-8))  # (B*T, S)
            time_bias = log_a1.unsqueeze(-1)  # (B*T, S, 1) over the t+1 axis
            if ss.shape[1] == S:
                ss = ss + time_bias
            else:
                ss = ss.clone()
                ss[:, :S] = ss[:, :S] + time_bias
        # CrossEntropy with identity target == -log_softmax over candidate rows at the diagonal.
        # First S rows are the time block; extra cand-neg rows (if any) sit at S:2S.
        logp = torch.log_softmax(ss, dim=1)
        diag = torch.diagonal(logp[:, :S], dim1=1, dim2=2)  # (B*T, S)
        loss = -(diag * pair).sum() / pair.sum().clamp(min=1.0)
        return loss


class Slot_Candidate_Repel_Loss(Loss):
    """Same-frame candidate-candidate repulsion. Live is not a negative.

    For L2-unit slots and detached occupancy, leftover vs live is relative
    but only after something is actually occupied:

        π̄ = π / max_s(π),   c = sg(1-π̄) sg(π_max)

        L = mean_{b,t} log(1 + sum_{k≠j} c_k c_j [u_k^T u_j]_+^2)

    Early π_max≈0 → no candidates. The inner sum is stricter as more
    cands overlap. log1p keeps it auxiliary to L_ss (weight 0.1 vs 0.5).
    Mean only over batch and time.
    """

    def __init__(self, pred_key: str, target_key: str, eps: float = 1e-8, **kwargs):
        super().__init__(pred_key, target_key, **kwargs)
        self.eps = float(eps)

    def forward(self, slots, _, occupancy=None):
        # slots: (B, T, S, D) or (B, S, D)
        u = nn.functional.normalize(slots, p=2.0, dim=-1)
        if u.ndim == 3:
            u = u.unsqueeze(1)
        if occupancy is None:
            c = torch.ones(u.shape[:-1], dtype=u.dtype, device=u.device)
        else:
            g = occupancy.float().detach()
            if g.ndim == 2:
                g = g.unsqueeze(1)
            pi_max = g.amax(dim=-1, keepdim=True)
            pi_bar = g / pi_max.clamp_min(self.eps)
            c = ((1.0 - pi_bar) * pi_max).clamp(0.0, 1.0)
        # (B, T, S, S)
        cos = torch.matmul(u, u.transpose(-2, -1))
        s = cos.shape[-1]
        off = 1.0 - torch.eye(s, device=cos.device, dtype=cos.dtype)
        w = c.unsqueeze(-1) * c.unsqueeze(-2) * off
        pair = w * cos.clamp_min(0.0).square()
        return torch.log1p(pair.sum(dim=(-1, -2))).mean()