"""Selective synergy pseudo-labels for video object-centric learning.

Boundary labels come from the grouper's softmax attention. Interior labels
come from argmax of the decoder's raw alpha, before the λ1 gate. Slot merging
uses pre-softmax dots, and only the pseudo-labels change.
"""

from typing import Dict, List

import torch
import torch.nn.functional as F


def neighbor_patches(padded_labels: torch.Tensor) -> List[torch.Tensor]:
    """8 spatial neighbors plus the previous and next frame.

    ``padded_labels`` is (B, T+2, H+2, W+2). Each returned tensor is (B, T, H, W).
    """
    top_left = padded_labels[:, 1:-1, :-2, :-2]
    top_center = padded_labels[:, 1:-1, :-2, 1:-1]
    top_right = padded_labels[:, 1:-1, :-2, 2:]
    middle_left = padded_labels[:, 1:-1, 1:-1, :-2]
    middle_right = padded_labels[:, 1:-1, 1:-1, 2:]
    bottom_left = padded_labels[:, 1:-1, 2:, :-2]
    bottom_center = padded_labels[:, 1:-1, 2:, 1:-1]
    bottom_right = padded_labels[:, 1:-1, 2:, 2:]
    prev_time = padded_labels[:, :-2, 1:-1, 1:-1]
    next_time = padded_labels[:, 2:, 1:-1, 1:-1]
    return [
        top_left,
        top_center,
        top_right,
        middle_left,
        middle_right,
        bottom_left,
        bottom_center,
        bottom_right,
        prev_time,
        next_time,
    ]


def _square_hw(n_patches: int) -> int:
    side = int(n_patches**0.5)
    if side * side != n_patches:
        raise ValueError(
            f"SSync maps expect a square patch grid, got {n_patches} patches"
        )
    return side


def _boundary_masks(labels: torch.Tensor, dif_n: int, dif_n_noise: int):
    """True where at least ``dif_n`` neighbors disagree and the pixel is not isolated."""
    padded = F.pad(labels.float(), (1, 1, 1, 1, 1, 1), mode="replicate").long()
    neighbors = neighbor_patches(padded)
    diff = torch.zeros_like(labels)
    for neighbor in neighbors:
        diff += labels != neighbor
    is_boundary = diff >= dif_n
    is_not_noise = (len(neighbors) - diff) >= dif_n_noise
    return is_boundary & is_not_noise, diff


def selective_synergy_targets(
    dots: torch.Tensor,
    attn: torch.Tensor,
    alpha: torch.Tensor,
    *,
    dif_n_bd: int = 1,
    dif_n_nbd: int = 1,
    dif_n_bd_noise: int = 1,
    merge_iou_threshold: float = 0.7,
) -> Dict[str, torch.Tensor]:
    """Build the four SSync tensors.

    ``dots``, ``attn`` and ``alpha`` are (B, T, S, HW). ``dots`` are pre-softmax
    grouper logits, ``attn`` is softmax-over-slots attention, and ``alpha`` is
    the ungated decoder logit.
    """
    if dots.ndim != 4 or attn.shape != dots.shape or alpha.shape != dots.shape:
        raise ValueError(
            "SSync expected dots, attn and alpha with shape (B, T, S, HW), "
            f"got {tuple(dots.shape)}, {tuple(attn.shape)}, {tuple(alpha.shape)}"
        )
    batch, time, n_slots, n_patches = dots.shape
    side = _square_hw(n_patches)
    device = dots.device

    grp_labels = torch.argmax(attn, dim=2).reshape(batch, time, side, side)
    dec_labels = torch.argmax(alpha, dim=2).reshape(batch, time, side, side)
    dots_hw = dots.permute(0, 1, 3, 2).reshape(batch, time, side, side, n_slots)
    alpha_hw = alpha.permute(0, 1, 3, 2).reshape(batch, time, side, side, n_slots)

    # Coverage is dots above each slot's own space-time mean, not softmax mass.
    per_slot_mean = dots.transpose(1, 2).flatten(2, 3).mean(-1)  # B S
    coverage = dots > per_slot_mean.unsqueeze(1).unsqueeze(-1)  # B T S HW
    flat = coverage.reshape(batch, time, n_slots, -1).float()
    areas = flat.sum(-1) + 1e-6
    flat_bt = flat.reshape(batch * time, n_slots, -1)
    intersection = torch.bmm(flat_bt, flat_bt.transpose(1, 2)).reshape(
        batch, time, n_slots, n_slots
    )
    union = areas.unsqueeze(-1) + areas.unsqueeze(-2) - intersection
    slot_iou = torch.where(
        union > 0, intersection / union, torch.zeros_like(union)
    ).mean(dim=1)

    grp_area = F.one_hot(grp_labels, num_classes=n_slots).sum(dim=(1, 2, 3)).float()
    eye = torch.eye(n_slots, device=device, dtype=torch.bool)
    merged_grp = grp_labels.clone()
    merged_dec = dec_labels.clone()
    for b in range(batch):
        valid_pairs = (slot_iou[b] > merge_iou_threshold) & (~eye)
        if not valid_pairs.any():
            continue
        redundant = valid_pairs | valid_pairs.t()
        visited = torch.zeros(n_slots, device=device, dtype=torch.bool)
        areas_b = grp_area[b]
        for i in range(n_slots):
            if visited[i] or not redundant[i].any():
                continue
            stack = [i]
            component = []
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                component.append(node)
                for n in torch.nonzero(redundant[node]).flatten().tolist():
                    if not visited[n]:
                        stack.append(int(n))
            if len(component) <= 1:
                continue
            comp = torch.tensor(component, device=device)
            winner = comp[areas_b[comp].argmax()].item()
            for loser in component:
                if loser == winner:
                    continue
                merged_grp[b][merged_grp[b] == loser] = winner
                merged_dec[b][merged_dec[b] == loser] = winner

    is_boundary, _ = _boundary_masks(merged_grp, dif_n_bd, dif_n_bd_noise)
    _, dec_diff = _boundary_masks(merged_dec, dif_n_nbd, dif_n_bd_noise)
    is_interior = dec_diff < dif_n_nbd

    grp_bd_labels = []
    dec_bd_preds = []
    dec_nbd_labels = []
    grp_nbd_preds = []
    for b in range(batch):
        for t in range(time):
            grp_bd_labels.append(torch.masked_select(merged_grp[b, t], is_boundary[b, t]))
            dec_nbd_labels.append(torch.masked_select(merged_dec[b, t], is_interior[b, t]))
            dec_bd_preds.append(alpha_hw[b, t][is_boundary[b, t]])
            grp_nbd_preds.append(dots_hw[b, t][is_interior[b, t]])

    return {
        "grp_bd_labels": torch.cat(grp_bd_labels, dim=0),
        "dec_bd_preds": torch.cat(dec_bd_preds, dim=0),
        "dec_nbd_labels": torch.cat(dec_nbd_labels, dim=0),
        "grp_nbd_preds": torch.cat(grp_nbd_preds, dim=0),
    }
