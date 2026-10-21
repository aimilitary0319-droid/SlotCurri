import torch

from slotcurri import configuration, losses, ssync
from slotcurri.modules.groupers import SlotAttention
from slotcurri.modules.video import LatentProcessor, ScanOverTime


def _maps(dots, attn, alpha):
    return ssync.selective_synergy_targets(
        dots, attn, alpha, merge_iou_threshold=0.7
    )


def test_interior_label_follows_alpha_not_a_gated_argmax():
    # Grouper is one flat slot, so every patch is interior.
    # Decoder alpha votes for slot 1. A detached gate π=[1, 0] would flip
    # argmax(softmax(α) * π) to slot 0; the label must stay slot 1.
    dots = torch.zeros(1, 1, 2, 4)
    dots[:, :, 0] = 5.0
    attn = torch.zeros(1, 1, 2, 4)
    attn[:, :, 0] = 1.0
    alpha = torch.zeros(1, 1, 2, 4)
    alpha[:, :, 1] = 8.0

    gated = torch.softmax(alpha, dim=2) * torch.tensor([1.0, 0.0]).view(1, 1, 2, 1)
    assert torch.argmax(gated, dim=2).eq(0).all()

    out = _maps(dots, attn, alpha)
    assert out["dec_nbd_labels"].numel() == 4
    assert out["dec_nbd_labels"].eq(1).all()
    assert out["grp_bd_labels"].numel() == 0
    assert out["dec_bd_preds"].shape == (0, 2)
    assert out["grp_nbd_preds"].shape == (4, 2)


def test_merge_uses_dot_coverage_above_softmax_coverage():
    # Both slots sit above their own dot mean on patch 0 only, so dot IoU is 1.
    # Softmax mass of the loser is below its mean there and above it elsewhere,
    # so softmax-above-mean IoU is 0 and would not merge.
    dots = torch.zeros(1, 2, 2, 4)
    dots[:, :, 0, 0] = 10.0
    dots[:, :, 1, 0] = 9.0
    attn = torch.zeros(1, 2, 2, 4)
    attn[:, :, 0] = 1.0
    attn[:, :, 0, 0] = 0.0
    attn[:, :, 1, 0] = 1.0
    alpha = torch.zeros(1, 2, 2, 4)
    alpha[:, :, 1] = 4.0

    soft = torch.softmax(dots, dim=2)
    soft_mean = soft.transpose(1, 2).flatten(2, 3).mean(-1)
    soft_cov = soft > soft_mean.unsqueeze(1).unsqueeze(-1)
    inter = (soft_cov[:, :, 0] & soft_cov[:, :, 1]).flatten(1).sum(-1)
    union = (soft_cov[:, :, 0] | soft_cov[:, :, 1]).flatten(1).sum(-1)
    assert (inter / union.clamp_min(1)).max() == 0

    merged = _maps(dots, attn, alpha)
    assert merged["dec_nbd_labels"].eq(0).all()
    assert merged["grp_bd_labels"].numel() == 0

    kept = ssync.selective_synergy_targets(
        dots, attn, alpha, merge_iou_threshold=1.1
    )
    assert kept["dec_nbd_labels"].eq(1).all()


def test_only_slots_above_iou_threshold_merge():
    # Slots 0 and 1 cover the same three patches. Slot 2 owns the last patch.
    dots = torch.zeros(1, 2, 3, 4)
    dots[:, :, 0, :3] = 5.0
    dots[:, :, 1, :3] = 4.0
    dots[:, :, 2, 3] = 5.0
    attn = torch.zeros(1, 2, 3, 4)
    attn[:, :, 0, 0] = 1.0
    attn[:, :, 0, 1] = 1.0
    attn[:, :, 1, 2] = 1.0
    attn[:, :, 2, 3] = 1.0
    alpha = torch.zeros(1, 2, 3, 4)
    alpha[:, :, 1] = 3.0

    merged = _maps(dots, attn, alpha)
    assert not torch.isin(merged["grp_bd_labels"], torch.tensor(1)).any()
    assert not torch.isin(merged["dec_nbd_labels"], torch.tensor(1)).any()
    assert torch.isin(merged["grp_bd_labels"], torch.tensor(2)).any()

    kept = ssync.selective_synergy_targets(
        dots, attn, alpha, merge_iou_threshold=1.1
    )
    assert torch.isin(kept["grp_bd_labels"], torch.tensor(1)).any()
    assert kept["dec_nbd_labels"].eq(1).all()


def test_scan_over_time_stacks_dots():
    grouper = SlotAttention(4, 4, n_iters=1, use_mlp=False, use_gru=False)
    scanner = ScanOverTime(LatentProcessor(grouper, predictor=None))
    state = torch.randn(2, 3, 4)
    features = torch.randn(2, 4, 6, 4)
    out = scanner(state, features)
    assert out["nosoftmax_dots"].shape == (2, 4, 3, 6)
    assert torch.allclose(
        torch.softmax(out["nosoftmax_dots"], dim=2), out["state_attn_mask"]
    )


def test_msecls_empty_selection_has_grad():
    loss = losses.MSECLSLoss(pred_key="dec_bd_preds", target_key="grp_bd_labels")
    logits = torch.zeros(0, 3, requires_grad=True)
    outputs = {
        "dec_bd_preds": logits,
        "grp_bd_labels": torch.zeros(0, dtype=torch.long),
    }
    value = loss(loss.get_prediction(outputs), loss.get_target({}, outputs))
    value.backward()
    assert logits.grad is not None
    assert value.detach() == 0


def test_ssync_config_loads_on_top_of_v39lam1gu():
    cfg = configuration.load_config(
        "configs/slotcurri/movi_c_attnmass_v39lam1gu_ssync.yaml"
    )
    parent = configuration.load_config(
        "configs/slotcurri/movi_c_attnmass_v39lam1gu.yaml"
    )
    assert cfg.experiment_name == "movi_c_attnmass_v39lam1gu_ssync"
    assert cfg.model.synergistic_start_step == 50000
    assert cfg.model.merge_iou_threshold == 0.7
    assert cfg.model.losses.loss_grp2deccls.name == "MSECLSLoss"
    assert cfg.model.losses.loss_dec2grpcls.pred_key == "grp_nbd_preds"
    assert cfg.model.loss_weights.loss_grp2deccls == 1.0
    assert cfg.model.loss_weights.loss_featrec == parent.model.loss_weights.loss_featrec
    assert cfg.model.attn_mass_curriculum.conf_kind == "spectral_graph_n8_lam1"
    assert cfg.model.feature_curriculum.anneal_steps == 50000
    assert parent.model.synergistic_start_step is None
    assert "loss_grp2deccls" not in parent.model.losses
