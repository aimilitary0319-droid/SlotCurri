"""How much does multiplying g into the decoder actually change the masks?

The decoder applies the gate as masks = softmax_s(alpha) * g renormalized over slots, which is
softmax_s(alpha + log g). Differentiating, d m_s / d log g_s = m_s (1 - m_s): the gate has
leverage where a patch is contested between slots and none where one slot already owns it.
Whether that leaves the gate with real influence is an empirical question about how contested
the model's masks are at a given point in training, so this measures it on real checkpoints.

Each batch is run once, then the decoder masks are rebuilt twice from the same captured alpha
logits -- once with the trained gate, once with g == 1 -- and compared.

Usage (inside the container, one GPU):
  python event_analysis/decoder_gate_leverage.py CONFIG CKPT STEP
"""

import sys
from typing import Dict, List, Optional

import torch

from slotcurri import configuration, data, metrics, models

N_BATCHES = 20


class _FakeTrainer:
    def __init__(self, step: int):
        self.global_step = step


def gated_masks(alpha: torch.Tensor, g: Optional[torch.Tensor]) -> torch.Tensor:
    """The decoder's own expression, applied to captured alpha logits."""
    m = torch.softmax(alpha, dim=1)
    if g is not None:
        m = m * g
        m = m / m.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return m


def main() -> int:
    cfg_path, ckpt_path = sys.argv[1], sys.argv[2]
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 100000

    config = configuration.load_config(cfg_path)
    config.model.visualize = False
    val_metrics = {n: metrics.build(c) for n, c in config.val_metrics.items()}
    model = models.build(config.model, config.optimizer, None, val_metrics)
    if ckpt_path != "none":
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded {ckpt_path}")
        print(f"  missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}")
    else:
        print("untrained model (random init)")
    model = model.cuda().eval()
    model._trainer = _FakeTrainer(step)

    dataset = data.build(config.dataset)
    dataset.setup("validate")
    loader = dataset.val_dataloader()

    # capture the alpha logits so both mask variants share exactly the same ones
    seen: Dict[str, torch.Tensor] = {}
    dec = model.decoder.module if hasattr(model.decoder, "module") else model.decoder
    orig_mlp = dec.mlp.forward

    def wrapped_mlp(x):
        out = orig_mlp(x)
        seen["alpha"] = out[..., dec.outp_dim :]
        return out

    dec.mlp.forward = wrapped_mlp

    stats: List[Dict[str, float]] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= N_BATCHES:
                break
            batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model.forward(batch, train=False, cycle=False)
            g = out["processor"]["active_mask"].float()
            alpha = seen["alpha"].float()
            g_flat = g.flatten(0, 1)
            while g_flat.dim() < alpha.dim():
                g_flat = g_flat.unsqueeze(-1)
            m_gated = gated_masks(alpha, g_flat)
            m_plain = gated_masks(alpha, None)

            # the model's own masks, as a check that the recomputation is faithful
            ref = out["decoder"]["masks"].float().flatten(0, 1)
            recomputed = m_gated.squeeze(-1)
            if ref.shape == recomputed.shape:
                agree = (ref - recomputed).abs().max().item()
            else:
                agree = float("nan")

            top = m_gated.max(dim=1).values           # (BT, P, 1)
            lev = (m_gated * (1 - m_gated)).sum(1)    # (BT, P, 1)
            stats.append({
                "recompute_err": agree,
                "top_mask": top.mean().item(),
                "frac_owned_99": (top > 0.99).float().mean().item(),
                "frac_contested": (top < 0.9).float().mean().item(),
                "leverage": lev.mean().item(),
                "l1": (m_gated - m_plain).abs().sum(1).mean().item(),
                "argmax_flip": (m_gated.argmax(1) != m_plain.argmax(1)).float().mean().item(),
                "g_min": g.amin().item(),
                "g_max": g.amax().item(),
                "g_ratio": (g.amax(-1) / g.amin(-1).clamp_min(1e-8)).mean().item(),
            })

    dec.mlp.forward = orig_mlp
    avg = {k: sum(s[k] for s in stats) / len(stats) for k in stats[0]}

    print(f"\n{len(stats)} batches, gate schedule evaluated at step {step:,}")
    print(f"  recomputation vs model masks : max abs diff {avg['recompute_err']:.2e}")
    print(f"  gate range                   : {avg['g_min']:.4f} .. {avg['g_max']:.4f} "
          f"(mean per-frame max/min {avg['g_ratio']:.2f}x)")
    print(f"  mean max_s m_s per patch     : {avg['top_mask']:.4f}   (1.0 = one slot owns it)")
    print(f"  patches with max_s m > 0.99  : {avg['frac_owned_99'] * 100:.1f}%  <- gate powerless")
    print(f"  patches with max_s m < 0.90  : {avg['frac_contested'] * 100:.1f}%  <- gate has leverage")
    print(f"  mean sum_s m(1-m)            : {avg['leverage']:.4f}   (max 1-1/S = 0.857)")
    print("  effect of the gate on the masks, same alpha:")
    print(f"    mean L1 |m_gated - m_plain| : {avg['l1']:.4f}   (0 = no change, 2 = total)")
    print(f"    patches whose argmax flips  : {avg['argmax_flip'] * 100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
