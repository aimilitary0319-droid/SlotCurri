"""Does vel_gain=0 actually freeze vel_proj, or only stall it for one step?

The gradient of vel_proj is proportional to vel_gain, so at exactly zero it vanishes. But
vel_gain's own gradient does not vanish there, so it should escape zero on the first step
and release vel_proj on the second. This runs Adam on a fixed toy objective and reports,
per step, whether each parameter moved.

Usage:
  python event_analysis/probe_vel_gain_init.py
"""

import torch

from slotcurri.modules.networks import TransformerEncoder

B, S, D = 4, 7, 64
STEPS = 6


def run(gain_init, lr=1e-3, steps=STEPS):
    torch.manual_seed(0)
    pred = TransformerEncoder(dim=D, n_blocks=1, n_heads=4, vel_dim=D, vel_gain_init=gain_init)
    blk = pred.blocks[0]
    opt = torch.optim.Adam(pred.parameters(), lr=lr)

    torch.manual_seed(1)
    x = torch.randn(B, S, D)
    v = torch.randn(B, S, D)
    # a target that genuinely depends on velocity, so using v is the way to reduce the loss
    target = x + 0.5 * v

    print(f"\n=== vel_gain_init = {gain_init} ===")
    print(f"{'step':>4}  {'loss':>9}  {'|vel_gain|':>11}  {'|dL/dWp|':>10}  "
          f"{'|Wp - Wp(0)|':>13}  {'|dL/dgain|':>11}")
    w0 = blk.vel_proj.weight.detach().clone()
    for i in range(steps):
        opt.zero_grad()
        loss = (pred(x, vel=v) - target).pow(2).mean()
        loss.backward()
        g_wp = blk.vel_proj.weight.grad.abs().max().item()
        g_gain = blk.vel_gain.grad.abs().max().item()
        moved = (blk.vel_proj.weight.detach() - w0).abs().max().item()
        print(f"{i:>4}  {loss.item():>9.6f}  {blk.vel_gain.detach().abs().max():>11.3e}  "
              f"{g_wp:>10.3e}  {moved:>13.3e}  {g_gain:>11.3e}")
        opt.step()
    final = (pred(x, vel=v) - target).pow(2).mean().item()
    return final


def main():
    print("Toy objective: predict x + 0.5*v, so velocity carries the answer.")
    print("Adam, lr=1e-3. Watch whether |dL/dWp| leaves zero after step 0.")
    losses = {g: run(g) for g in (0.0, 0.1, 1.0)}
    print("\nloss after", STEPS, "steps:")
    for g, l in losses.items():
        print(f"  vel_gain_init={g:<5} -> {l:.6f}")


if __name__ == "__main__":
    main()
