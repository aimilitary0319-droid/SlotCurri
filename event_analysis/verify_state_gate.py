"""Verify the separate temporal-mix gate (attn_mass_curriculum.state_p_mult).

Substitutes a fake corrector and a fake residual predictor into LatentProcessor so the
gate is the only thing real code computes, then checks:

  1. gate_p_state=None reproduces the single-threshold path, and the gate enters the
     temporal step exactly once: s + g (c - s) + g D, not the old s + g^2 (c - s) + g D
  2. the decoder gate (returned active_mask) always uses the annealed p, never state_p
  3. the predictor re-gate -- the only gated mix left -- uses state_p
  4. the gate values reproduce the table written into the v24 config
  5. the split actually separates: flat decoder gate, selective state gate
  6. state_max_norm reaches that one mix and nothing else
  7. gradients reach the corrector through both thresholds
"""

import torch
from torch import nn

from slotcurri.modules.video import LatentProcessor

B, S, F, D = 3, 7, 64, 8
DTYPE = torch.float64
TAU = 0.3 / 7


class FakeCorrector(nn.Module):
    def __init__(self, c, masks):
        super().__init__()
        self.c, self.masks = c, masks

    def forward(self, state, inputs, **kwargs):
        return {"slots": self.c, "masks": self.masks}


class FakeResidualPredictor(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, x, vel=None):
        return x + self.delta


def masks_with_mass(target_mass):
    """Slot-axis softmax attention whose gamma=2 sharpened mass hits `target_mass`.

    mass_gamma renormalizes A^2 over slots, so a per-patch attention that is constant
    over patches gives sharpened mass = A_s^2 / sum_s' A_s'^2. Solving for A from the
    desired sharpened mass is just a square root followed by a renormalization.
    """
    a = torch.tensor(target_mass, dtype=DTYPE).sqrt()
    a = a / a.sum()
    return a.view(1, S, 1).expand(B, S, F).contiguous()


def run(proc, state, inputs, gate_p, state_p=None, default_idx=(), max_norm=True):
    # default_idx=() matches `n_default: 0` in the configs. The module's own default is 0,
    # which would pin slot 0 fully active and hide the gate being tested.
    return proc(
        state,
        inputs,
        time_step=1,
        gate_p=gate_p,
        default_idx=default_idx,
        gate_mode="soft",
        gate_tau=TAU,
        mass_gamma=2.0,
        gate_p_state=state_p,
        state_max_norm=max_norm,
    )


def sigmoid_gate(mass, p):
    g = torch.sigmoid((mass - p) / TAU)
    return g / g.amax(dim=-1, keepdim=True)


def main():
    torch.manual_seed(0)
    state = torch.randn(B, S, D, dtype=DTYPE)
    c = torch.randn(B, S, D, dtype=DTYPE)
    delta = torch.randn(1, 1, D, dtype=DTYPE) * 0.3
    inputs = torch.randn(B, F, D, dtype=DTYPE)

    # Masses chosen to span the config's table, normalized to sum to 1 over slots.
    target = [0.50, 0.143, 0.143, 0.05, 0.05, 0.02, 0.01]
    target = [m / sum(target) for m in target]
    masks = masks_with_mass(target)
    mass = torch.tensor(target, dtype=DTYPE).view(1, S).expand(B, S)

    proc = LatentProcessor(
        corrector=FakeCorrector(c, masks),
        predictor=FakeResidualPredictor(delta),
        state_key="slots",
    )

    p_end = 0.1 / 7      # v21 end-of-schedule decoder threshold
    p_state = 0.5 / 7    # v24 state-mix floor
    ok = True

    def closed_form(g):
        """The single gated mix, with Pred(x) = x + D and the prior as the mix reference.

        The corrector output is not gated, so `state` is c for every slot and the gate
        enters once, on the temporal step:

            x_t          = c
            hat{x}_{t+1} = g*Pred(c) + (1-g)*s  =  s + g (c - s) + g D
        """
        return c, g * (c + delta) + (1 - g) * state

    # --- 1. state_p=None keeps the single-gate behaviour ---------------------------
    out_none = run(proc, state, inputs, p_end, state_p=None)
    ref_state, ref_pred = closed_form(sigmoid_gate(mass, p_end).unsqueeze(-1))
    d1 = (out_none["state"] - ref_state).abs().max().item()
    d2 = (out_none["state_predicted"] - ref_pred).abs().max().item()
    print(f"[1] state_p=None matches single-gate closed form: "
          f"state {d1:.2e}, predicted {d2:.2e}")
    ok &= d1 < 1e-12 and d2 < 1e-12
    same = torch.equal(out_none["state_gate"], out_none["active_mask"])
    print(f"    state_gate is active_mask when unset: {same}")
    ok &= same
    # the corrector output must pass through untouched, whatever the gate says
    d_ungated = (out_none["state"] - c).abs().max().item()
    print(f"    corrector output reaches `state` ungated: {d_ungated:.2e}")
    ok &= d_ungated < 1e-12
    # and the observation must arrive at g, not g^2: pin the difference explicitly
    g0 = sigmoid_gate(mass, p_end).unsqueeze(-1)
    lin = state + g0 * (c - state) + g0 * delta
    quad = state + g0.pow(2) * (c - state) + g0 * delta
    d_lin = (out_none["state_predicted"] - lin).abs().max().item()
    d_quad = (out_none["state_predicted"] - quad).abs().max().item()
    print(f"[1b] observation arrives at g ({d_lin:.2e}), not g^2 "
          f"(that form is off by {d_quad:.3e})")
    ok &= d_lin < 1e-12 and d_quad > 1e-6

    # --- 2/3. split: decoder keeps annealed p, the temporal mix uses state_p --------
    out = run(proc, state, inputs, p_end, state_p=p_state)
    ref_state, ref_pred = closed_form(sigmoid_gate(mass, p_state).unsqueeze(-1))
    d_state = (out["state"] - ref_state).abs().max().item()
    d_pred = (out["state_predicted"] - ref_pred).abs().max().item()
    print(f"[2] temporal mix uses state_p: predicted {d_pred:.2e} "
          f"(state stays ungated, {d_state:.2e})")
    ok &= d_state < 1e-12 and d_pred < 1e-12
    # state_p must move only the temporal mix, never the corrector output
    d_same = (out["state"] - out_none["state"]).abs().max().item()
    moved = (out["state_predicted"] - out_none["state_predicted"]).abs().max().item()
    print(f"    changing state_p leaves `state` alone ({d_same:.2e}) "
          f"and moves the next prior ({moved:.3e})")
    ok &= d_same < 1e-12 and moved > 1e-6

    # the returned mask the decoder consumes must be the *annealed* gate, un-normalized
    d_dec = (out["active_mask"] - torch.sigmoid((mass - p_end) / TAU)).abs().max().item()
    print(f"[3] decoder gate still uses annealed p: {d_dec:.2e}")
    ok &= d_dec < 1e-12

    # --- 4. reproduce the table in the v24 config ----------------------------------
    print("[4] g/max(g) entering the temporal mix, by threshold:")
    header = "      p        " + "".join(f"m={m:>6.3f} " for m in target)
    print(header)
    for name, p in (("1.5/7", 1.5 / 7), ("0.5/7", p_state), ("0.1/7", p_end)):
        row = sigmoid_gate(mass, p)[0]
        print(f"      {name:<8} " + "".join(f"{v:>8.3f} " for v in row.tolist()))

    # --- 5. the split separates: flat decoder gate, selective state gate -----------
    g_dec_flat = sigmoid_gate(mass, p_end)[0]
    g_state_sel = sigmoid_gate(mass, p_state)[0]
    spread_dec = (g_dec_flat.max() - g_dec_flat.min()).item()
    spread_state = (g_state_sel.max() - g_state_sel.min()).item()
    print(f"[5] gate spread at p_end: decoder {spread_dec:.3f} vs state {spread_state:.3f}")
    ok &= spread_state > spread_dec
    # effective slot counts, the two quantities logged during training
    print(f"    active_slots {g_dec_flat.sum():.2f} vs "
          f"gate_state_slots {g_state_sel.sum():.2f} (of {S})")

    # --- 5b. default slots stay pinned in both gates after the refactor ------------
    out_d = run(proc, state, inputs, p_end, state_p=p_state, default_idx=(0, 1))
    pinned_dec = out_d["active_mask"][:, :2]
    pinned_state = out_d["state_gate"][:, :2]
    both_one = (pinned_dec == 1).all().item() and (pinned_state == 1).all().item()
    print(f"[5b] default_idx pins slot 0,1 in both gates: {both_one}")
    ok &= both_one

    # --- 5c. state_max_norm now reaches the predictor re-gate and nothing else ------
    raw = run(proc, state, inputs, p_end, state_p=p_state, max_norm=False)
    g_raw = torch.sigmoid((mass - p_state) / TAU).unsqueeze(-1)
    _, ref_raw = closed_form(g_raw)
    d_raw = (raw["state_predicted"] - ref_raw).abs().max().item()
    d_state_same = (raw["state"] - out["state"]).abs().max().item()
    d_pred_diff = (raw["state_predicted"] - out["state_predicted"]).abs().max().item()
    print(f"[5c] max_norm=False uses the raw gate in the temporal mix: {d_raw:.2e}")
    print(f"     toggling it leaves `state` untouched ({d_state_same:.2e}) and moves only "
          f"the next prior ({d_pred_diff:.3e})")
    ok &= d_raw < 1e-12 and d_state_same < 1e-12 and d_pred_diff > 1e-6

    # --- 6. gradients flow through both thresholds --------------------------------
    for state_p in (None, p_state):
        masks_g = masks.clone().requires_grad_(True)
        proc_g = LatentProcessor(
            corrector=FakeCorrector(c, masks_g),
            predictor=FakeResidualPredictor(delta),
            state_key="slots",
        )
        o = run(proc_g, state, inputs, p_end, state_p=state_p)
        (o["state"].sum() + o["state_predicted"].sum()).backward()
        gnorm = masks_g.grad.abs().sum().item()
        finite = torch.isfinite(masks_g.grad).all().item()
        print(f"[6] state_p={state_p}: attention grad norm {gnorm:.3e}, finite {finite}")
        ok &= gnorm > 0 and finite

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
