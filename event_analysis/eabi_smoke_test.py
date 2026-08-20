"""CPU smoke test for the EABI cycle modes in ScanOverTime (no GPU, no checkpoint).

With a stub recurrent cell whose gate/attention depend on the inputs, checks that:
  1. cycle="evidence" anchors differ across batch samples when the evidence peaks differ
  2. the batched run (B=2, mixed anchors) equals the two B=1 runs exactly -- the
     injection/stitch logic never leaks state across samples
  3. frames >= anchor keep the single-pass outputs bit-exactly
  4. frames < anchor match a hand-rolled backward sweep from the anchor
  5. cycle=True (legacy last-frame cycle) matches a hand-rolled reference (regression)
  6. cycle="random" runs and records in-range anchors

Run (CPU is fine):
  python event_analysis/eabi_smoke_test.py
"""
import torch
from torch import nn

from slotcurri.modules.video import ScanOverTime, _evidence_anchors

torch.manual_seed(0)

B, T, S, D, F_ = 2, 9, 4, 8, 16


class StubCell(nn.Module):
    """Minimal cell with the output keys ScanOverTime's cycle paths rely on."""

    def forward(self, state, inputs, time_step=None, **kwargs):
        new_state = state + inputs.mean(dim=1, keepdim=True)
        logits = torch.einsum("bsd,bfd->bsf", new_state, inputs)
        att = logits.softmax(dim=1)
        gate = torch.sigmoid(new_state.mean(-1))
        pred = new_state * 1.1
        return {
            "state": new_state,
            "state_predicted": pred,
            "state_predicted_pregate": pred,
            "corrector": {"slots": new_state},
            "state_attn_mask": att,
            "active_mask": gate,
            "state_gate": gate,
        }


def forward_sweep(cell, init, inputs):
    outs, state = [], init
    for t in range(inputs.shape[1]):
        o = cell(state, inputs[:, t], t)
        outs.append(o)
        state = o["state_predicted"]
    return outs


def backward_from(cell, outs, inputs, anchor):
    """Hand-rolled EABI reference: stitched per-frame outputs for one sample."""
    stitched = list(outs)
    state = outs[anchor]["state_predicted"]
    for t in range(anchor - 1, -1, -1):
        o = cell(state, inputs[:, t])
        stitched[t] = o
        state = o["state_predicted"]
    return stitched


def legacy_reference(cell, init, inputs):
    outs = forward_sweep(cell, init, inputs)
    t_len = inputs.shape[1]
    new = [None] * t_len
    new[t_len - 1] = outs[-1]
    state = outs[-1]["state_predicted"]
    for t in range(t_len - 2, -1, -1):
        o = cell(state, inputs[:, t])
        new[t] = o
        state = o["state_predicted"]
    return new


def assert_tree_equal(tree, ref_outs, frames, sample, label):
    """Compare stacked-tree leaves at `frames` of `sample` against per-frame ref dicts."""
    for key in ("state", "state_predicted", "state_attn_mask", "active_mask"):
        for t in frames:
            got = tree[key][sample, t]
            want = ref_outs[t][key][0] if ref_outs[t][key].shape[0] == 1 else ref_outs[t][key][sample]
            assert torch.equal(got, want), f"{label}: mismatch at key={key} t={t} b={sample}"
    for t in frames:
        got = tree["corrector"]["slots"][sample, t]
        want = (
            ref_outs[t]["corrector"]["slots"][0]
            if ref_outs[t]["corrector"]["slots"].shape[0] == 1
            else ref_outs[t]["corrector"]["slots"][sample]
        )
        assert torch.equal(got, want), f"{label}: corrector mismatch at t={t} b={sample}"


def main():
    cell = StubCell()
    scan = ScanOverTime(cell, next_state_key="state_predicted", pass_step=True)

    init = torch.randn(B, S, D) * 0.1
    inputs = torch.randn(B, T, F_, D) * 0.1
    # shape the evidence: sample 0 peaks mid-clip (rise then fall), sample 1 at the end
    for t in range(T):
        inputs[0, t] += 1.0 if t < 5 else -2.0
        inputs[1, t] += 0.5

    # --- 1/2/3/4: both anchored modes, batched vs per-sample ---
    for mode, stat in (("evidence", "count"), ("evidence_mass", "mass")):
        tree = scan(init, inputs, cycle=mode, mass_gamma=2.0)
        anchors = scan.last_anchor_frames
        assert anchors is not None and anchors.shape == (B,)
        print(f"{mode} anchors: {anchors.tolist()}  (T={T})")
        assert anchors.max().item() > 0, f"degenerate test ({mode}): all anchors at 0"

        for b in range(B):
            outs_b = forward_sweep(cell, init[b : b + 1], inputs[b : b + 1])
            anchor_b = int(_evidence_anchors(outs_b, 2.0, stat=stat)[0].item())
            assert anchor_b == int(anchors[b].item()), (
                f"anchor mismatch {mode} b={b}: batched {int(anchors[b])} vs {anchor_b}"
            )
            ref = backward_from(cell, outs_b, inputs[b : b + 1], anchor_b)
            assert_tree_equal(tree, ref, range(T), b, f"{mode} b={b}")
            # frames >= anchor must equal the plain forward sweep bit-exactly
            assert_tree_equal(tree, outs_b, range(anchor_b, T), b, f"post-anchor b={b}")
    # the crafted inputs must exercise mixed anchors in one batch at least once
    tree = scan(init, inputs, cycle="evidence", mass_gamma=2.0)
    a = scan.last_anchor_frames
    assert a[0].item() != a[1].item(), "test wants mixed anchors in the batch"
    print("anchored modes: batched == per-sample reference, post-anchor frames untouched")

    # --- 5: legacy cycle regression ---
    tree_legacy = scan(init, inputs, cycle=True, mass_gamma=2.0)
    ref_legacy = legacy_reference(cell, init, inputs)
    assert_tree_equal(tree_legacy, ref_legacy, range(T), 0, "legacy b=0")
    assert_tree_equal(tree_legacy, ref_legacy, range(T), 1, "legacy b=1")
    assert scan.last_anchor_frames is None  # only set by the anchored modes
    print("legacy cycle: matches hand-rolled reference")

    # --- 6: random mode ---
    torch.manual_seed(123)
    tree_rand = scan(init, inputs, cycle="random", mass_gamma=2.0)
    ra = scan.last_anchor_frames
    assert ra is not None and ((ra >= 0) & (ra < T)).all()
    assert tree_rand["state"].shape == (B, T, S, D)
    print(f"random mode: anchors {ra.tolist()}, tree shapes ok")

    # --- plain single pass unaffected ---
    tree_single = scan(init, inputs, cycle=False, mass_gamma=2.0)
    outs0 = forward_sweep(cell, init, inputs)
    assert_tree_equal(tree_single, outs0, range(T), 0, "single b=0")
    print("single pass: unchanged")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
