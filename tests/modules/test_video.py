import torch

from slotcurri.modules import video


def test_map_over_time():
    class SingleModule(torch.nn.Module):
        def forward(self, inp):
            assert inp.ndim == 2
            return inp

    batch_size, seq_len, dims = 3, 4, 5

    time_mapper = video.MapOverTime(SingleModule())

    inp = torch.ones(batch_size, seq_len, dims)
    assert torch.allclose(time_mapper(inp), inp)


def test_scan_over_time():
    class RecurrentCell(torch.nn.Module):
        # ScanOverTime always forwards its gate kwargs; a wrapped module that does not
        # gate simply ignores them (mirrors LatentProcessor's signature).
        def forward(self, state, inputs, **kwargs):
            assert state.ndim == 2
            assert inputs.ndim == 2
            state_next = state + inputs
            return {"state": state, "state_next": state_next, "aux": {"state_next": state_next}}

    batch_size, seq_len, dims = 3, 4, 1
    scanner = video.ScanOverTime(RecurrentCell(), next_state_key="state_next", pass_step=False)

    initial_state = torch.zeros(batch_size, dims)
    inputs = torch.ones(batch_size, seq_len, dims)

    outputs = scanner(initial_state, inputs)

    assert torch.allclose(outputs["state"][:, 0], initial_state)
    assert torch.allclose(outputs["state"][:, 1:], torch.cumsum(inputs, dim=1)[:, :-1])
    assert torch.allclose(outputs["state_next"], torch.cumsum(inputs, dim=1))
    assert torch.allclose(outputs["aux"]["state_next"], torch.cumsum(inputs, dim=1))


def test_evidence_sum_anchor_ignores_purity():
    """cycle=evidence_sum uses E_t = sum_s g_s; it must not look at attention c."""
    n_slots, n_feat = 4, 16
    # t=0: g=1, uniform A (low c). t=1: g=0.1, one-hot A (c=1).
    outs = [
        {
            "active_mask": torch.ones(2, n_slots),
            "state_attn_mask": torch.ones(2, n_slots, n_feat) / float(n_feat),
        },
        {
            "active_mask": torch.full((2, n_slots), 0.1),
            "state_attn_mask": torch.zeros(2, n_slots, n_feat),
        },
    ]
    outs[1]["state_attn_mask"][:, :, 0] = 1.0
    a_sum = video._evidence_anchors(outs, 1.0, window=1, stat="sum")
    a_count = video._evidence_anchors(outs, 1.0, window=1, stat="count")
    assert torch.equal(a_sum, torch.zeros(2, dtype=torch.long))
    assert torch.equal(a_count, torch.ones(2, dtype=torch.long))


def test_eval_hard_threshold():
    g = torch.tensor([[0.49, 0.50, 0.97]])
    h = video._eval_hard_threshold(g, 0.5)
    assert h.dtype == torch.bool
    assert torch.equal(h, torch.tensor([[False, True, True]]))
    assert video._eval_hard_threshold(g, 0.0) is g
    assert video._eval_hard_threshold(h, 0.5) is h
