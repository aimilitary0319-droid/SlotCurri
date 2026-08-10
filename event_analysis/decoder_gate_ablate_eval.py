"""What do the reported metrics lose if the decoder simply ignores g?

`decoder_gate_leverage.py` measures how far the gate moves the masks; this measures whether that
movement is worth anything. Both passes use the same weights and the same batches, and the gate
still drives the temporal path in both -- the only difference is whether the decoder receives
`active_mask` or None, i.e. whether the masks are softmax_s(alpha + log g) or softmax_s(alpha).

This is an eval-time intervention on a model trained with the gate, so it answers "what is this
multiplication contributing to the masks this model produces", not "would training without it
have worked".

Usage (inside the container, one GPU):
  python event_analysis/decoder_gate_ablate_eval.py CONFIG CKPT STEP [N_BATCHES]
"""

import sys
from typing import Any, Dict

import torch

from slotcurri import configuration, data, metrics, models


class _FakeTrainer:
    def __init__(self, step: int):
        self.global_step = step


def collect(model, loader, n_batches: int, drop_gate: bool, val_metrics) -> None:
    dec = model.decoder.module if hasattr(model.decoder, "module") else model.decoder
    orig = dec.forward
    if drop_gate:
        dec.forward = lambda slots, active_mask=None: orig(slots, None)
    try:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break
                batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
                if "batch_padding_mask" in batch:
                    batch = model._remove_padding(batch, batch["batch_padding_mask"])
                    if batch is None:
                        continue
                outputs = model.forward(batch, train=False, cycle=model.cyclic_inference)
                aux = model.aux_forward(batch, outputs)
                for metric in val_metrics.values():
                    metric.update(**batch, **outputs, **aux)
    finally:
        dec.forward = orig


def flatten(name: str, value: Any, into: Dict[str, float]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{name}/{k}", v, into)
    else:
        into[name] = float(value)


def main() -> int:
    cfg_path, ckpt_path = sys.argv[1], sys.argv[2]
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
    n_batches = int(sys.argv[4]) if len(sys.argv) > 4 else 40

    config = configuration.load_config(cfg_path)
    config.model.visualize = False
    built = {n: metrics.build(c) for n, c in config.val_metrics.items()}
    model = models.build(config.model, config.optimizer, None, built)
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)
    model = model.cuda().eval()
    model._trainer = _FakeTrainer(step)

    dataset = data.build(config.dataset)
    dataset.setup("validate")

    results = {}
    for label, drop in (("with gate", False), ("gate dropped", True)):
        for m in model.val_metrics.values():
            m.reset()
        collect(model, dataset.val_dataloader(), n_batches, drop, model.val_metrics)
        flat: Dict[str, float] = {}
        for key, m in model.val_metrics.items():
            flatten(key, m.compute(), flat)
        results[label] = flat

    a, b = results["with gate"], results["gate dropped"]
    print(f"\n{n_batches} val batches, {ckpt_path}, gate schedule at step {step:,}")
    print(f"  {'metric':<34}{'with gate':>12}{'g ignored':>12}{'delta':>10}")
    for key in a:
        print(f"  {key:<34}{a[key]:>12.4f}{b[key]:>12.4f}{a[key] - b[key]:>+10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
