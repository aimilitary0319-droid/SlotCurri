"""Step-0 go/no-go for purity gating (v32) on already-trained checkpoints.

The v32 gate has no learned parameters and no schedule -- it is a pure runtime
modulation (decoder masks softmax(alpha + log c), temporal mix c / max(c) with
c = sg(ownership purity)). Any earlier checkpoint can therefore be evaluated under it
by flipping the model's amc_* attributes, no retraining involved:

  as-trained : the checkpoint's own gating (baseline = ungated; v26m / v29 = their
               threshold gate at the eval end values)
  purity     : v32 gate, decoder + temporal (conf_kind = purity_sharp)
  purity-dec : v32 gate on the decoder only (temporal mix left ungated) -- isolates
               how much of the effect is mask reweighting vs temporal selection
  purity-raw : conf_kind = purity (unsharpened attention), decoder + temporal

This is an off-policy check (the weights were shaped by a different gate), so it
answers "does ownership reweighting improve the masks this model already produces".
The decision rule for spending the 100k v32 run: the baseline checkpoint must improve
under `purity`. v26m / v29 quantify how the swap compares against a live mass gate.

Usage (inside the container, one GPU):
  python event_analysis/purity_weight_eval.py \
      --config configs/slotcurri/ytvis2021.yaml \
      --ckpt "logs/_ytvis/checkpoints/slotcurri_step=step=100000-v1.ckpt" \
      --max-clips 200
"""
import argparse
from typing import Any, Dict

import torch

from slotcurri import configuration, data, metrics, models


class _FakeTrainer:
    def __init__(self, step: int):
        self.global_step = step


# Everything the purity_weight path reads at forward time. amc_* attrs exist on every
# model (the amc block is parsed with defaults even when gating is disabled), so the
# same overrides work on baseline and gated checkpoints alike.
PURITY_OVERRIDES: Dict[str, Any] = {
    "attn_mass_enabled": True,
    "amc_gate_form": "purity_weight",
    "amc_conf_kind": "purity_sharp",
    "amc_mass_gamma": 2.0,
    "amc_gate_mode": "soft",
    "amc_state_max_norm": True,
    "amc_predictor_ungated": False,
    "amc_default_idx": [],
    "amc_p_mode": "absolute",
    "amc_purity_q_end": None,
    "amc_state_p_mult": None,
    "amc_gate_detach": False,
}

MODES: Dict[str, Dict[str, Any]] = {
    "as-trained": {},
    "purity": dict(PURITY_OVERRIDES),
    "purity-dec": {**PURITY_OVERRIDES, "amc_predictor_ungated": True},
    "purity-raw": {**PURITY_OVERRIDES, "amc_conf_kind": "purity"},
}


def flatten(name: str, value: Any, into: Dict[str, float]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{name}/{k}", v, into)
    else:
        into[name] = float(value)


@torch.no_grad()
def run_mode(model, loader, max_clips: int, overrides: Dict[str, Any]):
    saved = {k: getattr(model, k) for k in overrides}
    for k, v in overrides.items():
        setattr(model, k, v)
    gate_sum = gate_min_sum = 0.0
    gate_n = 0
    try:
        for m in model.val_metrics.values():
            m.reset()
        n_clips = 0
        for batch in loader:
            batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
            if "batch_padding_mask" in batch:
                batch = model._remove_padding(batch, batch["batch_padding_mask"])
                if batch is None:
                    continue
            outputs = model.forward(batch, train=False, cycle=model.cyclic_inference)
            aux = model.aux_forward(batch, outputs)
            for metric in model.val_metrics.values():
                metric.update(**batch, **outputs, **aux)
            gate = outputs["processor"].get("active_mask")
            if gate is not None and gate.dtype != torch.bool:
                g = gate.float()
                gate_sum += float(g.mean())
                gate_min_sum += float(g.amin(dim=-1).mean())  # weakest slot's gate
                gate_n += 1
            n_clips += outputs["batch_size"]
            if n_clips >= max_clips:
                break
        results: Dict[str, float] = {}
        for name, metric in model.val_metrics.items():
            flatten(name, metric.compute(), results)
        if gate_n:
            results["gate_mean"] = gate_sum / gate_n
            results["gate_min"] = gate_min_sum / gate_n
        return results, n_clips
    finally:
        for k, v in saved.items():
            setattr(model, k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/slotcurri/ytvis2021.yaml")
    ap.add_argument(
        "--ckpt", default="logs/_ytvis/checkpoints/slotcurri_step=step=100000-v1.ckpt"
    )
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--max-clips", type=int, default=200)
    ap.add_argument(
        "--modes", nargs="+", default=["as-trained", "purity", "purity-dec"],
        choices=sorted(MODES),
    )
    args = ap.parse_args()

    config = configuration.load_config(args.config)
    config.model.visualize = False
    built = {n: metrics.build(c) for n, c in config.val_metrics.items()}
    model = models.build(config.model, config.optimizer, None, built)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd, strict=False)
    model = model.cuda().eval()
    model._trainer = _FakeTrainer(100000)

    dataset = data.build(config.dataset, data_dir=args.data_dir)
    dataset.setup("validate")

    all_results: Dict[str, Dict[str, float]] = {}
    for mode in args.modes:
        res, n = run_mode(model, dataset.val_dataloader(), args.max_clips, MODES[mode])
        all_results[mode] = res
        print(f"[{mode}] ({n} clips) " + "  ".join(f"{k}={v:.4f}" for k, v in res.items()))

    keys = sorted({k for res in all_results.values() for k in res})
    header = f"{'metric':>16s} |" + "".join(f" {m:>11s}" for m in args.modes)
    print("\n" + header)
    print("-" * len(header))
    for k in keys:
        row = f"{k:>16s} |"
        for m in args.modes:
            v = all_results[m].get(k)
            row += f" {v:11.4f}" if v is not None else f" {'-':>11s}"
        print(row)
    if "as-trained" in all_results:
        base = all_results["as-trained"]
        print("\ndeltas vs as-trained:")
        for m in args.modes:
            if m == "as-trained":
                continue
            deltas = {
                k: all_results[m][k] - base[k]
                for k in all_results[m]
                if k in base and not k.startswith("gate_")
            }
            print(f"  {m}: " + "  ".join(f"{k}={d:+.4f}" for k, d in deltas.items()))


if __name__ == "__main__":
    main()
