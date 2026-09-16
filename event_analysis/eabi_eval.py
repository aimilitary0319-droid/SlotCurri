"""Evidence-Anchored Bidirectional Inference (EABI) evaluation.

Test-time-only protocol comparison on a trained checkpoint. After a standard forward
sweep, EABI picks the frame where the gate's own evidence E_t = sum_s g_{t,s} m_{t,s}
is maximal (the decomposition the model itself trusts most), re-runs the recurrent
sweep backward from that anchor, and keeps the forward outputs from the anchor onward
(re-running forward from the anchor state would reproduce them exactly). The legacy
cyclic inference (backward sweep anchored at the LAST frame) is the special case
anchor = T-1, so it is the natural ablation baseline for anchor *selection*.

Six protocols, one row each:
  single     cycle=False            standard forward pass (v26's reported setting)
  last       cycle=True             legacy cyclic inference (fixed anchor = last frame)
  evidence   cycle="evidence"       EABI, anchor = argmax_t sum_s g*c (purity-weighted
                                    soft object count: "most objects intactly bound")
  ev_mass    cycle="evidence_mass"  EABI, anchor = argmax_t sum_s g*m (trusted mass
                                    share) -- A/B variant of the anchor statistic
  ev_sum     cycle="evidence_sum"   EABI, anchor = argmax_t sum_s g (gate occupancy
                                    only; no extra c / m). Default for v39lam1gu.
  random     cycle="random"         EABI with a uniform random anchor -- controls
                                    whether evidence-based selection (not just
                                    re-sweeping from *somewhere*) is what helps

Decision rule agreed beforehand: adopt if evidence > last >= single with the
evidence-last gap >= +0.005 video ARI; demote to a cost argument if evidence ~= last;
drop if evidence <= single.

Usage (inside the slotcurri container, repo root as cwd):
  python event_analysis/eabi_eval.py \
      --ckpt "logs/_ytvis_attnmass_v26/checkpoints/slotcurri_step=step=100000-v1.ckpt" \
      --data-dir /workspace/dataset --max-clips 200
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from slotcurri import configuration, data, metrics, models

PROTOCOLS = {
    "single": False,
    "last": True,
    "evidence": "evidence",
    "ev_mass": "evidence_mass",
    "ev_sum": "evidence_sum",
    "random": "random",
}


@torch.no_grad()
def run(model, loader, protocols, max_clips, device, config):
    agg = {
        proto: {name: metrics.build(c).to(device) for name, c in config.val_metrics.items()}
        for proto in protocols
    }
    # fresh instance per (clip, protocol) for paired per-clip diagnostics
    clip_ari = metrics.build(config.val_metrics["ari"]).to(device)

    records = {proto: [] for proto in protocols}  # per-clip video ARI
    anchors = {proto: [] for proto in protocols if isinstance(PROTOCOLS[proto], str)}
    seq_lens = []
    n_clips = 0
    for batch in loader:
        if "batch_padding_mask" in batch:
            batch = model._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                continue
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        seq_lens.append(int(batch[model.input_key].shape[1]))

        for proto in protocols:
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model.forward(batch, train=False, cycle=PROTOCOLS[proto])
                aux_outputs = model.aux_forward(batch, outputs)
            for metric in agg[proto].values():
                metric.update(**batch, **outputs, **aux_outputs)
            clip_ari.update(**batch, **outputs, **aux_outputs)
            records[proto].append(float(clip_ari.compute()))
            clip_ari.reset()
            if proto in anchors:
                af = model.processor.last_anchor_frames
                anchors[proto].append(int(af[0].item()) if af is not None else -1)

        n_clips += 1
        if n_clips % 20 == 0:
            print(f"  {n_clips} clips ...", flush=True)
        if n_clips >= max_clips:
            break

    results = {
        proto: {name: float(metric.compute()) for name, metric in agg[proto].items()}
        for proto in protocols
    }
    return results, records, anchors, np.asarray(seq_lens), n_clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/slotcurri/ytvis2021_attnmass_v26.yaml")
    ap.add_argument(
        "--ckpt",
        default="logs/_ytvis_attnmass_v26/checkpoints/slotcurri_step=step=100000-v1.ckpt",
    )
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--max-clips", type=int, default=200)
    ap.add_argument(
        "--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS)
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="event_analysis/eabi_eval_v26")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    config = configuration.load_config(args.config)
    dataset = data.build(config.dataset, data_dir=args.data_dir)
    model = models.build(config.model, config.optimizer, None, None)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    dataset.setup("validate")
    loader = dataset.val_dataloader()

    results, records, anchors, seq_lens, n_clips = run(
        model, loader, args.protocols, args.max_clips, device, config
    )

    # --- aggregate table ---
    names = list(config.val_metrics.keys())
    print(f"\n=== EABI eval: {args.ckpt}  ({n_clips} clips) ===")
    header = f"{'protocol':>10s} | " + " | ".join(f"{n:>10s}" for n in names)
    print(header)
    print("-" * len(header))
    for proto in args.protocols:
        row = f"{proto:>10s} | " + " | ".join(f"{results[proto][n]:10.4f}" for n in names)
        print(row)

    # --- paired per-clip comparison (video ARI) ---
    per_clip = {p: np.asarray(records[p]) for p in args.protocols}
    if "evidence" in per_clip:
        for base in ("single", "last", "ev_mass", "ev_sum", "random"):
            if base not in per_clip:
                continue
            d = per_clip["evidence"] - per_clip[base]
            print(
                f"\nevidence vs {base:>8s}: mean {d.mean():+.4f}  median {np.median(d):+.4f}"
                f"  improved {(d > 0).mean():.1%}  worsened {(d < 0).mean():.1%}"
            )
    if "ev_sum" in per_clip:
        for base in ("single", "last"):
            if base not in per_clip:
                continue
            d = per_clip["ev_sum"] - per_clip[base]
            print(
                f"\nev_sum vs {base:>8s}: mean {d.mean():+.4f}  median {np.median(d):+.4f}"
                f"  improved {(d > 0).mean():.1%}  worsened {(d < 0).mean():.1%}"
            )

    # --- anchor statistics ---
    anchor_fracs = {}
    for proto in ("evidence", "ev_mass", "ev_sum"):
        if proto in anchors and len(anchors[proto]):
            a = np.asarray(anchors[proto], dtype=np.float64)
            t = seq_lens[: len(a)].astype(np.float64)
            anchor_fracs[proto] = a / np.maximum(t - 1, 1)
            print(
                f"\n{proto} anchors: at frame 0 (no-op) {(a == 0).mean():.1%}"
                f"  in last 10% of clip {(anchor_fracs[proto] >= 0.9).mean():.1%}"
                f"  mean position {anchor_fracs[proto].mean():.2f}"
            )
    if "evidence" in anchor_fracs and "ev_mass" in anchor_fracs:
        a1 = np.asarray(anchors["evidence"], dtype=np.float64)
        a2 = np.asarray(anchors["ev_mass"], dtype=np.float64)
        n = min(len(a1), len(a2))
        print(
            f"anchor agreement (count vs mass): same frame {(a1[:n] == a2[:n]).mean():.1%}"
            f"  mean |diff| {np.abs(a1[:n] - a2[:n]).mean():.1f} frames"
        )
    if "evidence" in anchor_fracs and "ev_sum" in anchor_fracs:
        a1 = np.asarray(anchors["evidence"], dtype=np.float64)
        a2 = np.asarray(anchors["ev_sum"], dtype=np.float64)
        n = min(len(a1), len(a2))
        print(
            f"anchor agreement (g*c vs sum g): same frame {(a1[:n] == a2[:n]).mean():.1%}"
            f"  mean |diff| {np.abs(a1[:n] - a2[:n]).mean():.1f} frames"
        )
    anchor_frac = anchor_fracs.get("ev_sum") or anchor_fracs.get("evidence")

    # --- save ---
    np.savez_compressed(
        f"{args.out}.npz",
        seq_lens=seq_lens,
        **{f"ari_{p}": per_clip[p] for p in per_clip},
        **{f"anchor_{p}": np.asarray(v) for p, v in anchors.items()},
        **{f"agg_{p}_{n}": results[p][n] for p in args.protocols for n in names},
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axes[0]
    vals = [results[p]["ari"] for p in args.protocols]
    ax.bar(
        args.protocols,
        vals,
        color=["gray", "tab:orange", "tab:green", "tab:blue", "tab:purple", "tab:red"][
            : len(vals)
        ],
    )
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("video ARI")
    ax.set_title(f"aggregate over {n_clips} clips")

    ax = axes[1]
    if anchor_fracs:
        for proto, color in (
            ("evidence", "tab:green"),
            ("ev_mass", "tab:blue"),
            ("ev_sum", "tab:purple"),
        ):
            if proto in anchor_fracs:
                ax.hist(
                    anchor_fracs[proto], bins=20, range=(0, 1), color=color,
                    alpha=0.55, label=proto,
                )
        ax.set_xlabel("anchor position (fraction of clip)")
        ax.set_title("where does evidence peak?")
        ax.legend()

    ax = axes[2]
    if anchor_frac is not None and "single" in per_clip:
        if "ev_sum" in per_clip:
            d = per_clip["ev_sum"] - per_clip["single"]
            label = "ev_sum - single"
        elif "evidence" in per_clip:
            d = per_clip["evidence"] - per_clip["single"]
            label = "evidence - single"
        else:
            d = None
            label = ""
        if d is not None:
            ax.scatter(anchor_frac, d[: len(anchor_frac)], s=12, alpha=0.6, color="tab:purple")
            ax.axhline(0.0, color="k", lw=1, ls="--")
            ax.set_xlabel("anchor position (fraction of clip)")
            ax.set_ylabel(f"ARI delta ({label})")
            ax.set_title("per-clip gain vs anchor position")

    fig.suptitle(os.path.basename(args.ckpt))
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=130)
    print(f"\nsaved: {args.out}.npz  {args.out}.png")


if __name__ == "__main__":
    main()
