"""Does the latent predictor predict motion? Diagnostics on a trained checkpoint.

The predictor is residual, Pred(x) = x + Delta(x), and nothing in the loss tree reads
its output; it is only consumed as the next frame's prior. This script measures whether
it has learned any dynamics, using eval-only forward passes.

With x_t the per-frame posterior slot state (processor.state), define

    Delta_t = Pred(x_t) - x_t          the predictor's residual step
    d_t     = x_{t+1} - x_t            the displacement it should have produced
    v_t     = x_t - x_{t-1}            the previous displacement (candidate extra input)

  M1  cos(Delta_t, d_t)           is the step aimed in the right direction
  M2  ||Delta_t|| / ||d_t||       does the predictor move at all, or is it identity-like
  M3  constant-velocity oracle    does the trivial Delta = v_t beat the learned Delta
  M4  linear probe                how much of d_t is linearly readable from v_t on top
                                  of what x_t already provides
  M5  (--metrics) replace the predictor with the identity and re-run the val metrics

M4 measures *linearly accessible* incremental information, which is neither an upper nor
a lower bound on what a nonlinear predictor could extract. Read it together with M3: the
constant-velocity oracle is itself linear in v, so agreement between the two is evidence
the signal is real rather than an artifact of the probe class.

Diagnostics run with cycle=False so the temporal order is causal. The optional metric
pass uses the checkpoint's own cyclic_inference setting so the numbers are comparable to
the logged validation metrics.

Usage (inside the container):
  python event_analysis/predictor_probe.py <settings.yaml> <checkpoint> [--clips N] [--metrics]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn

from slotcurri import configuration, data, metrics as metric_lib, models

EPS = 1e-12


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("settings")
    ap.add_argument("checkpoint")
    ap.add_argument("--clips", type=int, default=200, help="clips for the diagnostics pass")
    ap.add_argument("--metrics", action="store_true", help="also run the identity ablation (M5)")
    ap.add_argument("--metric-clips", type=int, default=None, help="cap clips in the M5 pass")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load_model(settings: str, ckpt: str, device):
    config = configuration.load_config(settings)
    config.model.visualize = False
    model = models.build(config.model, config.optimizer)
    model.load_weights_from_checkpoint(ckpt)
    model.to(device).eval()
    return config, model


def build_val_metrics(device):
    kw = dict(ignore_background=False, pred_key="decoder_masks_hard", true_key="segmentations")
    return {
        "ari": metric_lib.VideoARI(**kw).to(device),
        "image_ari": metric_lib.ImageARI(video_input=True, **kw).to(device),
        "mbo": metric_lib.VideoIoU(matching="overlap", **kw).to(device),
        "image_mbo": metric_lib.ImageIoU(matching="overlap", video_input=True, **kw).to(device),
    }


def to_device(batch, model, device):
    if "batch_padding_mask" in batch:
        batch = model._remove_padding(batch, batch["batch_padding_mask"])
        if batch is None:
            return None
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def collect(model, loader, n_clips, device):
    """Gather (x_t, v_t, d_t, Delta_t) over the common valid range t = 1 .. T-2.

    `clip` tags every row with its source clip so the probe in M4 can split by clip.
    A random split over rows would put frame t in train and frame t+1 in test, and since
    consecutive slot states are nearly identical that leaks the target.
    """
    ln = model.processor.module.corrector.norm_slots
    bufs = {k: [] for k in ("x", "v", "d", "delta", "xn", "vn", "dn", "deltan",
                            "xc", "vc", "dc", "deltac", "clip")}
    n_clips_used, n_frames = 0, 0
    for batch in loader:
        if n_clips_used >= n_clips:
            break
        if batch is None:
            continue
        batch = to_device(batch, model, device)
        if batch is None:
            continue
        out = model.forward(batch, train=False, cycle=False)
        xt = out["processor"]["state"]
        xpt = out["processor"]["state_predicted"]
        x = xt.double().cpu().numpy()
        xp = xpt.double().cpu().numpy()
        # The corrector LayerNorms the incoming prior before using it for queries and as
        # the GRU hidden state, so the scale of the raw prior is discarded downstream.
        # Repeat every quantity in that normalized space, which is what the model reads.
        xn = ln(xt).double().cpu().numpy()
        xpn = ln(xpt).double().cpu().numpy()
        # Queries are q_s = W_q LN(x_s) and the attention softmax runs over the slot axis,
        # so any component shared across slots cancels exactly. W_q is linear, hence the
        # attention-relevant signal is LN(x) centered across slots within each frame.
        xc = xn - xn.mean(axis=2, keepdims=True)
        xpc = xpn - xpn.mean(axis=2, keepdims=True)
        T = x.shape[1]
        if T < 3:
            continue
        pieces = {
            "x": x[:, 1 : T - 1],
            "v": x[:, 1 : T - 1] - x[:, 0 : T - 2],
            "d": x[:, 2:T] - x[:, 1 : T - 1],
            "delta": (xp - x)[:, 1 : T - 1],
            "xn": xn[:, 1 : T - 1],
            "vn": xn[:, 1 : T - 1] - xn[:, 0 : T - 2],
            "dn": xn[:, 2:T] - xn[:, 1 : T - 1],
            "deltan": (xpn - xn)[:, 1 : T - 1],
            "xc": xc[:, 1 : T - 1],
            "vc": xc[:, 1 : T - 1] - xc[:, 0 : T - 2],
            "dc": xc[:, 2:T] - xc[:, 1 : T - 1],
            "deltac": (xpc - xc)[:, 1 : T - 1],
        }
        for key, arr in pieces.items():
            bufs[key].append(arr.reshape(-1, arr.shape[-1]))
        n_rows = pieces["x"][..., 0].size
        bufs["clip"].append(np.full(n_rows, n_clips_used))
        n_clips_used += 1
        n_frames += x.shape[0] * (T - 2)
    if n_clips_used == 0:
        raise RuntimeError("no usable clips")
    out = {k: np.concatenate(v) for k, v in bufs.items()}
    print(f"clips={n_clips_used}  frame-steps={n_frames}  slot-step rows={len(out['x'])}")
    return out


def cosine(a, b):
    num = (a * b).sum(-1)
    den = np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), EPS)
    return num / den


def pct_row(name, arr):
    qs = [np.percentile(arr, q) for q in (5, 25, 50, 75, 95)]
    print(f"  {name:<28s} mean {arr.mean():+.4f} | p5 {qs[0]:+.4f}  p25 {qs[1]:+.4f} "
          f" p50 {qs[2]:+.4f}  p75 {qs[3]:+.4f}  p95 {qs[4]:+.4f}")


LAMS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def _ridge(Xtr, Ytr, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Z = np.concatenate([(Xtr - mu) / sd, np.ones((len(Xtr), 1))], 1)
    G = Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1])
    return np.linalg.solve(G, Z.T @ Ytr), mu, sd


def _apply(W, mu, sd, X):
    return np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1) @ W


def ridge_probe(X, Y, tr, va, te):
    """Fit on tr, pick the ridge strength on va, report the residual on te."""
    best = None
    for lam in LAMS:
        W, mu, sd = _ridge(X[tr], Y[tr], lam)
        r_va = float(((Y[va] - _apply(W, mu, sd, X[va])) ** 2).sum(1).mean())
        if best is None or r_va < best[0]:
            best = (r_va, lam, W, mu, sd)
    _, lam, W, mu, sd = best
    r_te = float(((Y[te] - _apply(W, mu, sd, X[te])) ** 2).sum(1).mean())
    return r_te, lam


def report(data_, seed):
    x, v, d, delta = data_["x"], data_["v"], data_["d"], data_["delta"]
    nd = np.linalg.norm(d, axis=-1)
    ndl = np.linalg.norm(delta, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    msd = (d ** 2).sum(-1).mean()  # mean ||d||^2, the identity predictor's error

    print("\n=== scale of the quantities (L2 norm per slot-step) ===")
    print(f"  ||x||     mean {np.linalg.norm(x, axis=-1).mean():.4f}")
    print(f"  ||d||     mean {nd.mean():.4f}   (true displacement)")
    print(f"  ||v||     mean {nv.mean():.4f}   (previous displacement)")
    print(f"  ||Delta|| mean {ndl.mean():.4f}   (predictor step)")

    print("\n=== M1  direction: cos(Delta_t, d_t) ===")
    c_learned = cosine(delta, d)
    pct_row("cos(Delta, d)", c_learned)
    c_cv = cosine(v, d)
    pct_row("cos(v, d)  [const-velocity]", c_cv)
    hi = nd >= np.median(nd)
    print(f"  restricted to ||d|| >= median ({np.median(nd):.4f}):")
    pct_row("cos(Delta, d) | moving", c_learned[hi])
    pct_row("cos(v, d)     | moving", c_cv[hi])

    print("\n=== M2  magnitude: ||Delta|| / ||d|| ===")
    ratio = ndl / np.maximum(nd, EPS)
    pct_row("||Delta||/||d||", ratio)
    print(f"  ||Delta||/||x|| mean {(ndl / np.maximum(np.linalg.norm(x, axis=-1), EPS)).mean():.4f}")

    print("\n=== M2b  is Delta a near-constant offset rather than a state-dependent step? ===")
    dbar = delta.mean(0)
    resid = delta - dbar
    print(f"  ||mean_t Delta||            {np.linalg.norm(dbar):.4f}")
    print(f"  mean ||Delta - mean Delta|| {np.linalg.norm(resid, axis=-1).mean():.4f}")
    print(f"  fraction of Delta energy in the constant component "
          f"{np.linalg.norm(dbar) ** 2 / (delta ** 2).sum(1).mean():.4f}")
    pct_row("cos(Delta_t, mean Delta)", cosine(delta, np.broadcast_to(dbar, delta.shape)))

    print("\n=== M3  one-step displacement error, normalized by the identity predictor ===")
    print(f"  (1.000 = Delta:=0, i.e. Pred = identity;  mean ||d||^2 = {msd:.6f})")

    def nmse(pred):
        return float(((pred - d) ** 2).sum(-1).mean() / msd)

    a_cv = float((v * d).sum() / max((v * v).sum(), EPS))
    a_ld = float((delta * d).sum() / max((delta * delta).sum(), EPS))
    rows = [
        ("identity  (Delta = 0)", 1.0),
        ("learned   (Delta = Pred(x)-x)", nmse(delta)),
        (f"learned  x{a_ld:.3f} global rescale", nmse(a_ld * delta)),
        ("const-vel (Delta = v)", nmse(v)),
        (f"const-vel x{a_cv:.3f} global rescale", nmse(a_cv * v)),
    ]
    for name, val in rows:
        print(f"  {name:<36s} {val:.4f}")
    ceil_learned = float((nd ** 2 * (1 - c_learned ** 2)).mean() / msd)
    ceil_cv = float((nd ** 2 * (1 - c_cv ** 2)).mean() / msd)
    print("  -- with per-sample optimal rescaling (direction-only ceiling) --")
    print(f"  {'learned direction':<36s} {ceil_learned:.4f}")
    print(f"  {'const-vel direction':<36s} {ceil_cv:.4f}")

    print("\n=== M4  linear probe for d_t (clip-level 60/20/20 split, ridge chosen on val) ===")
    clip = data_["clip"]
    uniq = np.unique(clip)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n1, n2 = int(0.6 * len(uniq)), int(0.8 * len(uniq))
    sets = [np.isin(clip, uniq[:n1]), np.isin(clip, uniq[n1:n2]), np.isin(clip, uniq[n2:])]
    tr, va, te = (np.flatnonzero(s) for s in sets)
    print(f"  clips  train {n1}  val {n2 - n1}  test {len(uniq) - n2}"
          f"   rows  {len(tr)} / {len(va)} / {len(te)}")

    r0 = float(((d[te] - d[tr].mean(0)) ** 2).sum(1).mean())
    xv = np.concatenate([x, v], 1)
    r_x, lam_x = ridge_probe(x, d, tr, va, te)
    r_v, lam_v = ridge_probe(v, d, tr, va, te)
    r_xv, lam_xv = ridge_probe(xv, d, tr, va, te)
    r_pred = float(((delta[te] - d[te]) ** 2).sum(1).mean())
    print(f"  {'predict the train mean':<26s} {r0 / r0:.4f}")
    print(f"  {'from x only':<26s} {r_x / r0:.4f}   (lam={lam_x:g})")
    print(f"  {'from v only':<26s} {r_v / r0:.4f}   (lam={lam_v:g})")
    print(f"  {'from (x, v)':<26s} {r_xv / r0:.4f}   (lam={lam_xv:g})")
    print(f"  {'the trained predictor':<26s} {r_pred / r0:.4f}   <- for reference")
    inc = (r_x - r_xv) / max(r_x, EPS)
    print(f"\n  incremental gain from v given x: (R_x - R_xv)/R_x = {100 * inc:.2f}%")
    return {"cos_learned": float(c_learned.mean()), "cos_cv": float(c_cv.mean()),
            "ratio_med": float(np.median(ratio)), "inc_v": inc,
            "cv_ceiling": ceil_cv, "cv_scaled": nmse(a_cv * v)}


@torch.no_grad()
def eval_metrics(model, dataset, device, cycle, tag, limit=None):
    ms = build_val_metrics(device)
    n = 0
    for batch in dataset.val_dataloader():
        if limit is not None and n >= limit:
            break
        if batch is None:
            continue
        batch = to_device(batch, model, device)
        if batch is None:
            continue
        out = model.forward(batch, train=False, cycle=cycle)
        aux = model.aux_forward(batch, out)
        for m in ms.values():
            m.update(**batch, **out, **aux)
        n += 1
    res = {k: float(m.compute()) for k, m in ms.items()}
    print(f"  {tag:<22s} " + "  ".join(f"{k}={v:.4f}" for k, v in res.items()) + f"   (clips={n})")
    return res


def main():
    args = parse_args()
    device = torch.device("cuda")
    config, model = load_model(args.settings, args.checkpoint, device)
    dataset = data.build(config.dataset)
    dataset.setup("validate")

    pred = model.processor.module.predictor
    n_par = sum(p.numel() for p in pred.parameters())
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"checkpoint: {args.checkpoint}")
    print(f"attn_mass_enabled: {model.attn_mass_enabled}   cyclic_inference: {model.cyclic_inference}")
    print(f"predictor: {type(pred).__name__}  params={n_par:,} ({100 * n_par / n_tot:.2f}% of {n_tot:,})")

    print("\n" + "=" * 78)
    print("diagnostics pass (cycle=False)")
    print("=" * 78)
    data_ = collect(model, dataset.val_dataloader(), args.clips, device)

    print("\n" + "#" * 78)
    print("# RAW state space (note: the corrector LayerNorms the prior, so absolute")
    print("# magnitudes here are discarded downstream and only directions carry over)")
    print("#" * 78)
    report(data_, args.seed)

    print("\n" + "#" * 78)
    print("# POST-LAYERNORM space -- what the corrector's GRU hidden state receives")
    print("#" * 78)
    report({"x": data_["xn"], "v": data_["vn"], "d": data_["dn"],
            "delta": data_["deltan"], "clip": data_["clip"]}, args.seed)

    print("\n" + "#" * 78)
    print("# SLOT-CENTERED POST-LAYERNORM space -- what the attention softmax can see")
    print("#" * 78)
    summary = report({"x": data_["xc"], "v": data_["vc"], "d": data_["dc"],
                      "delta": data_["deltac"], "clip": data_["clip"]}, args.seed)

    if args.metrics:
        print("\n" + "=" * 78)
        print(f"M5  identity ablation (cycle={model.cyclic_inference}, full val split)")
        print("=" * 78)
        lim = args.metric_clips
        eval_metrics(model, dataset, device, model.cyclic_inference, "trained predictor", lim)
        model.processor.module.predictor = nn.Identity()
        eval_metrics(model, dataset, device, model.cyclic_inference, "identity predictor", lim)
        model.processor.module.predictor = pred

    print("\n" + "=" * 78)
    print("verdict")
    print("=" * 78)
    print(f"  cos(Delta, d)          {summary['cos_learned']:+.4f}   "
          f"(>= +0.40 would mean the predictor already tracks motion)")
    print(f"  cos(v, d)              {summary['cos_cv']:+.4f}   "
          f"(direction a trivial const-velocity step already recovers)")
    print(f"  median ||Delta||/||d|| {summary['ratio_med']:.2f}   "
          f"(1 = right scale, << 1 identity-like, >> 1 off-manifold kick)")
    print(f"  const-vel one-step error, best global scale   {summary['cv_scaled']:.4f}")
    print(f"  const-vel one-step error, direction-only best {summary['cv_ceiling']:.4f}   "
          f"(1.0 = no better than Delta:=0)")
    print(f"  incremental linear information in v given x = {100 * summary['inc_v']:.2f}%   "
          f"(< 5% would make velocity conditioning not worth it)")


if __name__ == "__main__":
    main()
