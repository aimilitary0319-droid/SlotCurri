"""Do the v10-era catastrophic clips still fail under v20/v21 (trained method line)?

The vis_v10_loses_to_baseline ranking is from v10. Before designing a fix for the
background-carving failure (rank00 = sample 100, v10 ARI 0.001), check what the
finished v20/v21 checkpoints do on those same clips, next to the baseline.

Note: fg-ARI (ignore_background=True) is not computable on YTVIS GT here -- the
one-hot GT has unlabeled pixels, which the metric rejects -- so we report the
standard ARI/mBO convention used by the project (background counted).

Usage (inside the slotcurri container):
  python event_analysis/check_v25_on_losers.py --data-dir /workspace/dataset
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from slotcurri import configuration, data, metrics as metric_lib, models
from slotcurri.data.transforms import Denormalize
from slotcurri.metrics import adjusted_rand_index
from slotcurri.visualizations import mix_videos_with_masks


@torch.no_grad()
def fg_ari(gt_onehot: torch.Tensor, pred_hard: torch.Tensor) -> float:
    """ARI over labeled (foreground) pixels only.

    The stock metric cannot ignore background here because YTVIS GT encodes
    background as all-zero (no explicit channel), so we select the points with
    exactly one GT label and run the same SAVi ARI on them.

    gt_onehot: (T, C, H, W) bool; pred_hard: (T, S, H, W) bool, same H/W.
    """
    t, c, h, w = gt_onehot.shape
    true = gt_onehot.permute(0, 2, 3, 1).reshape(-1, c).float()  # (P, C)
    pred = pred_hard.permute(0, 2, 3, 1).reshape(-1, pred_hard.shape[1]).float()  # (P, S)
    labeled = true.sum(-1) == 1
    if labeled.sum() < 2:
        return float("nan")
    val = adjusted_rand_index(true[labeled][None], pred[labeled][None])
    return float(val[0])

MODEL_DIRS = {
    "baseline": "logs/_ytvis",
    "v20": "logs/_ytvis_attnmass_v20",
    "v21": "logs/_ytvis_attnmass_v21",
}


def load_model(root: Path, log_dir: str, device: torch.device):
    settings = root / log_dir / "settings/slotcurri/settings.yaml"
    ckpt = root / log_dir / "checkpoints/slotcurri_step=step=100000-v1.ckpt"
    config = configuration.load_config(str(settings))
    config.model.visualize = False
    model = models.build(config.model, config.optimizer)
    model.load_weights_from_checkpoint(str(ckpt))
    model.to(device).eval()
    return model


def build_val_metrics():
    return {
        "ari": metric_lib.VideoARI(
            ignore_background=False, pred_key="decoder_masks_hard", true_key="segmentations"
        ),
        "mbo": metric_lib.VideoIoU(
            matching="overlap",
            ignore_background=False,
            pred_key="decoder_masks_hard",
            true_key="segmentations",
        ),
    }


@torch.no_grad()
def score_and_masks(model, batch, device):
    batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    outputs = model.forward(batch_dev, train=False, cycle=model.cyclic_inference)
    aux = model.aux_forward(batch_dev, outputs)
    scores = {}
    for name, metric in build_val_metrics().items():
        metric = metric.to(device)
        metric.reset()
        metric.update(**batch_dev, **outputs, **aux)
        val = metric.compute()
        scores[name] = float(val.detach().cpu()) if torch.is_tensor(val) else float(val)
        metric.reset()
    key = "decoder_masks_vis_hard" if "decoder_masks_vis_hard" in aux else "decoder_masks_hard"
    return scores, aux[key].cpu()


def overlay(video, masks, spatial):
    if masks.shape[-2:] != spatial:
        b, t, s, h, w = masks.shape
        m = torch.nn.functional.interpolate(
            masks.float().reshape(b * t, s, h, w), size=spatial, mode="nearest"
        )
        masks = m.reshape(b, t, s, *spatial)
    mixed = mix_videos_with_masks(video, masks.float(), alpha=0.45)
    return mixed[0].permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/dataset")
    ap.add_argument("--samples", default="38,100,101,129,209")
    ap.add_argument("--out-dir", default="logs/split_score_probe")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    root = Path("/workspace/SlotCurri")
    if not (root / "slotcurri").exists():
        root = Path("/mnt/ssd2/hmlee/SlotCurri")
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = sorted(int(x) for x in args.samples.split(","))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    nets = {name: load_model(root, d, device) for name, d in MODEL_DIRS.items()}

    cfg = configuration.load_config(str(root / "logs/_ytvis/settings/slotcurri/settings.yaml"))
    cfg.dataset.num_val_workers = 0
    cfg.dataset.val_batch_size = 1
    dm = data.build(cfg.dataset, data_dir=args.data_dir)
    dm.setup("validate")
    loader = dm.val_dataloader()

    names = list(nets.keys())
    hdr = (
        f"{'sample':>6s} |"
        + "".join(f" {n + ' ARI':>9s}" for n in names)
        + " |"
        + "".join(f" {n + ' fgARI':>11s}" for n in names)
        + " |"
        + "".join(f" {n + ' mBO':>9s}" for n in names)
    )
    print(hdr)
    print("-" * len(hdr))
    denorm = Denormalize(input_type="video")
    for si, batch in enumerate(loader):
        if si > max(wanted):
            break
        if si not in wanted:
            continue
        if "batch_padding_mask" in batch:
            batch = nets["baseline"]._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                continue
        scores, masks = {}, {}
        gt = batch["segmentations"][0].bool()  # (T, C, H, W)
        for n in names:
            scores[n], masks[n] = score_and_masks(nets[n], batch, device)
            pred = masks[n][0].bool()  # (T, S, h, w)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred.float(), size=gt.shape[-2:], mode="nearest"
                ).bool()
            scores[n]["fg_ari"] = fg_ari(gt, pred)
            # per-GT-object ownership: which slot claims each object's pixels?
            pred_id = pred.float().argmax(dim=1)  # (T, H, W)
            own = []
            for c in range(gt.shape[1]):
                obj = gt[:, c]
                npx = int(obj.sum())
                if npx == 0:
                    continue
                counts = torch.bincount(pred_id[obj], minlength=pred.shape[1])
                top = int(counts.argmax())
                own.append(f"obj{c}: s{top} {float(counts[top]) / npx:.0%} ({npx}px)")
            scores[n]["own"] = "; ".join(own)
        print(
            f"{si:>6d} |"
            + "".join(f" {scores[n]['ari']:>9.3f}" for n in names)
            + " |"
            + "".join(f" {scores[n]['fg_ari']:>11.3f}" for n in names)
            + " |"
            + "".join(f" {scores[n]['mbo']:>9.3f}" for n in names)
        )
        for n in names:
            print(f"        {n:>8s}: {scores[n]['own']}")

        video = denorm(batch["video"][0].cpu()).clamp(0, 1).unsqueeze(0)
        spatial = video.shape[-2:]
        t_mid = video.shape[1] // 2
        panels = [(video[0, t_mid].permute(1, 2, 0).numpy() * 255).astype(np.uint8)]
        labels = ["rgb"]
        for n in names:
            panels.append(overlay(video, masks[n], spatial)[t_mid])
            labels.append(
                f"{n} ARI={scores[n]['ari']:.3f} fgARI={scores[n]['fg_ari']:.3f} "
                f"mBO={scores[n]['mbo']:.3f}"
            )
        parts = []
        for img, lab in zip(panels, labels):
            im = Image.fromarray(img)
            bar = Image.new("RGB", (im.width, 36), (20, 20, 20))
            ImageDraw.Draw(bar).text((8, 10), lab, fill=(240, 240, 240))
            canvas = Image.new("RGB", (im.width, im.height + 36))
            canvas.paste(bar, (0, 0))
            canvas.paste(im, (0, 36))
            parts.append(np.asarray(canvas))
        Image.fromarray(np.concatenate(parts, axis=1)).save(
            out_dir / f"sample{si:03d}_base_v20_v21.png"
        )

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
