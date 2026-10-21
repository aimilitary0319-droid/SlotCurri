"""YTVIS 2021 validation: GT appear-then-disappear events, stats, and a high-frequency subset.

An appear-then-disappear cycle is one contiguous visible run of a GT instance that
starts after frame 0 and ends before the last frame. Appearance / disappearance
locations are the mask centroid on the first and last visible frame of that run.
"""

import argparse
import csv
import io
import json
import os
import pickle
import tarfile
from collections import Counter, defaultdict

import numpy as np

YTVIS2021_CATEGORIES = {
    1: "airplane",
    2: "bear",
    3: "bird",
    4: "boat",
    5: "car",
    6: "cat",
    7: "cow",
    8: "deer",
    9: "dog",
    10: "duck",
    11: "earless_seal",
    12: "elephant",
    13: "fish",
    14: "flying_disc",
    15: "fox",
    16: "frog",
    17: "giant_panda",
    18: "giraffe",
    19: "horse",
    20: "leopard",
    21: "lizard",
    22: "monkey",
    23: "motorbike",
    24: "mouse",
    25: "parrot",
    26: "person",
    27: "rabbit",
    28: "shark",
    29: "skateboard",
    30: "snake",
    31: "snowboard",
    32: "squirrel",
    33: "surfboard",
    34: "tennis_racket",
    35: "tiger",
    36: "train",
    37: "truck",
    38: "turtle",
    39: "whale",
    40: "zebra",
}

BIN3_NAMES = (
    "top_left",
    "top_center",
    "top_right",
    "mid_left",
    "center",
    "mid_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
)


def bin3(v):
    if v < 1.0 / 3.0:
        return 0
    if v < 2.0 / 3.0:
        return 1
    return 2


def bin3_name(cx, cy):
    return BIN3_NAMES[bin3(cy) * 3 + bin3(cx)]


def border_labels(mask, margin=2):
    labels = []
    if mask[:, :margin].any():
        labels.append("left")
    if mask[:, -margin:].any():
        labels.append("right")
    if mask[:margin, :].any():
        labels.append("top")
    if mask[-margin:, :].any():
        labels.append("bottom")
    return labels


def centroid(mask):
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return float(xs.mean()) / float(w), float(ys.mean()) / float(h)


def visible_runs(present):
    """present: (T,) bool. Returns inclusive (start, end) runs."""
    if not present.any():
        return []
    padded = np.concatenate([[False], present, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _centroid_border_gpu(mask):
    """mask: (H, W) bool CUDA tensor. Centroid is normalized by width/height."""
    import torch

    weight = mask.float()
    mass = weight.sum().clamp_min(1)
    h, w = mask.shape
    xs = torch.arange(w, device=mask.device, dtype=torch.float32)
    ys = torch.arange(h, device=mask.device, dtype=torch.float32)
    cx = float((weight.sum(dim=0) * xs).sum() / mass / w)
    cy = float((weight.sum(dim=1) * ys).sum() / mass / h)
    labels = []
    if bool(mask[:, :2].any()):
        labels.append("left")
    if bool(mask[:, -2:].any()):
        labels.append("right")
    if bool(mask[:2, :].any()):
        labels.append("top")
    if bool(mask[-2:, :].any()):
        labels.append("bottom")
    return cx, cy, labels


def analyze_video(video_id, seg, src_tar, min_area=1, min_duration=2, seg_t=None, area_t=None, cat_t=None):
    t, h, w, n = seg.shape
    videos_events = []
    n_entries = 0
    n_exits = 0
    n_persistent_runs = 0
    n_gaps = 0
    n_objects = 0
    cat_ids = []

    for obj_i in range(n):
        if area_t is None:
            channel = seg[..., obj_i]
            area = np.count_nonzero(channel.reshape(t, -1), axis=1)
            cat = int(channel.max()) if int(area.max()) > 0 else 0
        else:
            area = area_t[:, obj_i].detach().cpu().numpy()
            cat = int(cat_t[obj_i].item()) if int(area.max()) > 0 else 0
            channel = None
        if int(area.max()) <= 0:
            continue
        n_objects += 1
        cat_ids.append(cat)
        present = area >= min_area
        runs = visible_runs(present)
        n_gaps += max(0, len(runs) - 1)
        for start, end in runs:
            duration = end - start + 1
            appeared = start > 0
            disappeared = end < t - 1
            if appeared:
                n_entries += 1
            if disappeared:
                n_exits += 1
            if (not appeared) and (not disappeared):
                n_persistent_runs += 1
            if not (appeared and disappeared and duration >= min_duration):
                continue
            if seg_t is None:
                appear_mask = channel[start] > 0
                disappear_mask = channel[end] > 0
                ax, ay = centroid(appear_mask)
                dx, dy = centroid(disappear_mask)
                a_border = border_labels(appear_mask)
                d_border = border_labels(disappear_mask)
            else:
                ax, ay, a_border = _centroid_border_gpu(seg_t[start, :, :, obj_i] > 0)
                dx, dy, d_border = _centroid_border_gpu(seg_t[end, :, :, obj_i] > 0)
            run_area = area[start : end + 1]
            videos_events.append(
                {
                    "video_id": video_id,
                    "obj_index": obj_i,
                    "category_id": cat,
                    "category_name": YTVIS2021_CATEGORIES.get(cat, str(cat)),
                    "start": int(start),
                    "end": int(end),
                    "duration": int(duration),
                    "T": int(t),
                    "H": int(h),
                    "W": int(w),
                    "appear_frame_frac": float(start) / float(t),
                    "disappear_frame_frac": float(end) / float(t),
                    "appear_cx": ax,
                    "appear_cy": ay,
                    "disappear_cx": dx,
                    "disappear_cy": dy,
                    "appear_bin": bin3_name(ax, ay),
                    "disappear_bin": bin3_name(dx, dy),
                    "appear_border": "|".join(a_border) if a_border else "interior",
                    "disappear_border": "|".join(d_border) if d_border else "interior",
                    "appear_interior": int(len(a_border) == 0),
                    "disappear_interior": int(len(d_border) == 0),
                    "mean_area": float(run_area.mean()),
                    "max_area": int(run_area.max()),
                }
            )

    n_cycles = len(videos_events)
    n_border_appear = sum(1 for e in videos_events if not e["appear_interior"])
    n_interior_appear = n_cycles - n_border_appear
    n_border_disappear = sum(1 for e in videos_events if not e["disappear_interior"])
    n_interior_disappear = n_cycles - n_border_disappear
    video_row = {
        "video_id": video_id,
        "src_tar": os.path.basename(src_tar),
        "T": int(t),
        "H": int(h),
        "W": int(w),
        "n_objects": int(n_objects),
        "n_cycles": int(n_cycles),
        "n_entries": int(n_entries),
        "n_exits": int(n_exits),
        "n_persistent_runs": int(n_persistent_runs),
        "n_gaps": int(n_gaps),
        "cycle_per_frame": float(n_cycles) / float(t),
        "n_border_appear": int(n_border_appear),
        "n_interior_appear": int(n_interior_appear),
        "n_border_disappear": int(n_border_disappear),
        "n_interior_disappear": int(n_interior_disappear),
        "categories": ",".join(
            YTVIS2021_CATEGORIES.get(c, str(c)) for c in sorted(set(cat_ids))
        ),
    }
    return video_row, videos_events


def process_shard(path, min_area, min_duration, device="cuda:0"):
    import torch

    rows = []
    events = []
    dev = torch.device(device)
    with tarfile.open(path) as tar:
        for member in tar:
            if not member.name.endswith("segmentations.npy"):
                continue
            video_id = member.name.split(".", 1)[0]
            raw = tar.extractfile(member).read()
            seg = np.load(io.BytesIO(raw))
            seg_t = torch.from_numpy(np.ascontiguousarray(seg)).to(dev, non_blocking=True)
            present = seg_t > 0
            area_t = present.sum(dim=(1, 2))
            cat_t = seg_t.amax(dim=(0, 1, 2))
            row, ev = analyze_video(
                video_id,
                seg,
                path,
                min_area,
                min_duration,
                seg_t=seg_t,
                area_t=area_t,
                cat_t=cat_t,
            )
            rows.append(row)
            events.extend(ev)
            del seg_t, present, area_t, cat_t
    return rows, events


def _gpu_worker(device_index, paths, min_area, min_duration, out_file):
    import torch

    torch.cuda.set_device(device_index)
    name = torch.cuda.get_device_name(device_index)
    print(f"GPU cuda:{device_index} {name} shards={len(paths)}", flush=True)
    rows = []
    events = []
    for i, path in enumerate(paths):
        shard_rows, shard_events = process_shard(
            path, min_area, min_duration, device=f"cuda:{device_index}"
        )
        rows.extend(shard_rows)
        events.extend(shard_events)
        print(
            f"GPU cuda:{device_index} {i + 1}/{len(paths)} {os.path.basename(path)} "
            f"videos={len(shard_rows)} cycles={len(shard_events)}",
            flush=True,
        )
    with open(out_file, "wb") as handle:
        pickle.dump(
            {"rows": rows, "events": events, "gpu": name, "device_index": device_index},
            handle,
        )


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def count_named(values):
    counts = Counter(values)
    return {k: int(counts[k]) for k in sorted(counts)}


def grid_counts(events, key):
    counts = Counter(e[key] for e in events)
    return {name: int(counts.get(name, 0)) for name in BIN3_NAMES}


def write_shards(selected, src_root, out_dir, per_shard=10):
    os.makedirs(out_dir, exist_ok=True)
    by_tar = defaultdict(list)
    for row in selected:
        by_tar[row["src_tar"]].append(row["video_id"])
    payloads = {}
    for tar_name, video_ids in by_tar.items():
        wanted = set()
        for video_id in video_ids:
            wanted.add(f"{video_id}.video.npy")
            wanted.add(f"{video_id}.segmentations.npy")
        with tarfile.open(os.path.join(src_root, tar_name)) as tar:
            for member in tar:
                if member.name in wanted:
                    payloads[member.name] = tar.extractfile(member).read()
    shard_paths = []
    for shard_i, start in enumerate(range(0, len(selected), per_shard)):
        chunk = selected[start : start + per_shard]
        shard_path = os.path.join(out_dir, f"ytvis-validation-{shard_i:06d}.tar")
        with tarfile.open(shard_path, "w") as out:
            for row in chunk:
                for ext in ("video.npy", "segmentations.npy"):
                    name = f"{row['video_id']}.{ext}"
                    data = payloads[name]
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    out.addfile(info, io.BytesIO(data))
        shard_paths.append(shard_path)
    return shard_paths


def save_plots(out_dir, videos, events):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths = []
    counts = np.array([v["n_cycles"] for v in videos])
    fig, ax = plt.subplots(figsize=(7, 4))
    hi = max(int(counts.max()), 1)
    ax.hist(counts, bins=np.arange(0, hi + 2) - 0.5, color="#3b6d8c", edgecolor="white")
    ax.set_xlabel("appear-then-disappear cycles per video")
    ax.set_ylabel("videos")
    ax.set_title("YTVIS 2021 validation")
    fig.tight_layout()
    hist_path = os.path.join(out_dir, "cycle_count_hist.png")
    fig.savefig(hist_path, dpi=140)
    plt.close(fig)
    paths.append(hist_path)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    for ax, key, title in (
        (axes[0], "appear_bin", "appearance location"),
        (axes[1], "disappear_bin", "disappearance location"),
    ):
        grid = np.array(
            [grid_counts(events, key)[name] for name in BIN3_NAMES], dtype=float
        ).reshape(3, 3)
        im = ax.imshow(grid, cmap="YlOrRd")
        ax.set_xticks([0, 1, 2], ["left", "center", "right"])
        ax.set_yticks([0, 1, 2], ["top", "middle", "bottom"])
        ax.set_title(title)
        for r in range(3):
            for c in range(3):
                ax.text(c, r, int(grid[r, c]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    loc_path = os.path.join(out_dir, "location_3x3.png")
    fig.savefig(loc_path, dpi=140)
    plt.close(fig)
    paths.append(loc_path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        default="/data/ytvis2021_resized",
    )
    parser.add_argument(
        "--stats-dir",
        default="/workspace/SlotCurri/event_analysis/ytvis_val_appear_disappear",
    )
    parser.add_argument(
        "--subset-dir",
        default="/data/ytvis2021_val_subsets/appear_disappear_high",
    )
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--min-duration", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--per-shard", type=int, default=10)
    args = parser.parse_args()

    import torch.multiprocessing as tmp

    os.makedirs(args.stats_dir, exist_ok=True)
    shard_paths = [
        os.path.join(args.src_root, f"ytvis-validation-{i:06d}.tar") for i in range(30)
    ]
    missing = [p for p in shard_paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"missing shards: {missing[:3]}")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_gpu = len([x for x in visible.split(",") if x.strip() != ""]) or 2
    gpu_names = []
    print(f"spawning {n_gpu} GPU workers (CUDA_VISIBLE_DEVICES={visible or 'all visible'})", flush=True)
    chunks = [shard_paths[i::n_gpu] for i in range(n_gpu)]
    ctx = tmp.get_context("spawn")
    procs = []
    part_files = []
    for device_index, chunk in enumerate(chunks):
        part = os.path.join(args.stats_dir, f".part_gpu{device_index}.pkl")
        part_files.append(part)
        proc = ctx.Process(
            target=_gpu_worker,
            args=(device_index, chunk, args.min_area, args.min_duration, part),
        )
        proc.start()
        procs.append(proc)
    failed = []
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            failed.append(proc.exitcode)
    if failed:
        raise SystemExit(f"GPU workers failed: {failed}")

    videos = []
    events = []
    for part in part_files:
        with open(part, "rb") as handle:
            payload = pickle.load(handle)
        videos.extend(payload["rows"])
        events.extend(payload["events"])
        gpu_names.append(f"cuda:{payload['device_index']} {payload['gpu']}")

    videos.sort(key=lambda r: (-r["n_cycles"], -r["cycle_per_frame"], r["video_id"]))
    counts = np.array([v["n_cycles"] for v in videos], dtype=np.int32)
    p50, p75, p90 = [float(x) for x in np.percentile(counts, [50, 75, 90])]
    # At least two in-clip lifecycles. If that is more than half the set, use p75.
    threshold = 2
    if int((counts >= threshold).sum()) > int(0.5 * len(counts)):
        threshold = max(2, int(np.ceil(p75)))
    high = [v for v in videos if v["n_cycles"] >= threshold]
    any_cycle = [v for v in videos if v["n_cycles"] >= 1]
    none = [v for v in videos if v["n_cycles"] == 0]

    def side_majority(row, kind):
        if row["n_cycles"] <= 0:
            return "none"
        border = row[f"n_border_{kind}"]
        interior = row[f"n_interior_{kind}"]
        if border > interior:
            return "border"
        if interior > border:
            return "interior"
        return "tie"

    for row in videos:
        row["appear_place"] = side_majority(row, "appear")
        row["disappear_place"] = side_majority(row, "disappear")
        row["in_high_subset"] = int(row["n_cycles"] >= threshold)

    n_events = len(events)
    summary = {
        "gpus": gpu_names,
        "definition": {
            "visible": f"GT mask pixel count >= {args.min_area}",
            "cycle": (
                "contiguous visible run with start > 0, end < T-1, "
                f"and duration >= {args.min_duration} frames"
            ),
            "location": "mask centroid on the first visible frame (appear) and last visible frame (disappear), normalized by width/height",
            "border": "mask touches the outer 2 pixels of the frame; otherwise interior",
            "high_subset": f"videos with n_cycles >= {threshold}",
        },
        "n_videos": len(videos),
        "n_cycles": n_events,
        "n_videos_with_cycle": len(any_cycle),
        "n_videos_without_cycle": len(none),
        "n_high": len(high),
        "high_threshold_cycles": threshold,
        "cycles_per_video": {
            "mean": float(counts.mean()),
            "std": float(counts.std()),
            "min": int(counts.min()),
            "p50": p50,
            "p75": p75,
            "p90": p90,
            "max": int(counts.max()),
            "histogram": {str(k): int(v) for k, v in sorted(Counter(counts.tolist()).items())},
        },
        "entries": int(sum(v["n_entries"] for v in videos)),
        "exits": int(sum(v["n_exits"] for v in videos)),
        "persistent_runs": int(sum(v["n_persistent_runs"] for v in videos)),
        "temporary_gaps": int(sum(v["n_gaps"] for v in videos)),
        "appear_border_vs_interior": {
            "border": int(sum(1 for e in events if not e["appear_interior"])),
            "interior": int(sum(e["appear_interior"] for e in events)),
        },
        "disappear_border_vs_interior": {
            "border": int(sum(1 for e in events if not e["disappear_interior"])),
            "interior": int(sum(e["disappear_interior"] for e in events)),
        },
        "appear_border_detail": count_named(e["appear_border"] for e in events),
        "disappear_border_detail": count_named(e["disappear_border"] for e in events),
        "appear_grid_3x3": grid_counts(events, "appear_bin"),
        "disappear_grid_3x3": grid_counts(events, "disappear_bin"),
        "category_cycles": count_named(e["category_name"] for e in events),
        "appear_time_decile": count_named(
            str(min(9, int(e["appear_frame_frac"] * 10))) for e in events
        ),
        "disappear_time_decile": count_named(
            str(min(9, int(e["disappear_frame_frac"] * 10))) for e in events
        ),
        "high_appear_place": count_named(v["appear_place"] for v in high),
        "high_disappear_place": count_named(v["disappear_place"] for v in high),
        "duration_frames": {
            "mean": float(np.mean([e["duration"] for e in events])) if events else 0.0,
            "p50": float(np.median([e["duration"] for e in events])) if events else 0.0,
            "histogram": count_named(str(e["duration"]) for e in events),
        },
    }

    video_fields = [
        "video_id",
        "src_tar",
        "T",
        "H",
        "W",
        "n_objects",
        "n_cycles",
        "n_entries",
        "n_exits",
        "n_persistent_runs",
        "n_gaps",
        "cycle_per_frame",
        "n_border_appear",
        "n_interior_appear",
        "n_border_disappear",
        "n_interior_disappear",
        "appear_place",
        "disappear_place",
        "in_high_subset",
        "categories",
    ]
    event_fields = [
        "video_id",
        "obj_index",
        "category_id",
        "category_name",
        "start",
        "end",
        "duration",
        "T",
        "H",
        "W",
        "appear_frame_frac",
        "disappear_frame_frac",
        "appear_cx",
        "appear_cy",
        "disappear_cx",
        "disappear_cy",
        "appear_bin",
        "disappear_bin",
        "appear_border",
        "disappear_border",
        "appear_interior",
        "disappear_interior",
        "mean_area",
        "max_area",
    ]
    write_csv(os.path.join(args.stats_dir, "per_video.csv"), videos, video_fields)
    write_csv(os.path.join(args.stats_dir, "per_event.csv"), events, event_fields)

    shard_paths_out = write_shards(high, args.src_root, args.subset_dir, args.per_shard)
    if shard_paths_out:
        last = len(shard_paths_out) - 1
        brace = f"{{000000..{last:06d}}}"
    else:
        brace = "{000000..000000}"
    rel_subset = "ytvis2021_val_subsets/appear_disappear_high/ytvis-validation-" + brace + ".tar"
    summary["subset"] = {
        "n_videos": len(high),
        "threshold_cycles": threshold,
        "order": "n_cycles descending",
        "shards": [os.path.basename(p) for p in shard_paths_out],
        "val_shards": rel_subset,
        "video_ids": [v["video_id"] for v in high],
    }
    bins = {
        "none": [v["video_id"] for v in none],
        "any": [v["video_id"] for v in any_cycle],
        "high": [v["video_id"] for v in high],
        "high_border_appear": [
            v["video_id"] for v in high if v["appear_place"] == "border"
        ],
        "high_interior_appear": [
            v["video_id"] for v in high if v["appear_place"] == "interior"
        ],
    }
    with open(os.path.join(args.stats_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.stats_dir, "subset_ids.json"), "w") as f:
        json.dump({"threshold_cycles": threshold, "bins": bins}, f, indent=2)
    with open(os.path.join(args.subset_dir, "manifest.json"), "w") as f:
        json.dump(
            {
                "threshold_cycles": threshold,
                "n_videos": len(high),
                "val_shards": rel_subset,
                "videos": [
                    {
                        "video_id": v["video_id"],
                        "n_cycles": v["n_cycles"],
                        "appear_place": v["appear_place"],
                        "disappear_place": v["disappear_place"],
                    }
                    for v in high
                ],
            },
            f,
            indent=2,
        )
    plots = save_plots(args.stats_dir, videos, events)
    print(json.dumps({k: summary[k] for k in (
        "n_videos",
        "n_cycles",
        "n_videos_with_cycle",
        "n_videos_without_cycle",
        "n_high",
        "high_threshold_cycles",
        "cycles_per_video",
        "entries",
        "exits",
        "persistent_runs",
        "temporary_gaps",
        "appear_border_vs_interior",
        "disappear_border_vs_interior",
        "appear_grid_3x3",
        "disappear_grid_3x3",
        "high_appear_place",
        "high_disappear_place",
        "duration_frames",
    )}, indent=2))
    print("subset", rel_subset, "shards", len(shard_paths_out))
    print("plots", plots)


if __name__ == "__main__":
    main()
