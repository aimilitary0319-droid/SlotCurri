#!/usr/bin/env bash
# Launch YTVIS attn-mass v39h (v39 π + shared leaky-max hysteresis).
#
# v39h vs v39ema:
#   - π formula, curriculum, SlotAttention, Pred: unchanged.
#   - Sticky statistic is applied BEFORE both paths:
#       π̃_t = max(π_t, 0.95 π̃_{t-1})
#     decoder and temporal mix both see π̃. No state_gate_ema.
#
# Usage:
#   GPUS=2,3 bash scripts/launch_ytvis_attnmass_v39h.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=2,3}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v39h_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v39h.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v39h on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v39h" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v39h_run.out 2>&1"

echo "container: slotcurri_attnmass_v39h"
echo "tail -f ${OUT_LOG}"
