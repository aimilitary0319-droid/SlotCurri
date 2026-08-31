#!/usr/bin/env bash
# Launch YTVIS attn-mass v39s-ema (v39s π + temporal-mix EMA of π).
#
# v39sema vs v39s:
#   - Feature curriculum, decoder, SlotAttention, Pred, π formula: unchanged.
#   - Mix only: π̃_t = 0.2 π_t + 0.8 π̃_{t-1}; decoder still uses π_t.
#
# Usage:
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v39sema.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v39sema_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v39sema.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v39sema on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v39sema" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v39sema_run.out 2>&1"

echo "container: slotcurri_attnmass_v39sema"
echo "tail -f ${OUT_LOG}"
