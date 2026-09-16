#!/usr/bin/env bash
# Launch YTVIS attn-mass v39th (v39h leaky-max on the temporal mix only).
#
# v39th vs v39h:
#   - Feature curriculum, 8-neighbor π, SlotAttention, Pred: unchanged.
#   - Decoder stays on instantaneous π (v39 / not sticky).
#   - Temporal mix: π̃_t = max(π_t, 0.75 π̃_{t-1}), then α = π̃ / max(π̃).
#     state_gate_hold=0.75, state_gate_ema=1. No gate_hysteresis.
#
# Usage:
#   GPUS=2,3 bash scripts/launch_ytvis_attnmass_v39th.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=2,3}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v39th_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v39th.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v39th on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v39th" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v39th_run.out 2>&1"

echo "container: slotcurri_attnmass_v39th"
echo "tail -f ${OUT_LOG}"
