#!/usr/bin/env bash
# Launch YTVIS attn-mass v39d (v39 π + decoder-only leaky-max γ=0.85).
#
# v39d vs v39h / v39ema:
#   - π formula, curriculum, SlotAttention, Pred: unchanged.
#   - Decoder: π̃_t = max(π_t, 0.85 π̃_{t-1})
#   - Temporal mix: instantaneous π (then max-norm). No state_gate_ema.
#
# Usage:
#   GPUS=2,3 bash scripts/launch_ytvis_attnmass_v39d.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=2,3}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v39d_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v39d.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v39d on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v39d" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v39d_run.out 2>&1"

echo "container: slotcurri_attnmass_v39d"
echo "tail -f ${OUT_LOG}"
