#!/usr/bin/env bash
# Launch YTVIS attn-mass v24 (v21 + state_p_mult floor) on 2 GPUs.
#
# v24 vs v21: same linear p schedule (1.5 -> 0.1), but the predictor re-gate uses a
# floor at state_p_mult=0.3 instead of following decoder p into the flat tail.
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v24.sh
#   GPUS=6,7 bash scripts/launch_ytvis_attnmass_v24.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-6,7}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v24_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v24.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v24 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v24" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v24_run.out 2>&1"

echo "container: slotcurri_attnmass_v24"
echo "tail -f ${OUT_LOG}"
