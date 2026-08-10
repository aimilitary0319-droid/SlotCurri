#!/usr/bin/env bash
# Launch YTVIS attn-mass v21 (v20 with p 1.5 -> 0.1 linear) on 2 GPUs (default: 0,1).
#
# v21 is the baseline the v23 (velocity predictor) and v24 (split gate threshold) runs are
# meant to be compared against: both are byte-identical to this config apart from their one
# change, so run this alongside them rather than comparing to an older lineage.
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v21.sh
#   GPUS=2,3 bash scripts/launch_ytvis_attnmass_v21.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-0,1}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v21_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v21.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v21 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v21" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v21_run.out 2>&1"

echo "container: slotcurri_attnmass_v21"
echo "tail -f ${OUT_LOG}"
