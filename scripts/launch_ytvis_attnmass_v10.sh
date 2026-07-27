#!/usr/bin/env bash
# Launch YTVIS attn-mass v10 (v9 FULL + fixed tau=0.3) on 2 GPUs (default: 0,1).
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v10.sh
#   GPUS=2,3 bash scripts/launch_ytvis_attnmass_v10.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-0,1}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v10_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v10.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v10 on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "device=${GPUS}" \
  --name "slotcurri_attnmass_v10" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v10_run.out 2>&1"

echo "container: slotcurri_attnmass_v10"
echo "tail -f ${OUT_LOG}"
