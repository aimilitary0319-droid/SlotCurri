#!/usr/bin/env bash
# Launch YTVIS SlotContrast baseline with 7 slots on 2 GPUs.
#
# Usage:
#   bash scripts/launch_ytvis_slotcontrast.sh
#   GPUS=4,5 bash scripts/launch_ytvis_slotcontrast.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_slotcontrast_run.out"
CFG=configs/slotcurri/ytvis2021_slotcontrast.yaml

mkdir -p "${ROOT}/logs"
echo "Launching slotcontrast (7 slots) on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_slotcontrast" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_slotcontrast_run.out 2>&1"

echo "container: slotcurri_slotcontrast"
echo "tail -f ${OUT_LOG}"
