#!/usr/bin/env bash
# Launch YTVIS featcur (SlotContrast + feature curriculum, no purity gate) on 2 GPUs.
# Completes the 2x2 synergy table with SlotContrast / v32 / v33.
#
# Usage:
#   GPUS=0,1 bash scripts/launch_ytvis_featcur.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS=e.g. 0,1}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_featcur_run.out"
CFG=configs/slotcurri/ytvis2021_featcur.yaml

mkdir -p "${ROOT}/logs"
echo "Launching featcur on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_featcur" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_featcur_run.out 2>&1"

echo "container: slotcurri_featcur"
echo "tail -f ${OUT_LOG}"
