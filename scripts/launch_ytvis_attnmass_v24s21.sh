#!/usr/bin/env bash
# Launch v24 with v21's seed on 2 GPUs (default 1,2).
#
# Diff vs failed v24: seed 1148460988 -> 556986375 (v21).
# Diff vs v21: state_p_mult=0.3 only.
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v24s21.sh
#   GPUS=1,2 bash scripts/launch_ytvis_attnmass_v24s21.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-1,2}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v24s21_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v24s21.yaml
NAME=slotcurri_attnmass_v24s21

mkdir -p "${ROOT}/logs"
echo "Launching v24s21 on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "${NAME}" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v24s21_run.out 2>&1"

echo "container: ${NAME}"
echo "tail -f ${OUT_LOG}"
