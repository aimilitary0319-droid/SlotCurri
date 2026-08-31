#!/usr/bin/env bash
# Launch YTVIS attn-mass v26f0 (v26f with p_end_mult=0).
# Default GPUs 2,3 — same seat as YTVIS v26fmax.
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v26f0.sh
#   GPUS=2,3 bash scripts/launch_ytvis_attnmass_v26f0.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-2,3}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v26f0_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v26f0.yaml

mkdir -p "${ROOT}/logs"
echo "Launching YTVIS v26f0 on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v26f0" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v26f0_run.out 2>&1"

echo "container: slotcurri_attnmass_v26f0"
echo "tail -f ${OUT_LOG}"
