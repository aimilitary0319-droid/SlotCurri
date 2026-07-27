#!/usr/bin/env bash
# Launch YTVIS attn-mass v9 FULL on 2 GPUs (default: physical 0,1).
# Run from host; uses the same image/pattern as v6/v7/v8.
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v9.sh          # GPUs 0,1
#   GPUS=2,3 bash scripts/launch_ytvis_attnmass_v9.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-0,1}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v9_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v9.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v9 on GPUs=${GPUS} -> ${OUT_LOG}"

# Match prior runs: map host GPUS into the container, then expose as 0,1 inside.
docker run -d --rm --gpus "device=${GPUS}" \
  --name "slotcurri_attnmass_v9" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v9_run.out 2>&1"

echo "container: slotcurri_attnmass_v9"
echo "tail -f ${OUT_LOG}"
