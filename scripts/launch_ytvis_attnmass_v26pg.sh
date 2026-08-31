#!/usr/bin/env bash
# Launch YTVIS attn-mass v26pg (v26p temporal mix = purity ⊙ decoder mass).
# Default GPUs 0,1 (v26p uses 2,3; v38 uses 4-7).
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v26pg.sh
#   GPUS=0,1 bash scripts/launch_ytvis_attnmass_v26pg.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-0,1}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v26pg_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v26pg.yaml

mkdir -p "${ROOT}/logs"
echo "Launching YTVIS v26pg on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v26pg" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v26pg_run.out 2>&1"

echo "container: slotcurri_attnmass_v26pg"
echo "tail -f ${OUT_LOG}"
