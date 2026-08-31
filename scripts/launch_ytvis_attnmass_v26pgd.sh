#!/usr/bin/env bash
# Launch YTVIS attn-mass v26pgd (v26pg + decoder masks = g ⊙ π).
# Does not default GPUs — 0,1 are the YTVIS v26pg default.
#
# Usage:
#   GPUS=0,1 bash scripts/launch_ytvis_attnmass_v26pgd.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS=i,j}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v26pgd_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v26pgd.yaml

mkdir -p "${ROOT}/logs"
echo "Launching YTVIS v26pgd on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v26pgd" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v26pgd_run.out 2>&1"

echo "container: slotcurri_attnmass_v26pgd"
echo "tail -f ${OUT_LOG}"
