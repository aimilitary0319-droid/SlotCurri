#!/usr/bin/env bash
# Launch COCO attn-mass v39lam1gu (stills as length-1 video).
# Build shards first: bash scripts/prepare_coco_shards.sh
#
# Usage:
#   GPUS=0,1 bash scripts/launch_coco_attnmass_v39lam1gu.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=0,1}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/coco_attnmass_v39lam1gu_run.out"
CFG=configs/slotcurri/coco_attnmass_v39lam1gu.yaml

mkdir -p "${ROOT}/logs"
echo "Launching COCO v39lam1gu on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_coco_attnmass_v39lam1gu" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/coco_attnmass_v39lam1gu_run.out 2>&1"

echo "container: slotcurri_coco_attnmass_v39lam1gu"
echo "tail -f ${OUT_LOG}"
