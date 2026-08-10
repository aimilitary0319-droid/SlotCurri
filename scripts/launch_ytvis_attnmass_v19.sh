#!/usr/bin/env bash
# Launch YTVIS attn-mass v19 (v10 + attention centroid repulsion) on 2 GPUs (default: 0,1).
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v19.sh
#   GPUS=2,3 bash scripts/launch_ytvis_attnmass_v19.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-0,1}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v19_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v19.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v19 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v19" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v19_run.out 2>&1"

echo "container: slotcurri_attnmass_v19"
echo "tail -f ${OUT_LOG}"
