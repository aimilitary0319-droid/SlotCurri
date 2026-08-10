#!/usr/bin/env bash
# Launch YTVIS attn-mass v17 (v10 + strict purity rescue) on 2 GPUs (default: 6,7).
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v17.sh
#   GPUS=0,1 bash scripts/launch_ytvis_attnmass_v17.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-6,7}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v17_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v17.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v17 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v17" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v17_run.out 2>&1"

echo "container: slotcurri_attnmass_v17"
echo "tail -f ${OUT_LOG}"
