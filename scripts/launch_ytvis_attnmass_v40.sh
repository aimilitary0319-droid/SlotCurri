#!/usr/bin/env bash
# Launch YTVIS attn-mass v40 (SlotContrast + K-normalized ownership gate +
# coupled slot-entropy / 8-neighbor impurity losses, T=50k, w0=0.2).
#
# Usage:
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v40.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v40_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v40.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v40 on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v40" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v40_run.out 2>&1"

echo "container: slotcurri_attnmass_v40"
echo "tail -f ${OUT_LOG}"
