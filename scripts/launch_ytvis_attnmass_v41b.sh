#!/usr/bin/env bash
# Launch YTVIS attn-mass v41b (v41 + L_ss anchors under sg(z_t) sg(z_{t+1})).
# gate_negatives stays false (log z is live).
#
# Usage:
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v41b.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v41b_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v41b.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v41b on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v41b" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v41b_run.out 2>&1"

echo "container: slotcurri_attnmass_v41b"
echo "tail -f ${OUT_LOG}"
