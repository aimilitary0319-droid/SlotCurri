#!/usr/bin/env bash
# Launch YTVIS attn-mass v39lam1gu_entmax15_asqrt (7 slots).
# Slot attention is 1.5-entmax with its Jacobian. sqrt_a on.
# NUM_SLOTS 7. Train num_classes 10, val 25.
#
# Usage:
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v39lam1gu_entmax15_asqrt.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v39lam1gu_entmax15_asqrt_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v39lam1gu_entmax15_asqrt.yaml

mkdir -p "${ROOT}/logs"
echo "Launching YTVIS v39lam1gu_entmax15_asqrt on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_ytvis_attnmass_v39lam1gu_entmax15_asqrt" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v39lam1gu_entmax15_asqrt_run.out 2>&1"

echo "container: slotcurri_ytvis_attnmass_v39lam1gu_entmax15_asqrt"
echo "tail -f ${OUT_LOG}"
