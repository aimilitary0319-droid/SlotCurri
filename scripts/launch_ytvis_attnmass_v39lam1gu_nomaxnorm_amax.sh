#!/usr/bin/env bash
# Launch YTVIS attn-mass v39lam1gu_nomaxnorm_amax.
# Same method as movi_e_attnmass_v39lam1gu_nomaxnorm_amax.yaml:
# state_max_norm: false, and a_slot_max so G reads a / max_s a.
#
# Usage:
#   GPUS=0,1 bash scripts/launch_ytvis_attnmass_v39lam1gu_nomaxnorm_amax.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=0,1}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v39lam1gu_nomaxnorm_amax_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v39lam1gu_nomaxnorm_amax.yaml

mkdir -p "${ROOT}/logs"
echo "Launching YTVIS v39lam1gu_nomaxnorm_amax on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_ytvis_attnmass_v39lam1gu_nomaxnorm_amax" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v39lam1gu_nomaxnorm_amax_run.out 2>&1"

echo "container: slotcurri_ytvis_attnmass_v39lam1gu_nomaxnorm_amax"
echo "tail -f ${OUT_LOG}"
