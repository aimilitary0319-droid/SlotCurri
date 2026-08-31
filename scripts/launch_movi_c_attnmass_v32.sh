#!/usr/bin/env bash
# Launch MOVi-C v32 (purity_weight, no feature curriculum) on 2 GPUs.
# Paired control for movi_c_attnmass_v33.
#
# Usage:
#   GPUS=2,3 bash scripts/launch_movi_c_attnmass_v32.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS=e.g. 2,3}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_c_attnmass_v32_run.out"
CFG=configs/slotcurri/movi_c_attnmass_v32.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-C v32 on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_c_attnmass_v32" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/movi_c_attnmass_v32_run.out 2>&1"

echo "container: slotcurri_movi_c_attnmass_v32"
echo "tail -f ${OUT_LOG}"
