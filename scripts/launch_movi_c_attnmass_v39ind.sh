#!/usr/bin/env bash
# Launch MOVi-C attn-mass v39ind (same method as ytvis2021_attnmass_v39ind.yaml).
#
# Usage:
#   GPUS=6,7 bash scripts/launch_movi_c_attnmass_v39ind.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=6,7}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_c_attnmass_v39ind_run.out"
CFG=configs/slotcurri/movi_c_attnmass_v39ind.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-C v39ind on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_c_attnmass_v39ind" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/movi_c_attnmass_v39ind_run.out 2>&1"

echo "container: slotcurri_movi_c_attnmass_v39ind"
echo "tail -f ${OUT_LOG}"
