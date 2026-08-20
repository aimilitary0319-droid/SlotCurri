#!/usr/bin/env bash
# Launch MOVi-C attn-mass v29sg: v29 with sg(g) and p: 1.0/11 -> 0.1/11.
#
# Usage:
#   bash scripts/launch_movi_c_attnmass_v29sg.sh
#   GPUS=6,7 bash scripts/launch_movi_c_attnmass_v29sg.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-6,7}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_c_attnmass_v29sg_run.out"
CFG=configs/slotcurri/movi_c_attnmass_v29sg.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-C v29sg on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_c_attnmass_v29sg" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/movi_c_attnmass_v29sg_run.out 2>&1"

echo "container: slotcurri_movi_c_attnmass_v29sg"
echo "tail -f ${OUT_LOG}"
