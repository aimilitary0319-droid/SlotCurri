#!/usr/bin/env bash
# Launch MOVi-C attn-mass v34 (YTVIS v34 method: purity_weight + window-annealed
# feature curriculum) on 2 GPUs.
#
# Dataset / backbone / slot-count follow movi_c_attnmass_v29.yaml; the method is
# copied from ytvis2021_attnmass_v34.yaml. window_ref_grid=37 scales w0=13 -> weff~8.
#
# Usage:
#   bash scripts/launch_movi_c_attnmass_v34.sh
#   GPUS=4,5 bash scripts/launch_movi_c_attnmass_v34.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_c_attnmass_v34_run.out"
CFG=configs/slotcurri/movi_c_attnmass_v34.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-C v34 on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_c_attnmass_v34" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/movi_c_attnmass_v34_run.out 2>&1"

echo "container: slotcurri_movi_c_attnmass_v34"
echo "tail -f ${OUT_LOG}"
