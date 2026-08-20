#!/usr/bin/env bash
# Launch MOVi-C attn-mass v26 (YTVIS v26 method: entropy log-ratio gate + coupled
# curriculum) on 2 GPUs.
#
# Same gate as ytvis2021_attnmass_v26.yaml. Only data / encoder / slot count change
# (11 slots, ViT-S/14, 336).
#
# Usage:
#   bash scripts/launch_movi_c_attnmass_v26.sh
#   GPUS=0,1 bash scripts/launch_movi_c_attnmass_v26.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-0,1}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_c_attnmass_v26_run.out"
CFG=configs/slotcurri/movi_c_attnmass_v26.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-C v26 on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_c_attnmass_v26" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/movi_c_attnmass_v26_run.out 2>&1"

echo "container: slotcurri_movi_c_attnmass_v26"
echo "tail -f ${OUT_LOG}"
