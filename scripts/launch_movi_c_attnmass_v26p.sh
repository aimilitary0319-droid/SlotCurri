#!/usr/bin/env bash
# Launch MOVi-C attn-mass v26p (decoder mass curriculum + v36 purity temporal mix)
# on 2 GPUs.
#
# Same method as ytvis2021_attnmass_v26p.yaml. Only data / encoder / slot count
# change (11 slots, ViT-S/14, 336).
#
# Usage:
#   bash scripts/launch_movi_c_attnmass_v26p.sh
#   GPUS=0,1 bash scripts/launch_movi_c_attnmass_v26p.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-0,1}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_c_attnmass_v26p_run.out"
CFG=configs/slotcurri/movi_c_attnmass_v26p.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-C v26p on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_c_attnmass_v26p" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/movi_c_attnmass_v26p_run.out 2>&1"

echo "container: slotcurri_movi_c_attnmass_v26p"
echo "tail -f ${OUT_LOG}"
