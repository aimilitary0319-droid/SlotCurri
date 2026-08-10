#!/usr/bin/env bash
# Launch YTVIS attn-mass v22 (v20 ablation: p_end=0.2, tau=0.2, log) on 2 GPUs.
#
# v22 vs v20: same log schedule family and no SSIM / no cyclic; only
#   p_end_mult 0.3 -> 0.2
#   tau        0.3 -> 0.2
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v22.sh
#   GPUS=2,4 bash scripts/launch_ytvis_attnmass_v22.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-2,4}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v22_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v22.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v22 on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v22" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v22_run.out 2>&1"

echo "container: slotcurri_attnmass_v22"
echo "tail -f ${OUT_LOG}"
