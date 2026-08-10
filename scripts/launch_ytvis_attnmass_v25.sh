#!/usr/bin/env bash
# Launch YTVIS attn-mass v25 (v21 + velocity injection, no loss_dyn) on 2 GPUs.
#
# v25 vs v21: predictor receives previous displacement in K/V (vel_dim).
# v25 vs v23: same velocity path, but predictor_dynamics / loss_dyn is off -- featrec
# alone decides whether to use velocity (or ignore it via vel_gain).
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v25.sh
#   GPUS=2,4 bash scripts/launch_ytvis_attnmass_v25.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-2,4}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v25_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v25.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v25 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v25" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v25_run.out 2>&1"

echo "container: slotcurri_attnmass_v25"
echo "tail -f ${OUT_LOG}"
