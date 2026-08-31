#!/usr/bin/env bash
# Launch YTVIS attn-mass v34 (v32 purity gating + window-annealed feature curriculum)
# on 2 GPUs.
#
# v34 vs v33: the curriculum clock is the Chebyshev radius, not the raw/smooth blend.
#   w(t) = round(13 * (1 - s(t))), s cosine 0->1 over 30k
# Fully smoothed at the current w while w>0; raw when w=0. tau=0.1, n_steps=3.
# Eval always runs on raw features. Does not start the job unless you run this script.
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v34.sh
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v34.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-2,3}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v34_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v34.yaml

mkdir -p "${ROOT}/logs"
echo "Launching YTVIS v34 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v34" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v34_run.out 2>&1"

echo "container: slotcurri_attnmass_v34"
echo "tail -f ${OUT_LOG}"
