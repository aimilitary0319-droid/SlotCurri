#!/usr/bin/env bash
# Launch YTVIS attn-mass v35 (v32 purity gating + mean-shift mode-cover curriculum)
# on 2 GPUs.
#
# v35 vs v34: tokens are covered by density-mode means. Clock is bandwidth
#   h(t) = h0 * (1 - s(t)), s cosine 0->1 over 30k
# C = #{modes} is scene-dependent. Eval always runs on raw features.
# Does not start the job unless you run this script.
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v35.sh
#   GPUS=6,7 bash scripts/launch_ytvis_attnmass_v35.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-6,7}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v35_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v35.yaml

mkdir -p "${ROOT}/logs"
echo "Launching YTVIS v35 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v35" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v35_run.out 2>&1"

echo "container: slotcurri_attnmass_v35"
echo "tail -f ${OUT_LOG}"
