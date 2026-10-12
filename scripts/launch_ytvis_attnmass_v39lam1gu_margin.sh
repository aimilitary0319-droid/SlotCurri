#!/usr/bin/env bash
# Launch YTVIS attn-mass v39lam1gu_margin (7 slots).
# Slot update and π share α_s = σ(z_s - max_{k≠s} z_k). No slot-axis renorm.
# NUM_SLOTS 7. Train num_classes 10, val 25.
#
# Usage:
#   GPUS=6,7 bash scripts/launch_ytvis_attnmass_v39lam1gu_margin.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=6,7}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v39lam1gu_margin_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v39lam1gu_margin.yaml

mkdir -p "${ROOT}/logs"
echo "Launching YTVIS v39lam1gu_margin on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_ytvis_attnmass_v39lam1gu_margin" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v39lam1gu_margin_run.out 2>&1"

echo "container: slotcurri_ytvis_attnmass_v39lam1gu_margin"
echo "tail -f ${OUT_LOG}"
