#!/usr/bin/env bash
# Launch YTVIS attn-mass v30 (v29 purity gate + L_pred) on 2 GPUs.
#
# Completes the 2x2 grid: v26 (base) / v28 (+L_pred) / v29 (+purity gate) / v30 (both).
#   conf_kind=purity_sharp: ownership confidence, halves ghost leakage (gate hygiene)
#   loss_pred (w=0.2): Dec(Pred(x_t)) reconstructs F_{t+1} (splits merged instances)
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v30.sh
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v30.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v30_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v30.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v30 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v30" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v30_run.out 2>&1"

echo "container: slotcurri_attnmass_v30"
echo "tail -f ${OUT_LOG}"
