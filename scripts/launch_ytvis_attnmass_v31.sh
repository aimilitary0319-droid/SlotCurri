#!/usr/bin/env bash
# Launch YTVIS attn-mass v31 (v29 purity gate + counterfactual slot-utility rent) on 2 GPUs.
#
# v31 vs v29: the slot_utility block is the ONLY change (weight 0.05, margin 1.0,
# lambda ramp). Per sample one gated slot is dropped, the batch re-decoded (no_grad),
# and rent relu(1 - rel_delta/margin) is charged on the slot's live gate: duplicates
# pay full rent (twin covers their territory), irreplaceable slots live rent-free.
# Targets the residual ghost class purity cannot see (slots that win junk patches).
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v31.sh
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v31.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v31_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v31.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v31 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v31" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v31_run.out 2>&1"

echo "container: slotcurri_attnmass_v31"
echo "tail -f ${OUT_LOG}"
