#!/usr/bin/env bash
# Launch YTVIS attn-mass v39l1 (v39 + temporal mix on n8 λ1).
#
# v39l1 vs v39:
#   - Feature curriculum, 8-neighbor π, decoder, SlotAttention, Pred: unchanged.
#   - Decoder still softmax(alpha + log(π+eps)).
#   - Temporal mix uses λ1 / max(λ1) instead of π / max(π).
#     state_conf_kind=lambda1 (not gate_l1, which stays 0).
#
# Usage:
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v39l1.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v39l1_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v39l1.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v39l1 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v39l1" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v39l1_run.out 2>&1"

echo "container: slotcurri_attnmass_v39l1"
echo "tail -f ${OUT_LOG}"
