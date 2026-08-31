#!/usr/bin/env bash
# Launch YTVIS attn-mass v39s (v39 8-nbr G_s, π = [λ1 - λ2⁺/(λ1+ε)]_+).
#
# v39s vs v39:
#   - Feature curriculum, decoder, SlotAttention, Pred: unchanged.
#   - π_s = [λ1 - max(λ2, 0) / (λ1 + ε)]_+  instead of λ1 - max(λ2, 0).
#     conf_kind=spectral_graph_n8_l1imp.
#
# Usage:
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v39s.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v39s_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v39s.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v39s on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v39s" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v39s_run.out 2>&1"

echo "container: slotcurri_attnmass_v39s"
echo "tail -f ${OUT_LOG}"
