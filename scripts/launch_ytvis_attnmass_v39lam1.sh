#!/usr/bin/env bash
# Launch YTVIS attn-mass v39lam1 (v39 with π := max(λ1, 0)).
#
# v39lam1 vs v39:
#   - Feature curriculum, SlotAttention, Pred: unchanged.
#   - Gate is the same 8-neighbor G_s, but π = max(λ1, 0) (no λ2).
#   - Decoder and temporal mix both see λ1 (unlike v39l1).
#     conf_kind=spectral_graph_n8_lam1.
#
# Usage:
#   GPUS=0,1 bash scripts/launch_ytvis_attnmass_v39lam1.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=0,1}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v39lam1_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v39lam1.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v39lam1 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v39lam1" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v39lam1_run.out 2>&1"

echo "container: slotcurri_attnmass_v39lam1"
echo "tail -f ${OUT_LOG}"
