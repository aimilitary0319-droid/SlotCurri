#!/usr/bin/env bash
# Launch YTVIS attn-mass v39si (v39s π + v39i identity cosine on the mix).
#
# v39si vs v39s:
#   - π is unchanged: [λ1 - λ2⁺/(λ1+ε)]_+  (conf_kind=spectral_graph_n8_l1imp).
#   - Temporal mix gate is ρ = π̄ ⊙ ReLU(cos(û_t, u_t)). Decoder still sees π.
#
# Usage:
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v39si.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v39si_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v39si.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v39si on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v39si" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v39si_run.out 2>&1"

echo "container: slotcurri_attnmass_v39si"
echo "tail -f ${OUT_LOG}"
