#!/usr/bin/env bash
# Launch YTVIS attn-mass v37r (v37 feature Gram, π = (λ1-λ2)/(λ1+λ2)).
#
# v37r vs v37g:
#   - Feature curriculum unchanged: ReLU-cosine P X, Key-only, barrier=false, 50k mix.
#   - Same C_s = Z^T diag(a^2) Z. Gate is the bounded λ1/λ2 map
#     (λ1-λ2)/(λ1+λ2) instead of the raw gap λ1-λ2.
#     conf_kind=spectral_ratio. spectral_proj_dim=64 (set 0 for full D).
#
# Usage:
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v37r.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v37r_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v37r.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v37r on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v37r" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v37r_run.out 2>&1"

echo "container: slotcurri_attnmass_v37r"
echo "tail -f ${OUT_LOG}"
