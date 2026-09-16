#!/usr/bin/env bash
# Launch MOVi-C attn-mass v37r (v37 feature Gram, π = (λ1-λ2)/(λ1+λ2)).
# Same method as ytvis2021_attnmass_v37r.yaml; backbone / pool follow
# movi_c_attnmass_v29.yaml (ViT-S/14, 11 slots, 336px, ignore_background).
# spectral_proj_dim=64 (set 0 for full D).
#
# Usage:
#   GPUS=6,7 bash scripts/launch_movi_c_attnmass_v37r.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=6,7}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_c_attnmass_v37r_run.out"
CFG=configs/slotcurri/movi_c_attnmass_v37r.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-C v37r on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_c_attnmass_v37r" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/movi_c_attnmass_v37r_run.out 2>&1"

echo "container: slotcurri_movi_c_attnmass_v37r"
echo "tail -f ${OUT_LOG}"
