#!/usr/bin/env bash
# Launch MOVi-C attn-mass v44 (softmax z, sg(z) recon, ungated drop + σ-m insert, ψ=0.05).
#
# Usage:
#   GPUS=4,5 bash scripts/launch_movi_c_attnmass_v44.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_c_attnmass_v44_run.out"
CFG=configs/slotcurri/movi_c_attnmass_v44.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-C v44 on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_c_attnmass_v44" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/movi_c_attnmass_v44_run.out 2>&1"

echo "container: slotcurri_movi_c_attnmass_v44"
echo "tail -f ${OUT_LOG}"
