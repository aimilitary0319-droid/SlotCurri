#!/usr/bin/env bash
# Launch MOVi-E attn-mass v39lam1gu_w1.
# Same as movi_e_attnmass_v39lam1gu.yaml except loss_weights.
# loss_featrec, loss_featrec_ungated, and loss_ss are all 1.0.
#
# Usage:
#   GPUS=0,1 bash scripts/launch_movi_e_attnmass_v39lam1gu_w1.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=0,1}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_e_attnmass_v39lam1gu_w1_run.out"
CFG=configs/slotcurri/movi_e_attnmass_v39lam1gu_w1.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-E v39lam1gu_w1 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_e_attnmass_v39lam1gu_w1" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/movi_e_attnmass_v39lam1gu_w1_run.out 2>&1"

echo "container: slotcurri_movi_e_attnmass_v39lam1gu_w1"
echo "tail -f ${OUT_LOG}"
