#!/usr/bin/env bash
# Launch a second MOVi-C attn-mass v39lam1gu_xframe_lin40 from scratch.
# Same hyperparameters as movi_c_attnmass_v39lam1gu_xframe_lin40.yaml.
# Logs go to logs/_movi_c_attnmass_v39lam1gu_xframe_lin40_r2, so the GPU 2-3
# run, the finished archive, and the kept checkpoints are left in place.
#
# This script does not pass --continue.
#
# Usage:
#   GPUS=4,5 bash scripts/launch_movi_c_attnmass_v39lam1gu_xframe_lin40_r2.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_c_attnmass_v39lam1gu_xframe_lin40_r2_run.out"
CFG=configs/slotcurri/movi_c_attnmass_v39lam1gu_xframe_lin40_r2.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-C v39lam1gu_xframe_lin40 r2 on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_c_attnmass_v39lam1gu_xframe_lin40_r2" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/movi_c_attnmass_v39lam1gu_xframe_lin40_r2_run.out 2>&1"

echo "container: slotcurri_movi_c_attnmass_v39lam1gu_xframe_lin40_r2"
echo "tail -f ${OUT_LOG}"
