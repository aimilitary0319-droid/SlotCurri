#!/usr/bin/env bash
# Eval the finished MOVi-C xframe_lin40 checkpoint with eval-only Perron.
# Weights: logs/_movi_c_attnmass_v39lam1gu_xframe_lin40/checkpoints/slotcurri_step=step=100000-v1.ckpt
# Eval: vanilla Pred SA, Perron decoder readout, live-live / ghost-ghost Pred SA.
#
# Usage:
#   GPUS=2,3 bash scripts/launch_movi_c_attnmass_v39lam1gu_xframe_lin40_perron.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=2,3}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_c_attnmass_v39lam1gu_xframe_lin40_perron_run.out"
CFG=/workspace/SlotCurri/configs/slotcurri/movi_c_attnmass_v39lam1gu_xframe_lin40_perron.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-C v39lam1gu_xframe_lin40 Perron eval on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_c_attnmass_v39lam1gu_xframe_lin40_perron" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.eval --log-dir logs --config-file ${CFG} > logs/movi_c_attnmass_v39lam1gu_xframe_lin40_perron_run.out 2>&1"

echo "container: slotcurri_movi_c_attnmass_v39lam1gu_xframe_lin40_perron"
echo "tail -f ${OUT_LOG}"
