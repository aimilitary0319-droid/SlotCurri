#!/usr/bin/env bash
# Eval-only Perron of the kept MOVi-E xframe_lin40 step-135000 checkpoint.
# Weights: logs/_movi_e_attnmass_v39lam1gu_xframe_lin40_kept/slotcurri_step=step=135000.ckpt
# Does not touch the 133000 Perron result or the live GPU 0-1 / 2-3 training runs.
# Eval: vanilla Pred SA, Perron decoder readout, live-live / ghost-ghost Pred SA.
#
# Usage:
#   GPUS=4,5 bash scripts/launch_movi_e_attnmass_v39lam1gu_xframe_lin40_perron135k.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_e_attnmass_v39lam1gu_xframe_lin40_perron135k_run.out"
CFG=/workspace/SlotCurri/configs/slotcurri/movi_e_attnmass_v39lam1gu_xframe_lin40_perron135k.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-E v39lam1gu_xframe_lin40 Perron135k eval on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_e_attnmass_v39lam1gu_xframe_lin40_perron135k" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.eval --log-dir logs --config-file ${CFG} > logs/movi_e_attnmass_v39lam1gu_xframe_lin40_perron135k_run.out 2>&1"

echo "container: slotcurri_movi_e_attnmass_v39lam1gu_xframe_lin40_perron135k"
echo "tail -f ${OUT_LOG}"
