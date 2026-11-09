#!/usr/bin/env bash
# Plain eval of the kept MOVi-E xframe_lin40 step-133000 checkpoint.
# Same forward as training validation: no Perron readout, Pred source-gate stays on.
# Weights: logs/_movi_e_attnmass_v39lam1gu_xframe_lin40_kept/slotcurri_step=step=133000.ckpt
# Does not touch the live GPU 0-1 or GPU 2-3 training runs.
#
# Usage:
#   GPUS=4,5 bash scripts/launch_movi_e_attnmass_v39lam1gu_xframe_lin40_eval133k.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_e_attnmass_v39lam1gu_xframe_lin40_eval133k_run.out"
CFG=/workspace/SlotCurri/configs/slotcurri/movi_e_attnmass_v39lam1gu_xframe_lin40_eval133k.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-E v39lam1gu_xframe_lin40 plain eval133k on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_e_attnmass_v39lam1gu_xframe_lin40_eval133k" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.eval --log-dir logs --config-file ${CFG} > logs/movi_e_attnmass_v39lam1gu_xframe_lin40_eval133k_run.out 2>&1"

echo "container: slotcurri_movi_e_attnmass_v39lam1gu_xframe_lin40_eval133k"
echo "tail -f ${OUT_LOG}"
