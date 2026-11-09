#!/usr/bin/env bash
# Eval-only Perron of the finished YTVIS SlotContrast xframe_lin40 step-100000 checkpoint.
# Weights: logs/_ytvis_slotcontrast_xframe_lin40/checkpoints/slotcurri_step=step=100000-v1.ckpt
# Does not touch the live MOVi-E r2 run on GPU 2-3.
# Eval: vanilla Pred SA, Perron decoder readout, live-live / ghost-ghost Pred SA.
#
# Usage:
#   GPUS=0,1 bash scripts/launch_ytvis_slotcontrast_xframe_lin40_perron.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=0,1}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_slotcontrast_xframe_lin40_perron_run.out"
CFG=/workspace/SlotCurri/configs/slotcurri/ytvis2021_slotcontrast_xframe_lin40_perron.yaml

mkdir -p "${ROOT}/logs"
echo "Launching YTVIS slotcontrast xframe_lin40 Perron eval on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_ytvis_slotcontrast_xframe_lin40_perron" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.eval --log-dir logs --config-file ${CFG} > logs/ytvis_slotcontrast_xframe_lin40_perron_run.out 2>&1"

echo "container: slotcurri_ytvis_slotcontrast_xframe_lin40_perron"
echo "tail -f ${OUT_LOG}"
