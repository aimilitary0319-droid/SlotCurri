#!/usr/bin/env bash
# Eval the MOVi-C tsqrt2 seed 1769938927 checkpoint with eval-only Perron.
# Weights: logs/_movi_c_attnmass_v39lam1gu_tsqrt2_s1769938927/checkpoints/slotcurri_step=step=100000-v1.ckpt
# Eval: vanilla Pred SA, Perron decoder readout, live-live / ghost-ghost Pred SA.
#
# Usage:
#   GPUS=4,6 bash scripts/launch_movi_c_attnmass_v39lam1gu_tsqrt2_s1769938927_perron.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=4,6}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_c_attnmass_v39lam1gu_tsqrt2_s1769938927_perron_run.out"
CFG=/workspace/SlotCurri/configs/slotcurri/movi_c_attnmass_v39lam1gu_tsqrt2_s1769938927_perron.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-C tsqrt2 s1769938927 Perron eval on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_c_tsqrt2_s1769938927_perron" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.eval --log-dir logs --config-file ${CFG} > logs/movi_c_attnmass_v39lam1gu_tsqrt2_s1769938927_perron_run.out 2>&1"

echo "container: slotcurri_movi_c_tsqrt2_s1769938927_perron"
echo "tail -f ${OUT_LOG}"
