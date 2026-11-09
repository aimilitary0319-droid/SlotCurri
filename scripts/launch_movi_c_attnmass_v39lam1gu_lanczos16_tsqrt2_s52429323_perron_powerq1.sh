#!/usr/bin/env bash
# Eval the finished MOVi-C lanczos16_tsqrt2 checkpoint with Perron readout.
# λ1 stays 16-step Lanczos. q1 is the power-iteration Perron vector.
# Weights: logs/_movi_c_attnmass_v39lam1gu_lanczos16_tsqrt2_s52429323/checkpoints/slotcurri_step=step=100000-v1.ckpt
# Seed 52429323. Eval: vanilla Pred SA, Perron decoder readout, live-live / ghost-ghost Pred SA.
#
# Usage:
#   GPUS=6 bash scripts/launch_movi_c_attnmass_v39lam1gu_lanczos16_tsqrt2_s52429323_perron_powerq1.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=6}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_c_attnmass_v39lam1gu_lanczos16_tsqrt2_s52429323_perron_powerq1_run.out"
CFG=/workspace/SlotCurri/configs/slotcurri/movi_c_attnmass_v39lam1gu_lanczos16_tsqrt2_s52429323_perron_powerq1.yaml

IFS=',' read -ra GPU_ARR <<< "${GPUS}"
VISIBLE="$(seq -s, 0 $((${#GPU_ARR[@]} - 1)))"

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-C lanczos16_tsqrt2 s52429323 power-q1 Perron eval on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_c_lanczos16_tsqrt2_s52429323_perron_powerq1" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES="${VISIBLE}" \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.eval --log-dir logs --config-file ${CFG} > logs/movi_c_attnmass_v39lam1gu_lanczos16_tsqrt2_s52429323_perron_powerq1_run.out 2>&1"

echo "container: slotcurri_movi_c_lanczos16_tsqrt2_s52429323_perron_powerq1"
echo "tail -f ${OUT_LOG}"
