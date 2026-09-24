#!/usr/bin/env bash
# Launch MOVi-C v39lam1gu_fcn8_perron_s18 (n8 Key curriculum + eval Perron, 18 slots).
#
# Usage:
#   GPUS=6,7 bash scripts/launch_movi_c_attnmass_v39lam1gu_fcn8_perron_s18.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=6,7}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_c_attnmass_v39lam1gu_fcn8_perron_s18_run.out"
CFG=configs/slotcurri/movi_c_attnmass_v39lam1gu_fcn8_perron_s18.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-C v39lam1gu_fcn8_perron_s18 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_c_attnmass_v39lam1gu_fcn8_perron_s18" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/movi_c_attnmass_v39lam1gu_fcn8_perron_s18_run.out 2>&1"

echo "container: slotcurri_movi_c_attnmass_v39lam1gu_fcn8_perron_s18"
echo "tail -f ${OUT_LOG}"
