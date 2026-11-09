#!/usr/bin/env bash
# Launch YTVIS SlotContrast + the v39lam1gu feature-curriculum schedule.
# Cosine anneal over 50k. No attention-mass gate, no xvid_energy.
#
# Usage:
#   GPUS=0,1 bash scripts/launch_ytvis_slotcontrast_v39lam1gu.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=0,1}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_slotcontrast_v39lam1gu_run.out"
CFG=configs/slotcurri/ytvis2021_slotcontrast_v39lam1gu.yaml

mkdir -p "${ROOT}/logs"
echo "Launching YTVIS slotcontrast v39lam1gu on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_ytvis_slotcontrast_v39lam1gu" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_slotcontrast_v39lam1gu_run.out 2>&1"

echo "container: slotcurri_ytvis_slotcontrast_v39lam1gu"
echo "tail -f ${OUT_LOG}"
