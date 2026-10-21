#!/usr/bin/env bash
# Launch MOVi-E attn-mass v39lam1gu_xbal.
# Same as movi_e_attnmass_v39lam1gu_xvid.yaml except feature_curriculum.xvid_energy: balance
# (X^rel <- mu + g (P X - mu), g puts the centered batch-shared energy equal to the
# rest, clamped to [1, xvid gain]). Backbone / pool follow movi_e.yaml (ViT-B/14,
# 15 slots, slot dim 128, 336px, ignore_background).
#
# Usage:
#   GPUS=6,7 bash scripts/launch_movi_e_attnmass_v39lam1gu_xbal.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=6,7}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_e_attnmass_v39lam1gu_xbal_run.out"
CFG=configs/slotcurri/movi_e_attnmass_v39lam1gu_xbal.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-E v39lam1gu_xbal on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_e_attnmass_v39lam1gu_xbal" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/movi_e_attnmass_v39lam1gu_xbal_run.out 2>&1"

echo "container: slotcurri_movi_e_attnmass_v39lam1gu_xbal"
echo "tail -f ${OUT_LOG}"
