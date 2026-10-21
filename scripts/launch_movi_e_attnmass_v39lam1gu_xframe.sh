#!/usr/bin/env bash
# Launch MOVi-E attn-mass v39lam1gu_xframe.
# Same as movi_e_attnmass_v39lam1gu.yaml except feature_curriculum.xvid_energy: frame
# (X^rel <- mu + g (m_f - mu) + (P X - m_f), m_f = frame mean of P X).
# Backbone / pool follow movi_e.yaml (ViT-B/14, 15 slots, slot dim 128, 336px,
# ignore_background).
#
# Usage:
#   GPUS=6,7 bash scripts/launch_movi_e_attnmass_v39lam1gu_xframe.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=6,7}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_e_attnmass_v39lam1gu_xframe_run.out"
CFG=configs/slotcurri/movi_e_attnmass_v39lam1gu_xframe.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-E v39lam1gu_xframe on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_e_attnmass_v39lam1gu_xframe" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/movi_e_attnmass_v39lam1gu_xframe_run.out 2>&1"

echo "container: slotcurri_movi_e_attnmass_v39lam1gu_xframe"
echo "tail -f ${OUT_LOG}"
