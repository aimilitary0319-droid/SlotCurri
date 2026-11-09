#!/usr/bin/env bash
# Launch MOVi-C attn-mass v39lam1gu_locgap_lin40.
# Same as movi_c_attnmass_v39lam1gu_locgap.yaml except schedule: linear
# and anneal_steps: 40000 (s = t/40000). xvid_energy stays loc_gap.
#
# Usage:
#   GPUS=2,3 bash scripts/launch_movi_c_attnmass_v39lam1gu_locgap_lin40.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=2,3}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_c_attnmass_v39lam1gu_locgap_lin40_run.out"
CFG=configs/slotcurri/movi_c_attnmass_v39lam1gu_locgap_lin40.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-C v39lam1gu_locgap_lin40 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_c_attnmass_v39lam1gu_locgap_lin40" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/movi_c_attnmass_v39lam1gu_locgap_lin40_run.out 2>&1"

echo "container: slotcurri_movi_c_attnmass_v39lam1gu_locgap_lin40"
echo "tail -f ${OUT_LOG}"
