#!/usr/bin/env bash
# Launch YTVIS attn-mass v39lam1g_xvid.
# Same as ytvis2021_attnmass_v39lam1g.yaml except feature_curriculum.xvid_energy: true
# (X^rel <- mu + g (P X - mu), mu = batch mean of X). No ungated L_featrec.
#
# Usage:
#   GPUS=0,1 bash scripts/launch_ytvis_attnmass_v39lam1g_xvid.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=0,1}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v39lam1g_xvid_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v39lam1g_xvid.yaml

mkdir -p "${ROOT}/logs"
echo "Launching YTVIS v39lam1g_xvid on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_ytvis_attnmass_v39lam1g_xvid" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v39lam1g_xvid_run.out 2>&1"

echo "container: slotcurri_ytvis_attnmass_v39lam1g_xvid"
echo "tail -f ${OUT_LOG}"
