#!/usr/bin/env bash
# Launch YTVIS v39lam1gu_umix_perron_occkernel (umix hold + eval Perron + occ_kernel).
#
# Usage:
#   GPUS=2,3 bash scripts/launch_ytvis_attnmass_v39lam1gu_umix_perron_occkernel.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=2,3}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v39lam1gu_umix_perron_occkernel_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v39lam1gu_umix_perron_occkernel.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v39lam1gu_umix_perron_occkernel on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v39lam1gu_umix_perron_occkernel" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v39lam1gu_umix_perron_occkernel_run.out 2>&1"

echo "container: slotcurri_attnmass_v39lam1gu_umix_perron_occkernel"
echo "tail -f ${OUT_LOG}"
