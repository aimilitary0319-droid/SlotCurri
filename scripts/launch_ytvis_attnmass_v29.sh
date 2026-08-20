#!/usr/bin/env bash
# Launch YTVIS attn-mass v29 (v26 with the confidence branch switched from entropy to
# ownership purity on the gamma-sharpened attention) on 2 GPUs.
#
# v29 vs v26: conf_kind=purity_sharp is the ONLY change. Probe on v20 @ 100k
# (event_analysis/conf_vs_purity_probe.py): conf-only ghost-vs-small AUC 0.647 -> 0.999;
# at the unchanged working point (beta 0.7, tau_g 0.5, p_end 0.1/7) ghost passes drop
# 16.2% -> 9.3% and the ghost median gate 0.304 -> 0.125, while small-object gates rise.
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v29.sh
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v29.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v29_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v29.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v29 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v29" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v29_run.out 2>&1"

echo "container: slotcurri_attnmass_v29"
echo "tail -f ${OUT_LOG}"
