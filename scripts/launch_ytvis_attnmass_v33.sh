#!/usr/bin/env bash
# Launch YTVIS attn-mass v33 (v32 purity gating + feature curriculum) on 2 GPUs.
#
# v33 vs v32: single addition -- the backbone tokens are annealed from affinity-smoothed
# (object-level, within-object variance removed) to raw patch features over the first
# 30k steps. Early on a part-split earns nothing and cannot hold clean ownership, so the
# purity gate's one blind spot (clean head/torso splits) is converted into the
# mixed-ownership regime it already suppresses. Eval always runs on raw features.
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v33.sh
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v33.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v33_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v33.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v33 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v33" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v33_run.out 2>&1"

echo "container: slotcurri_attnmass_v33"
echo "tail -f ${OUT_LOG}"
