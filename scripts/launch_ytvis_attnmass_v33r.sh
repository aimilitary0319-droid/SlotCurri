#!/usr/bin/env bash
# Launch YTVIS attn-mass v33r (v33 feature curriculum + RAW-attention purity gate) on 2 GPUs.
#
# v33r vs v33: single change -- conf_kind=purity (raw attention) instead of purity_sharp.
# Same single-variable swap as v32 -> v32r, now under the feature curriculum.
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v33r.sh
#   GPUS=0,1 bash scripts/launch_ytvis_attnmass_v33r.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-0,1}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v33r_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v33r.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v33r on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v33r" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v33r_run.out 2>&1"

echo "container: slotcurri_attnmass_v33r"
echo "tail -f ${OUT_LOG}"
