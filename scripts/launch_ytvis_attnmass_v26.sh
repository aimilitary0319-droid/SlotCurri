#!/usr/bin/env bash
# Launch YTVIS attn-mass v26 (final method: evidence-aware log-ratio gate + coupled
# activity curriculum) on 2 GPUs.
#
# v26 vs v21/v25: gate_form=logratio (coverage x sg(confidence) against log-ratio
# threshold), p_anneal=cosine over the full run, beta 1.0 -> 0.7 driven by the same
# lambda. No state_p_mult, no velocity conditioning.
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v26.sh
#   GPUS=2,4 bash scripts/launch_ytvis_attnmass_v26.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-2,4}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v26_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v26.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v26 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v26" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v26_run.out 2>&1"

echo "container: slotcurri_attnmass_v26"
echo "tail -f ${OUT_LOG}"
