#!/usr/bin/env bash
# Launch YTVIS attn-mass v29f (ablation of v29: purity confidence held at beta=0.7 from
# step 0 instead of ramping 1.0 -> 0.7 with the curriculum) on 2 GPUs.
#
# v29f vs v29: beta_start=0.7 is the ONLY change. Purity counterpart of v26f (which
# collapsed with the entropy confidence). On untrained attention log(purity) ~= -2 vs
# log(c_ent) ~= -6.6, so the fixed early penalty is ~3x milder here; if v29f trains ~= v29
# the beta ramp is unnecessary with purity and the method drops one schedule.
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v29f.sh
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v29f.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v29f_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v29f.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v29f on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v29f" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v29f_run.out 2>&1"

echo "container: slotcurri_attnmass_v29f"
echo "tail -f ${OUT_LOG}"
