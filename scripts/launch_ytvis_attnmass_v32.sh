#!/usr/bin/env bash
# Launch YTVIS attn-mass v32 (purity gating: the threshold curriculum replaced by the
# detached ownership statistic c = sum A~^2 / sum A~ used directly as the gate) on 2 GPUs.
#
# v32 vs v29/v26: gate_form=purity_weight -- no p schedule, no tau_g, no beta; decoder
# masks become softmax(alpha + log c) and the temporal mix uses c / max(c). Run the
# Step-0 check first (event_analysis/purity_weight_eval.py on the baseline / v26m / v29
# checkpoints) before spending the full 100k.
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v32.sh
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v32.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v32_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v32.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v32 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v32" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v32_run.out 2>&1"

echo "container: slotcurri_attnmass_v32"
echo "tail -f ${OUT_LOG}"
