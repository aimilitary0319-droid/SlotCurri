#!/usr/bin/env bash
# Launch YTVIS attn-mass v27 (v26 + predictive feature reconstruction + counterfactual
# slot-utility rent) on 2 GPUs.
#
# v27 vs v26: two auxiliary losses, both ramped by the existing lambda_t.
#   loss_pred (w=0.2): Dec(Pred(x_t)) must reconstruct F_{t+1} -- pays for splitting
#     same-appearance adjacent instances that static featrec cannot separate.
#   loss_util (w=0.05): drop-one-slot counterfactual rent on the gate -- marginal-utility
#     replacement for gate_l1 that exempts small-object slots by construction.
# Gate mechanism, curriculum, featrec/loss_ss are unchanged from v26.
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v27.sh
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v27.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v27_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v27.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v27 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v27" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v27_run.out 2>&1"

echo "container: slotcurri_attnmass_v27"
echo "tail -f ${OUT_LOG}"
