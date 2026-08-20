#!/usr/bin/env bash
# Launch YTVIS attn-mass v28 (v26 + predictive feature reconstruction ONLY; the L_pred
# ablation of v27) on 2 GPUs.
#
# v28 vs v26: one auxiliary loss, ramped by the existing lambda_t.
#   loss_pred (w=0.2): Dec(Pred(x_t)) must reconstruct F_{t+1} -- pays for splitting
#     same-appearance adjacent instances that static featrec cannot separate.
# No slot_utility (that is v27's second term). Gate mechanism, curriculum, featrec/loss_ss
# unchanged from v26.
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v28.sh
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v28.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v28_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v28.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v28 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v28" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v28_run.out 2>&1"

echo "container: slotcurri_attnmass_v28"
echo "tail -f ${OUT_LOG}"
