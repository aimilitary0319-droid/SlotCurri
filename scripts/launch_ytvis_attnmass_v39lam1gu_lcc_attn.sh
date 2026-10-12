#!/usr/bin/env bash
# Launch YTVIS v39lam1gu_lcc_attn (v39lam1gu + attention L_cc 0.1, contrastive_gate false).
#
# Same as v39lam1g plus:
#   loss_featrec         1.0  gated mix   softmax(α) ⊙ π
#   loss_featrec_ungated 0.5  plain mix   softmax(α)
#
# Usage:
#   GPUS=6,7 bash scripts/launch_ytvis_attnmass_v39lam1gu_lcc_attn.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=0,1}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v39lam1gu_lcc_attn_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v39lam1gu_lcc_attn.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v39lam1gu_lcc_attn on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v39lam1gu_lcc_attn" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v39lam1gu_lcc_attn_run.out 2>&1"

echo "container: slotcurri_attnmass_v39lam1gu_lcc_attn"
echo "tail -f ${OUT_LOG}"
