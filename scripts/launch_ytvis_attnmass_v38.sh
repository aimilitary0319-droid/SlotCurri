#!/usr/bin/env bash
# Launch YTVIS attn-mass v38 (v37 Key-only mix + relation-graph spectral purity).
#
# v38 vs v37:
#   - Feature curriculum is unchanged: ReLU-cosine P X, Key-only, barrier=false,
#     50k cosine mix. Value and recon target stay on original DINO. Eval is raw Keys.
#   - Gate statistic is π = λ1 - max(λ2, 0) on G_s = diag(a) S diag(a) with
#     S = D^{-1/2} R D^{-1/2} from the same ReLU-cosine R on X^bind, not
#     (λ1-λ2)/mass on the D×D feature Gram.
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v38.sh
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v38.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v38_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v38.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v38 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v38" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v38_run.out 2>&1"

echo "container: slotcurri_attnmass_v38"
echo "tail -f ${OUT_LOG}"
