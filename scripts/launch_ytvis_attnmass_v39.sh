#!/usr/bin/env bash
# Launch YTVIS attn-mass v39 (v38 Key-only mix + 8-neighbor graph purity).
#
# v39 vs v38:
#   - Feature curriculum is unchanged: ReLU-cosine P X, Key-only, barrier=false,
#     50k cosine mix. Value and recon target stay on original DINO. Eval is raw Keys.
#   - Gate R is 8-neighbor ReLU-cosine only. π = λ1 - max(λ2, 0) on
#     G_s = diag(a) S diag(a). conf_kind=spectral_graph_n8.
#
# Do not launch onto GPUs 0-3 (v26p) or 4-7 while v38 is running.
#
# Usage:
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v39.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=4,5 (not while v38 holds 4-7)}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v39_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v39.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v39 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v39" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v39_run.out 2>&1"

echo "container: slotcurri_attnmass_v39"
echo "tail -f ${OUT_LOG}"
