#!/usr/bin/env bash
# Launch MOVi-C attn-mass v37 (v36 Key-only mix without the N-cut region + spectral
# slot purity). Same method as ytvis2021_attnmass_v37.yaml; backbone / pool follow
# movi_c_attnmass_v29.yaml (ViT-S/14, 11 slots, 336px, ignore_background).
#
# v37 vs v36:
#   - Feature curriculum is the same ReLU-cosine P X, Key-only, 50k cosine mix.
#     barrier=false skips the 2-way Fiedler/median cut (one global region).
#     Value and the reconstruction target stay on original DINO. Eval is raw Keys.
#   - Gate statistic is spectral π = (λ1-λ2)/(Σ A + eps) on C = Z^T diag(a^2) Z
#     with Z = X^bind projected to 64-d (spectral_proj_dim=64). Raw A, no gamma,
#     no 1/K map. Set spectral_proj_dim=0 for full D.
#
# Usage:
#   bash scripts/launch_movi_c_attnmass_v37.sh
#   GPUS=2,3 bash scripts/launch_movi_c_attnmass_v37.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:-2,3}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/movi_c_attnmass_v37_run.out"
CFG=configs/slotcurri/movi_c_attnmass_v37.yaml

mkdir -p "${ROOT}/logs"
echo "Launching MOVi-C v37 on GPUs=${GPUS} -> ${OUT_LOG}"

# Quote device list so docker sets DeviceIDs (not Count=-1 / all GPUs).
docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_movi_c_attnmass_v37" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/movi_c_attnmass_v37_run.out 2>&1"

echo "container: slotcurri_movi_c_attnmass_v37"
echo "tail -f ${OUT_LOG}"
