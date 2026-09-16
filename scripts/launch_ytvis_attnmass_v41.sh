#!/usr/bin/env bash
# Launch YTVIS attn-mass v41 (SlotContrast + learned usage-head gate z +
# coupled slot-entropy / v37 Gram C_s impurity, T=50k,
# w_ent 0.3→0.1, w_imp 0.1→0.3).
# No ncut curriculum. L_imp C_s uses the same 2D subspace top-2 as v37,
# not batched 64×64 eigh. proj_dim=64 is JL width.
#
# Usage:
#   GPUS=4,5 bash scripts/launch_ytvis_attnmass_v41.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS="${GPUS:?set GPUS, e.g. GPUS=4,5}"
IMAGE="${IMAGE:-slotcurri:cuda11.6-torch1.13}"
OUT_LOG="${ROOT}/logs/ytvis_attnmass_v41_run.out"
CFG=configs/slotcurri/ytvis2021_attnmass_v41.yaml

mkdir -p "${ROOT}/logs"
echo "Launching v41 on GPUs=${GPUS} -> ${OUT_LOG}"

docker run -d --rm --gpus "\"device=${GPUS}\"" --shm-size=16g \
  --name "slotcurri_attnmass_v41" \
  -v "${ROOT}:/workspace/SlotCurri" \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e PYTHONPATH=/workspace/SlotCurri \
  "${IMAGE}" \
  bash -lc "python -m slotcurri.train --run-eval-after-training --log-dir logs ${CFG} trainer.log_every_n_steps=100 > logs/ytvis_attnmass_v41_run.out 2>&1"

echo "container: slotcurri_attnmass_v41"
echo "tail -f ${OUT_LOG}"
