#!/usr/bin/env bash
# Fresh-start v23 (GPUs 0,3) and v24 (GPUs 6,7) together.
#
# Archives any existing log dirs/runouts first so train does not resume/overwrite mid-run.
#
# Usage:
#   bash scripts/launch_ytvis_attnmass_v23v24.sh
#   GPUS_V23=0,3 GPUS_V24=6,7 bash scripts/launch_ytvis_attnmass_v23v24.sh
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
GPUS_V23="${GPUS_V23:-0,3}"
GPUS_V24="${GPUS_V24:-6,7}"
TS="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="${ROOT}/logs/_archive"
mkdir -p "${ARCHIVE}"

archive_one() {
  local ver="$1"
  local dir="${ROOT}/logs/_ytvis_attnmass_${ver}"
  local out="${ROOT}/logs/ytvis_attnmass_${ver}_run.out"
  if docker ps --format '{{.Names}}' | grep -qx "slotcurri_attnmass_${ver}"; then
    echo "ERROR: slotcurri_attnmass_${ver} is still running; stop it first" >&2
    exit 1
  fi
  docker rm -f "slotcurri_attnmass_${ver}" 2>/dev/null || true
  if [[ -d "${dir}" ]]; then
    mv "${dir}" "${ARCHIVE}/_ytvis_attnmass_${ver}_${TS}"
    echo "archived ${dir} -> ${ARCHIVE}/_ytvis_attnmass_${ver}_${TS}"
  fi
  if [[ -f "${out}" ]]; then
    mv "${out}" "${ARCHIVE}/ytvis_attnmass_${ver}_${TS}.out"
    echo "archived ${out}"
  fi
}

archive_one v23
archive_one v24

echo "=== config check ==="
grep -E 'p_anneal:|state_p_mult:|vel_dim:|predictor_dynamics:|p_end_mult:' \
  "${ROOT}/configs/slotcurri/ytvis2021_attnmass_v23.yaml" \
  "${ROOT}/configs/slotcurri/ytvis2021_attnmass_v24.yaml" | grep -v '^#'

echo "=== launch v23 on ${GPUS_V23} ==="
GPUS="${GPUS_V23}" bash "${ROOT}/scripts/launch_ytvis_attnmass_v23.sh"
sleep 3
echo "=== launch v24 on ${GPUS_V24} ==="
GPUS="${GPUS_V24}" bash "${ROOT}/scripts/launch_ytvis_attnmass_v24.sh"
sleep 3
docker ps --filter name=slotcurri_attnmass_v2 --format '{{.Names}} {{.Status}}'
echo "logs:"
echo "  tail -f ${ROOT}/logs/ytvis_attnmass_v23_run.out"
echo "  tail -f ${ROOT}/logs/ytvis_attnmass_v24_run.out"
