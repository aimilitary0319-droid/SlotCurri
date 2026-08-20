#!/usr/bin/env bash
# After slotcontrast s12 finishes, launch the 7-slot SlotContrast baseline
# on the same GPU pair.
#
#   slotcontrast_s12 (GPUs 4,5) -> slotcontrast (NUM_SLOTS=7)
#
# Usage:
#   bash scripts/handoff_s12_to_slotcontrast.sh
#   nohup bash scripts/handoff_s12_to_slotcontrast.sh > logs/handoff_s12_to_slotcontrast.out 2>&1 &
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
POLL_SECS="${POLL_SECS:-30}"
SRC_NAME="slotcurri_slotcontrast_s12"
DST_NAME="slotcurri_slotcontrast"
DST_GPUS="${GPUS:-4,5}"

mkdir -p "${ROOT}/logs"

log() { echo "[$(date '+%F %T')] $*"; }

wait_container_gone() {
  local name="$1"
  if ! docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
    log "${name}: not present (already finished / never started)"
    return 0
  fi
  local runlog="${ROOT}/logs/ytvis_slotcontrast_s12_run.out"
  log "waiting for ${name} to exit (poll ${POLL_SECS}s)..."
  while docker ps --format '{{.Names}}' | grep -qx "${name}"; do
    if [[ -f "${runlog}" ]]; then
      local it
      it=$(tail -c 8000 "${runlog}" | tr '\r' '\n' | grep -oE '[0-9]+it \[' | tail -1 | grep -oE '^[0-9]+' || true)
      log "  ${name} still running (last_it=${it:-?})"
    else
      log "  ${name} still running"
    fi
    sleep "${POLL_SECS}"
  done
  log "${name}: gone"
}

launch_s7() {
  local out="${ROOT}/logs/ytvis_slotcontrast_run.out"

  if docker ps --format '{{.Names}}' | grep -qx "${DST_NAME}"; then
    log "${DST_NAME} already running — skip launch"
    return 0
  fi
  if [[ -d "${ROOT}/logs/_ytvis_slotcontrast/checkpoints" ]] && \
     compgen -G "${ROOT}/logs/_ytvis_slotcontrast/checkpoints/*.ckpt" > /dev/null; then
    log "WARNING: slotcontrast checkpoints already exist under logs/_ytvis_slotcontrast/ — aborting to avoid overwrite"
    return 1
  fi
  if [[ -d "${ROOT}/logs/_ytvis_slotcontrast" ]]; then
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    mkdir -p "${ROOT}/logs/_archive"
    mv "${ROOT}/logs/_ytvis_slotcontrast" \
      "${ROOT}/logs/_archive/_ytvis_slotcontrast_pre_handoff_${ts}"
    log "archived stale logs/_ytvis_slotcontrast"
  fi
  if [[ -f "${out}" ]]; then
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    mkdir -p "${ROOT}/logs/_archive"
    mv "${out}" "${ROOT}/logs/_archive/ytvis_slotcontrast_pre_handoff_${ts}.out"
  fi

  log "launching slotcontrast (7 slots) on GPUs=${DST_GPUS}"
  GPUS="${DST_GPUS}" bash "${ROOT}/scripts/launch_ytvis_slotcontrast.sh"
  sleep 5
  if docker ps --format '{{.Names}}' | grep -qx "${DST_NAME}"; then
    log "${DST_NAME} UP — log: ${out}"
  else
    log "ERROR: ${DST_NAME} failed to start; check ${out}"
    return 1
  fi
}

log "=== handoff start: s12 -> slotcontrast-7 (GPUs ${DST_GPUS}) ==="
wait_container_gone "${SRC_NAME}"
launch_s7
log "=== handoff done ==="
