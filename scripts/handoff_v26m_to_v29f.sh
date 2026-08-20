#!/usr/bin/env bash
# After v26m finishes, launch v29f on the same GPU pair.
#
#   v26m (GPUs 6,7) -> v29f (purity coupling ablation: beta=0.7 from step 0)
#
# Usage:
#   bash scripts/handoff_v26m_to_v29f.sh
#   nohup bash scripts/handoff_v26m_to_v29f.sh > logs/handoff_v26m_to_v29f.out 2>&1 &
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
POLL_SECS="${POLL_SECS:-30}"
SRC_NAME="slotcurri_attnmass_v26m"
DST_VER="v29f"
DST_GPUS="${GPUS:-6,7}"

mkdir -p "${ROOT}/logs"

log() { echo "[$(date '+%F %T')] $*"; }

wait_container_gone() {
  local name="$1"
  if ! docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
    log "${name}: not present (already finished / never started)"
    return 0
  fi
  local runlog="${ROOT}/logs/ytvis_attnmass_v26m_run.out"
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

launch_v29f() {
  local cname="slotcurri_attnmass_${DST_VER}"
  local out="${ROOT}/logs/ytvis_attnmass_${DST_VER}_run.out"

  if docker ps --format '{{.Names}}' | grep -qx "${cname}"; then
    log "${cname} already running — skip launch"
    return 0
  fi
  if [[ -d "${ROOT}/logs/_ytvis_attnmass_${DST_VER}/checkpoints" ]] && \
     compgen -G "${ROOT}/logs/_ytvis_attnmass_${DST_VER}/checkpoints/*.ckpt" > /dev/null; then
    log "WARNING: ${DST_VER} checkpoints already exist under logs/_ytvis_attnmass_${DST_VER}/ — aborting to avoid overwrite"
    return 1
  fi
  if [[ -d "${ROOT}/logs/_ytvis_attnmass_${DST_VER}" ]]; then
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    mkdir -p "${ROOT}/logs/_archive"
    mv "${ROOT}/logs/_ytvis_attnmass_${DST_VER}" \
      "${ROOT}/logs/_archive/_ytvis_attnmass_${DST_VER}_pre_handoff_${ts}"
    log "archived stale logs/_ytvis_attnmass_${DST_VER}"
  fi
  if [[ -f "${out}" ]]; then
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    mkdir -p "${ROOT}/logs/_archive"
    mv "${out}" "${ROOT}/logs/_archive/ytvis_attnmass_${DST_VER}_pre_handoff_${ts}.out"
  fi

  log "launching ${DST_VER} on GPUs=${DST_GPUS}"
  GPUS="${DST_GPUS}" bash "${ROOT}/scripts/launch_ytvis_attnmass_${DST_VER}.sh"
  sleep 5
  if docker ps --format '{{.Names}}' | grep -qx "${cname}"; then
    log "${cname} UP — log: ${out}"
  else
    log "ERROR: ${cname} failed to start; check ${out}"
    return 1
  fi
}

log "=== handoff start: v26m -> v29f (GPUs ${DST_GPUS}) ==="
wait_container_gone "${SRC_NAME}"
launch_v29f
log "=== handoff done ==="
