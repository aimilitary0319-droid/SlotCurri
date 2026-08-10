#!/usr/bin/env bash
# After v20/v21 finish, launch v23/v24 on the same GPU pairs.
#
#   v20 (GPUs 0,3) -> v23 (velocity-conditioned predictor)
#   v21 (GPUs 6,7) -> v24 (state_p_mult floor)
#
# Each handoff is independent: v23 starts as soon as v20 exits, without waiting for v21.
#
# Usage:
#   bash scripts/handoff_v20v21_to_v23v24.sh          # foreground (blocks)
#   nohup bash scripts/handoff_v20v21_to_v23v24.sh > logs/handoff_v23v24.out 2>&1 &
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
POLL_SECS="${POLL_SECS:-30}"
LOG="${ROOT}/logs/handoff_v23v24.out"

mkdir -p "${ROOT}/logs"

# stdout-only: invoke with `nohup ... > logs/handoff_v23v24.out 2>&1 &` (avoid tee+redirect double lines)
log() { echo "[$(date '+%F %T')] $*"; }

wait_container_gone() {
  local name="$1"
  if ! docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
    log "${name}: not present (already finished / never started)"
    return 0
  fi
  local ver="${name##*_}"  # v20 / v21
  local runlog="${ROOT}/logs/ytvis_attnmass_${ver}_run.out"
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

launch_one() {
  local ver="$1"
  local gpus="$2"
  local cname="slotcurri_attnmass_${ver}"
  local out="${ROOT}/logs/ytvis_attnmass_${ver}_run.out"

  if docker ps --format '{{.Names}}' | grep -qx "${cname}"; then
    log "${cname} already running — skip launch"
    return 0
  fi
  if [[ -d "${ROOT}/logs/_ytvis_attnmass_${ver}/checkpoints" ]] && \
     compgen -G "${ROOT}/logs/_ytvis_attnmass_${ver}/checkpoints/*.ckpt" > /dev/null; then
    log "WARNING: ${ver} checkpoints already exist under logs/_ytvis_attnmass_${ver}/ — aborting to avoid overwrite"
    return 1
  fi

  log "launching ${ver} on GPUs=${gpus}"
  GPUS="${gpus}" bash "${ROOT}/scripts/launch_ytvis_attnmass_${ver}.sh"
  # confirm
  sleep 5
  if docker ps --format '{{.Names}}' | grep -qx "${cname}"; then
    log "${cname} UP — log: ${out}"
  else
    log "ERROR: ${cname} failed to start; check ${out}"
    return 1
  fi
}

log "=== handoff start: v20->v23 (0,3), v21->v24 (6,7) ==="

# Parallel waiters so the first free GPU pair is reused immediately.
(
  wait_container_gone slotcurri_attnmass_v20
  launch_one v23 "0,3"
) &
pid23=$!

(
  wait_container_gone slotcurri_attnmass_v21
  launch_one v24 "6,7"
) &
pid24=$!

ec=0
wait "${pid23}" || ec=1
wait "${pid24}" || ec=1
log "=== handoff done (exit=${ec}) ==="
exit "${ec}"
