#!/usr/bin/env bash
# After v23/v24 finish, launch v25/v22 on the same GPU pairs.
#
#   v23 (GPUs 0,3) -> v25 (v21 + vel injection, no loss_dyn)
#   v24 (GPUs 6,7) -> v22 (v20 ablation: log, p_end/tau 0.2)
#
# Each handoff is independent: v25 starts as soon as v23 exits, without waiting for v24.
#
# Usage:
#   bash scripts/handoff_v23v24_to_v22v25.sh          # foreground (blocks)
#   nohup bash scripts/handoff_v23v24_to_v22v25.sh > logs/handoff_v22v25.out 2>&1 &
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
POLL_SECS="${POLL_SECS:-30}"

mkdir -p "${ROOT}/logs"

# stdout-only: invoke with `nohup ... > logs/handoff_v22v25.out 2>&1 &`
log() { echo "[$(date '+%F %T')] $*"; }

wait_container_gone() {
  local name="$1"
  if ! docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
    log "${name}: not present (already finished / never started)"
    return 0
  fi
  local ver="${name##*_}"  # v23 / v24
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
  # Clear stale empty/partial log dirs from accidental launches (no checkpoints).
  if [[ -d "${ROOT}/logs/_ytvis_attnmass_${ver}" ]]; then
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    mkdir -p "${ROOT}/logs/_archive"
    mv "${ROOT}/logs/_ytvis_attnmass_${ver}" \
      "${ROOT}/logs/_archive/_ytvis_attnmass_${ver}_pre_handoff_${ts}"
    log "archived stale logs/_ytvis_attnmass_${ver}"
  fi
  if [[ -f "${out}" ]]; then
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    mkdir -p "${ROOT}/logs/_archive"
    mv "${out}" "${ROOT}/logs/_archive/ytvis_attnmass_${ver}_pre_handoff_${ts}.out"
  fi

  log "launching ${ver} on GPUs=${gpus}"
  GPUS="${gpus}" bash "${ROOT}/scripts/launch_ytvis_attnmass_${ver}.sh"
  sleep 5
  if docker ps --format '{{.Names}}' | grep -qx "${cname}"; then
    log "${cname} UP — log: ${out}"
  else
    log "ERROR: ${cname} failed to start; check ${out}"
    return 1
  fi
}

log "=== handoff start: v23->v25 (0,3), v24->v22 (6,7) ==="

(
  wait_container_gone slotcurri_attnmass_v23
  launch_one v25 "0,3"
) &
pid25=$!

(
  wait_container_gone slotcurri_attnmass_v24
  launch_one v22 "6,7"
) &
pid22=$!

ec=0
wait "${pid25}" || ec=1
wait "${pid22}" || ec=1
log "=== handoff done (exit=${ec}) ==="
exit "${ec}"
