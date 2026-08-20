#!/usr/bin/env bash
# After v22/v24s21 finish, launch the v26 ablations on the same GPU pairs.
#
#   v22    (GPUs 6,7) -> v26m (v26 ablation: confidence removed, beta_final=1.0)
#   v24s21 (GPUs 1,2) -> v26f (v26 ablation: beta fixed at 0.7, coupling removed)
#
# Each handoff is independent: v26m starts as soon as v22 exits, without waiting
# for v24s21 (and vice versa). v26 itself runs separately on GPUs 0,3.
#
# Usage:
#   bash scripts/handoff_v22v24s21_to_v26m_v26f.sh          # foreground (blocks)
#   nohup bash scripts/handoff_v22v24s21_to_v26m_v26f.sh > logs/handoff_v26m_v26f.out 2>&1 &
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
POLL_SECS="${POLL_SECS:-30}"

mkdir -p "${ROOT}/logs"

# stdout-only: invoke with `nohup ... > logs/handoff_v26m_v26f.out 2>&1 &`
log() { echo "[$(date '+%F %T')] $*"; }

wait_container_gone() {
  local name="$1"
  if ! docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
    log "${name}: not present (already finished / never started)"
    return 0
  fi
  local ver="${name##*_}"  # v22 / v24s21
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

log "=== handoff start: v22->v26m (6,7), v24s21->v26f (1,2) ==="

(
  wait_container_gone slotcurri_attnmass_v22
  launch_one v26m "6,7"
) &
pid_m=$!

(
  wait_container_gone slotcurri_attnmass_v24s21
  launch_one v26f "1,2"
) &
pid_f=$!

ec=0
wait "${pid_m}" || ec=1
wait "${pid_f}" || ec=1
log "=== handoff done (exit=${ec}) ==="
exit "${ec}"
