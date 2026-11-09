#!/usr/bin/env bash
# Start MOVi-E xframe_lin40 r2 on GPUs 2,3 after the current MOVi-C container exits.
# Does not stop that MOVi-C run. Does not touch the GPU 0-1 MOVi-E run.
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
WAIT_NAME=slotcurri_movi_c_attnmass_v39lam1gu_xframe_lin40
QUEUE_LOG="${ROOT}/logs/movi_e_attnmass_v39lam1gu_xframe_lin40_r2_queue.out"

mkdir -p "${ROOT}/logs"
exec >> "${QUEUE_LOG}" 2>&1

echo "$(date '+%F %T %Z') waiting for ${WAIT_NAME} to exit"

while docker ps --format '{{.Names}}' | grep -qx "${WAIT_NAME}"; do
  sleep 30
done

echo "$(date '+%F %T %Z') ${WAIT_NAME} is gone; waiting for GPU 2 and 3 to idle"

idle=0
for _ in $(seq 1 60); do
  if python3 - << 'PY'
import subprocess
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
used = {}
for line in out.splitlines():
    idx, mem = [p.strip() for p in line.split(",")]
    used[int(idx)] = int(mem)
raise SystemExit(0 if used.get(2, 99999) < 2000 and used.get(3, 99999) < 2000 else 1)
PY
  then
    idle=1
    break
  fi
  sleep 10
done

if [[ "${idle}" -ne 1 ]]; then
  echo "$(date '+%F %T %Z') GPU 2,3 did not go idle; not launching"
  exit 1
fi

if docker ps --format '{{.Names}}' | grep -qx "slotcurri_movi_e_attnmass_v39lam1gu_xframe_lin40_r2"; then
  echo "$(date '+%F %T %Z') r2 container already running; not launching"
  exit 0
fi

echo "$(date '+%F %T %Z') launching MOVi-E r2 on GPUs 2,3"
GPUS=2,3 bash "${ROOT}/scripts/launch_movi_e_attnmass_v39lam1gu_xframe_lin40_r2.sh"
echo "$(date '+%F %T %Z') launch command finished"
