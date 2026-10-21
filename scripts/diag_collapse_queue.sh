#!/usr/bin/env bash
# MOVi-E collapse diagnosis / fix candidates. One GPU per queue, accumulate 2
# so the effective batch matches the 2-GPU runs. LR decay and model.max_steps
# are pinned to the original configs; only trainer.max_steps is shortened.
set -u
cd /mnt/ssd2/hmlee/SlotCurri

IMG=slotcurri:cuda11.6-torch1.13
MOVI_E=configs/slotcurri/movi_e_attnmass_v39lam1gu.yaml
YTVIS=configs/slotcurri/ytvis2021_attnmass_v39lam1gu.yaml

run() {
  local gpu=$1 name=$2 cfg=$3 steps=$4 total=$5
  shift 5
  docker run --rm --name "slotcurri_${name}" --gpus "\"device=${gpu}\"" --shm-size=16g \
    -v /mnt/ssd2/hmlee/SlotCurri:/workspace/SlotCurri \
    -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
    -w /workspace/SlotCurri -e PYTHONPATH=/workspace/SlotCurri \
    "$IMG" bash -lc "python -m slotcurri.train --log-dir logs ${cfg} \
      experiment_name=${name} seed=1 model.diag_log=true \
      trainer.max_steps=${steps} trainer.accumulate_grad_batches=2 \
      trainer.log_every_n_steps=50 trainer.val_check_interval=100000 \
      optimizer.lr_scheduler.decay_steps=${total} model.max_steps=${total} $* \
      >> logs/${name}_run.out 2>&1"
}

queue0() {
  run 0 diag_movi_e_base "$MOVI_E" 500 300000
  run 0 diag_movi_e_sqrtpi "$MOVI_E" 1000 300000 model.attn_mass_curriculum.sqrt_pi=true
  run 0 diag_ytvis_k15d128 "$YTVIS" 1000 100000 globals.NUM_SLOTS=15 globals.SLOT_DIM=128
  run 0 diag_movi_e_k7d64 "$MOVI_E" 1000 300000 globals.NUM_SLOTS=7 globals.SLOT_DIM=64
}

queue1() {
  run 1 diag_movi_e_scwarm2500 "$MOVI_E" 4500 300000 model.slotcontrast_steps=2500
}

case "${1:-all}" in
  0) queue0 ;;
  1) queue1 ;;
  all) queue0 & queue1 & wait ;;
esac
