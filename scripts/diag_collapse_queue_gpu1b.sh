#!/usr/bin/env bash
# After the ungated ln19 run: π from per-slot patch-centered logits (gate on,
# standard Slot Attention), then the ungated nofc control.
set -u
cd /mnt/ssd2/hmlee/SlotCurri
while docker ps --format '{{.Names}}' | grep -q '^slotcurri_diag_movi_e_ungated_ln19$'; do sleep 30; done
IMG=slotcurri:cuda11.6-torch1.13
MOVI_E=configs/slotcurri/movi_e_attnmass_v39lam1gu.yaml
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
run 1 diag_movi_e_picenter "$MOVI_E" 1500 300000 model.grouper.pi_logit_center=true
run 1 diag_movi_e_ungated_nofc "$MOVI_E" 800 300000 model.slotcontrast_steps=1000000 \
  model.feature_curriculum.enabled=false
