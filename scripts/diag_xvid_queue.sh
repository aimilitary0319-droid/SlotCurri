#!/usr/bin/env bash
# feature_curriculum.xvid_energy collapse check. Same 1-GPU / accumulate-2 protocol
# as diag_collapse_queue.sh; controls are diag_movi_e_base (step ~199) and
# movi_c_attnmass_v39lam1gu_s18 (step ~499).
set -u
cd /mnt/ssd2/hmlee/SlotCurri

IMG=slotcurri:cuda11.6-torch1.13
MOVI_E=configs/slotcurri/movi_e_attnmass_v39lam1gu.yaml
MOVI_C18=configs/slotcurri/movi_c_attnmass_v39lam1gu_s18.yaml

run() {
  local gpu=$1 name=$2 cfg=$3 steps=$4 total=$5
  shift 5
  docker run -d --rm --name "slotcurri_${name}" --gpus "\"device=${gpu}\"" --shm-size=16g \
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

case "${1:-xvid}" in
  xvid)
    run 6 diag_movi_e_xvid "$MOVI_E" 1000 300000 model.feature_curriculum.xvid_energy=true
    run 7 diag_movi_c_s18_xvid "$MOVI_C18" 1000 100000 model.feature_curriculum.xvid_energy=true
    ;;
  xframe)
    run 6 diag_movi_e_xframe "$MOVI_E" 1000 300000 model.feature_curriculum.xvid_energy=frame
    run 7 diag_movi_c_s18_xframe "$MOVI_C18" 1000 100000 model.feature_curriculum.xvid_energy=frame
    ;;
  more)
    # auto-resumes diag_movi_e_xvid from its step-1000 checkpoint
    cp logs/_diag_movi_e_xvid/metrics/slotcurri/metrics.csv logs/_diag_movi_e_xvid/metrics_part1.csv
    run 0 diag_movi_e_xvid "$MOVI_E" 3000 300000 model.feature_curriculum.xvid_energy=true
    run 1 diag_movi_e_xframe_s2 "$MOVI_E" 1000 300000 model.feature_curriculum.xvid_energy=frame seed=2
    ;;
esac
