#!/usr/bin/env bash
# Extend the π-center run to 3000 steps (auto-resume from its last checkpoint).
set -u
cd /mnt/ssd2/hmlee/SlotCurri
while docker ps --format '{{.Names}}' | grep -q '^slotcurri_diag_movi_e_picenter$'; do sleep 30; done
cp logs/_diag_movi_e_picenter/metrics/slotcurri/metrics.csv logs/_diag_movi_e_picenter/metrics_part1.csv
docker run --rm --name slotcurri_diag_movi_e_picenter --gpus '"device=1"' --shm-size=16g \
  -v /mnt/ssd2/hmlee/SlotCurri:/workspace/SlotCurri \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri -e PYTHONPATH=/workspace/SlotCurri \
  slotcurri:cuda11.6-torch1.13 bash -lc "python -m slotcurri.train --log-dir logs \
    configs/slotcurri/movi_e_attnmass_v39lam1gu.yaml \
    experiment_name=diag_movi_e_picenter seed=1 model.diag_log=true \
    model.grouper.pi_logit_center=true \
    trainer.max_steps=3000 trainer.accumulate_grad_batches=2 \
    trainer.log_every_n_steps=50 trainer.val_check_interval=100000 \
    optimizer.lr_scheduler.decay_steps=300000 model.max_steps=300000 \
    >> logs/diag_movi_e_picenter_run.out 2>&1"
