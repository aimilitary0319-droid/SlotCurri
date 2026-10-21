#!/usr/bin/env bash
# YTVIS with π from per-slot patch-centered logits: does the fix leave YTVIS alone?
set -u
cd /mnt/ssd2/hmlee/SlotCurri
while pgrep -f "bash scripts/diag_collapse_queue_gpu0b.sh" >/dev/null; do sleep 30; done
docker run --rm --name slotcurri_diag_ytvis_picenter --gpus '"device=0"' --shm-size=16g \
  -v /mnt/ssd2/hmlee/SlotCurri:/workspace/SlotCurri \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri -e PYTHONPATH=/workspace/SlotCurri \
  slotcurri:cuda11.6-torch1.13 bash -lc "python -m slotcurri.train --log-dir logs \
    configs/slotcurri/ytvis2021_attnmass_v39lam1gu.yaml \
    experiment_name=diag_ytvis_picenter seed=1 model.diag_log=true \
    model.grouper.pi_logit_center=true \
    trainer.max_steps=1500 trainer.accumulate_grad_batches=2 \
    trainer.log_every_n_steps=50 trainer.val_check_interval=100000 \
    optimizer.lr_scheduler.decay_steps=100000 model.max_steps=100000 \
    >> logs/diag_ytvis_picenter_run.out 2>&1"
