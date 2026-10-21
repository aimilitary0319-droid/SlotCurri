#!/usr/bin/env bash
# YTVIS with the MOVi-E head (15 slots x 128). 32/GPU OOMs, so 16 x accumulate 4
# keeps the effective batch at the original 2 x 32.
set -u
cd /mnt/ssd2/hmlee/SlotCurri
while pgrep -f "bash scripts/diag_collapse_queue.sh" >/dev/null; do sleep 30; done
docker run --rm --name slotcurri_diag_ytvis_k15d128_b16 --gpus '"device=0"' --shm-size=16g \
  -v /mnt/ssd2/hmlee/SlotCurri:/workspace/SlotCurri \
  -v /mnt/ssd2/hmlee/dataset:/workspace/dataset \
  -w /workspace/SlotCurri -e PYTHONPATH=/workspace/SlotCurri \
  slotcurri:cuda11.6-torch1.13 bash -lc "python -m slotcurri.train --log-dir logs \
    configs/slotcurri/ytvis2021_attnmass_v39lam1gu.yaml \
    experiment_name=diag_ytvis_k15d128_b16 seed=1 model.diag_log=true \
    globals.NUM_SLOTS=15 globals.SLOT_DIM=128 globals.BATCH_SIZE_PER_GPU=16 \
    trainer.max_steps=1000 trainer.accumulate_grad_batches=4 \
    trainer.log_every_n_steps=50 trainer.val_check_interval=100000 \
    optimizer.lr_scheduler.decay_steps=100000 model.max_steps=100000 \
    >> logs/diag_ytvis_k15d128_b16_run.out 2>&1"
