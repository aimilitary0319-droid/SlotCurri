#!/usr/bin/env bash
# DAVIS 2017 unsupervised 480p -> webdataset shards.
# 60 train / 30 val, maxcount 2 -> davis-train-000000..000029,
# davis-validation-000000..000014. Those ranges are what
# configs/slotcurri/davis_attnmass_v39lam1gu.yaml expects.
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
DATA=/mnt/ssd2/hmlee/dataset

cd "${ROOT}/data"
python save_davis.py \
  --maxcount 2 \
  --root "${DATA}/davis_raw" \
  --out_path "${DATA}/davis"

echo "train shards: $(ls "${DATA}/davis"/davis-train-*.tar | wc -l)"
echo "val shards:   $(ls "${DATA}/davis"/davis-validation-*.tar | wc -l)"
