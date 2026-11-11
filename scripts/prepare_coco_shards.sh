#!/usr/bin/env bash
# COCO 2017 -> 256px webdataset shards.
# Train is images only. Val keeps instance masks.
# maxcount 128 -> coco-train-000000..000924 (118287 images),
# coco-validation-000000..000039 (5000 images). Those ranges are what
# configs/slotcurri/coco_attnmass_v39lam1gu.yaml expects.
# The val run prints the max instances per image. num_classes in the yaml
# must be at least that number.
set -euo pipefail

ROOT=/mnt/ssd2/hmlee/SlotCurri
DATA=/mnt/ssd2/hmlee/dataset

cd "${ROOT}/data"
python save_coco.py \
  --split train \
  --maxcount 128 \
  --only-images \
  --download-dir "${DATA}/coco_raw" \
  --out-path "${DATA}/coco"
python save_coco.py \
  --split validation \
  --maxcount 128 \
  --download-dir "${DATA}/coco_raw" \
  --out-path "${DATA}/coco"

echo "train shards: $(ls "${DATA}/coco"/coco-train-*.tar | wc -l)"
echo "val shards:   $(ls "${DATA}/coco"/coco-validation-*.tar | wc -l)"
