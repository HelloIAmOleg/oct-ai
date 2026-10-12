#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: bash scripts/run-cv.sh <2d|mil|3d> <image-root>}"
IMAGE_ROOT="${2:?provide an authorized image root}"
SPLITS="csv"
case "$MODEL" in
  2d)
    SAVED_RUNS="runs/2026-09-10_NESTED-HPO_2D-40-epochs"
    RUNS="runs/reproduction_2d"
    ;;
  mil)
    SAVED_RUNS="runs/2026-09-10_NESTED-HPO_MIL-40-epochs"
    RUNS="runs/reproduction_mil"
    ;;
  3d)
    SAVED_RUNS="runs/2026-09-17_NESTED-HPO_3D-40-epochs"
    RUNS="runs/reproduction_3d"
    ;;
  *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac

for fold in 0 1 2 3 4; do
  python scripts/train.py \
    --config-fn "$SAVED_RUNS/split$fold/config.json" \
    --image-root "$IMAGE_ROOT" \
    --run-dir "$RUNS/split$fold" \
    --train-csv "$SPLITS/train_split$fold.csv" \
    --val-csv "$SPLITS/val_split$fold.csv" \
    --test-csv "$SPLITS/test_split$fold.csv" \
    --max-epochs 125 --gpus 0 --random-state 42
done
