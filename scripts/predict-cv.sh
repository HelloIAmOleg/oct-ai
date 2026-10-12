#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: bash scripts/predict-cv.sh <1d|2d|mil|3d> <image-root> <output-csv>}"
IMAGE_ROOT="${2:?provide an authorized image root}"
OUTPUT_CSV="${3:?provide output CSV path}"
if [[ "$MODEL" == "1d" ]]; then
  python scripts/predict_1d_cv.py --image-root "$IMAGE_ROOT" --output-csv "$OUTPUT_CSV"
  exit 0
fi
case "$MODEL" in
  2d) RUNS="runs/2026-09-10_NESTED-HPO_2D-40-epochs" ;;
  mil) RUNS="runs/2026-09-10_NESTED-HPO_MIL-40-epochs" ;;
  3d) RUNS="runs/2026-09-17_NESTED-HPO_3D-40-epochs" ;;
  *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac

mkdir -p "$(dirname "$OUTPUT_CSV")"
python scripts/predict_cv.py \
  --base-run-dir "$RUNS" \
  --csv-template 'csv/test_split{}.csv' \
  --image-root "$IMAGE_ROOT" \
  --output-csv "$OUTPUT_CSV" --splits-num 5
