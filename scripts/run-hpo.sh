#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: bash scripts/run-hpo.sh <1d|2d|mil|3d> <image-root>}"
IMAGE_ROOT="${2:?provide an authorized image root}"
SHARED="scripts/tuning/2026-09-07_NESTED-HPO_rot_wd_drop-path_tuning_pat-dir-metrics.py"
case "$MODEL" in
  1d)
    for fold in 0 1 2 3 4; do
      python scripts/tuning/2026-09-23_NESTED-HPO_1D_fast-axis_NMF.py \
        --hpo-run-name reproduction_hpo_1d \
        --outer-idx "$fold" --n-trials 40 --image-root "$IMAGE_ROOT"
    done
    exit 0
    ;;
  2d)
    NAME="reproduction_hpo_2d"
    BASE="configs/tuning/20250312_rot_wd_droppath_tuning_base_config.json"
    EXTRA=()
    ;;
  mil)
    NAME="reproduction_hpo_mil"
    BASE="configs/tuning/20250318_MIL_rot_wd_droppath_tuning_base_config .json"
    EXTRA=()
    ;;
  3d)
    NAME="reproduction_hpo_3d"
    BASE="configs/tuning/20250909_MedicalNet_wd_rot_tuning_base_config.json"
    EXTRA=(--is-3d)
    ;;
  *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac

for fold in 0 1 2 3 4; do
  python "$SHARED" --hpo-run-name "$NAME" --base-config "$BASE" \
    --outer-idx "$fold" --n-trials 40 --max-epochs 125 \
    --gpus 0 --image-root "$IMAGE_ROOT" "${EXTRA[@]}"
done
