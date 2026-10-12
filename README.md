# OCT classification reproducibility package

Code and model artifacts for four OCT classification approaches: 1D NMF/logistic regression, 2D ResNet34, attention MIL, and 3D MedicalNet. The package includes five patient-level cross-validation splits, selected configurations, fitted models, and hyperparameter search results.

## Contents

- `octai/`: data loading, preprocessing, models, training modules, and inference utilities.
- `scripts/`: training, prediction, cross-validation, and hyperparameter search entry points.
- `csv/`: `train_splitN.csv`, `val_splitN.csv`, and `test_splitN.csv` for folds 0–4, plus `agg_pred_df.csv`.
- `configs/tuning/` and `scripts/tuning/artifacts/`: search configurations, selected parameters for each fold, and fitted 1D pipelines.
- `runs/`: selected 2D, MIL, and 3D checkpoints and final configurations for each fold.
- `metadata/`: software environment and hyperparameter search tables.
- `third_party/MedicalNet/`: MedicalNet source and license; initialization weights are in `octai/models_pretrained/`.

The CSV files use pseudonymous `full_id` and `measurement_id` values and contain 48 patients, 164 measurements, and 26 slice positions per measurement. Their `LOCAL_FILEPATH` column points to images relative to an image root. The image layout is `Condition/full_id__measurement_id/slice_NNN.png`, where `Condition` is `Norm`, `Early VLS`, or `VLS`. Images are stored separately from this code repository.

## Install

Use Python 3.11. Software versions are listed in `requirements.txt` and `metadata/environment.json`. Large model files use Git LFS.

```bash
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/check_manifests.py
```

## Run

From the repository root, supply the directory containing the three `Condition` folders as the image root:

```bash
bash scripts/predict-cv.sh 1d /path/to/images predictions/1d.csv
bash scripts/predict-cv.sh 2d /path/to/images predictions/2d.csv
bash scripts/predict-cv.sh mil /path/to/images predictions/mil.csv
bash scripts/predict-cv.sh 3d /path/to/images predictions/3d.csv
```

To repeat neural training or hyperparameter search, use `bash scripts/run-cv.sh 2d /path/to/images` or `bash scripts/run-hpo.sh 2d /path/to/images`. Replace `2d` with `mil` or `3d`; `run-hpo.sh` also accepts `1d`. Each wrapper processes all five folds and writes new outputs to `reproduction_*` directories.
