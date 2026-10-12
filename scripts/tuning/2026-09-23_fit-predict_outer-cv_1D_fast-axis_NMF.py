"""Fit the best 1D NMF configuration per outer fold and predict its test fold."""

import argparse
import importlib.util
import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning


REPO_ROOT = Path(__file__).resolve().parents[2]
HPO_SCRIPT = Path(__file__).with_name("2026-09-23_NESTED-HPO_1D_fast-axis_NMF.py")
spec = importlib.util.spec_from_file_location("nested_hpo_1d", HPO_SCRIPT)
hpo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hpo)

DEFAULT_RUN_NAME = "2026-09-23_NESTED-HPO_1D_fast-axis_NMF"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpo-run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--outer-idx", type=int, choices=range(hpo.shared.FOLD_COUNT),
                        help="Fit and predict one fold for a smoke check; default: all five.")
    parser.add_argument("--image-root", type=Path, default=hpo.shared.DEFAULT_IMAGE_ROOT)
    parser.add_argument("--splits-dir", type=Path, default=hpo.shared.DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", type=Path,
                        help="Default: scripts/tuning/artifacts/<HPO_RUN_NAME>/final_outer_cv")
    args = parser.parse_args()
    if Path(args.hpo_run_name).name != args.hpo_run_name or args.hpo_run_name in {"", ".", ".."}:
        parser.error("--hpo-run-name must be a single directory name")
    if args.output_dir is None:
        args.output_dir = (hpo.shared.ARTIFACTS_ROOT / args.hpo_run_name / "final_outer_cv")
    return args


def atomic_csv_dump(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as fp:
            temp_path = Path(fp.name)
            df.to_csv(fp, index=False)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def load_best_configs(hpo_dir, outer_indices):
    configs = {}
    for outer_idx in outer_indices:
        config_path = hpo_dir / f"outer{outer_idx}/best_config.json"
        with open(config_path, encoding="utf-8") as fp:
            config = json.load(fp)
        if config["axis"] != "fast_axis" or config["dim_red"]["type"] != "NMF":
            raise ValueError(f"Expected fast-axis NMF configuration: {config_path}")
        if config["profile_aggregation"] not in {"mean", "median"}:
            raise ValueError(f"Unexpected profile aggregation: {config_path}")
        configs[outer_idx] = config
    return configs


def fit_predict_fold(outer_idx, config, folds, profiles, output_dir):
    train_indices = [idx for idx in range(hpo.shared.FOLD_COUNT) if idx != outer_idx]
    train_df = pd.concat([folds[idx] for idx in train_indices], ignore_index=True)
    test_df = folds[outer_idx]
    train_ids = hpo.shared.patient_ids(train_df)
    test_ids = hpo.shared.patient_ids(test_df)
    assert not train_ids & test_ids, f"outer{outer_idx}: patient overlap"
    assert len(train_ids | test_ids) == hpo.shared.EXPECTED_PATIENT_COUNT
    if train_df["LOCAL_FILEPATH"].fillna("").ne("").all():
        assert not train_df["LOCAL_FILEPATH"].duplicated().any()

    aggregation = config["profile_aggregation"]
    x_train = np.concatenate([profiles[idx][aggregation] for idx in train_indices])
    y_train = train_df["Condition"].to_numpy()
    x_test = profiles[outer_idx][aggregation]
    pipeline = hpo.make_pipeline(config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)
        predictions = pipeline.predict(x_test)

    pred_df = test_df.copy()
    for column_idx, class_name in enumerate(pipeline.classes_):
        pred_df[f"{class_name}_proba"] = probabilities[:, column_idx]
    pred_df["Condition_pred"] = predictions
    pred_df["outer_idx"] = outer_idx
    assert len(pred_df) == len(test_df)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    fold_dir = output_dir / f"outer{outer_idx}"
    hpo.shared.atomic_pickle_dump(pipeline, fold_dir / "pipeline.pkl")
    atomic_csv_dump(pred_df, fold_dir / "pred_df.csv")
    print(f"outer{outer_idx}: train={len(train_df)} B-scans / {len(train_ids)} patients; "
          f"test={len(test_df)} B-scans / {len(test_ids)} patients; "
          f"config=outer{outer_idx}/best_config.json")
    return pred_df


def main():
    args = parse_args()
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {args.image_root}")
    outer_indices = ([args.outer_idx] if args.outer_idx is not None
                     else list(range(hpo.shared.FOLD_COUNT)))
    hpo_dir = hpo.shared.ARTIFACTS_ROOT / args.hpo_run_name
    configs = load_best_configs(hpo_dir, outer_indices)

    # Every source fold is loaded once. For outer N, its four peers form outer-train.
    folds = []
    profiles = []
    for idx in range(hpo.shared.FOLD_COUNT):
        csv_path = hpo.shared.fold_csv(args.splits_dir, idx)
        df, fold_profiles = hpo.load_profiles(csv_path, args.image_root)
        folds.append(df)
        profiles.append(fold_profiles)

    all_ids = set().union(*(hpo.shared.patient_ids(df) for df in folds))
    assert len(all_ids) == hpo.shared.EXPECTED_PATIENT_COUNT
    for left in range(hpo.shared.FOLD_COUNT):
        for right in range(left + 1, hpo.shared.FOLD_COUNT):
            assert not (hpo.shared.patient_ids(folds[left]) & hpo.shared.patient_ids(folds[right]))

    predictions = [fit_predict_fold(idx, configs[idx], folds, profiles, args.output_dir)
                   for idx in outer_indices]
    agg_pred_df = pd.concat(predictions, ignore_index=True)
    output_path = args.output_dir / (
        "agg_pred_df.csv" if args.outer_idx is None else f"outer{args.outer_idx}_pred_df.csv"
    )
    atomic_csv_dump(agg_pred_df, output_path)
    print(f"Saved {len(agg_pred_df)} B-scan predictions to {output_path}")


if __name__ == "__main__":
    main()
