"""Nested Optuna HPO for fast-axis B-scan profiles, NMF, and logistic regression.

Run from the repository root. The outer-test CSV is read only by --check-splits;
ordinary HPO reads the other four outer folds and evaluates on inner_val.
"""

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_PATH = Path(__file__).with_name(
    "2026-09-07_NESTED-HPO_rot_wd_drop-path_tuning_pat-dir-metrics.py"
)
spec = importlib.util.spec_from_file_location("nested_hpo_shared", SHARED_PATH)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)

from octai.data import CLASS_NAMES
from octai.inference import group_preds_by_id, plot_id_pred_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpo-run-name", required=True)
    parser.add_argument("--outer-idx", type=int, choices=range(shared.FOLD_COUNT))
    parser.add_argument("--n-trials", type=int, default=40,
                        help="Maximum number of completed trials per outer fold.")
    parser.add_argument("--nmf-max-iter", type=int, default=2000)
    parser.add_argument("--image-root", type=Path, default=shared.DEFAULT_IMAGE_ROOT)
    parser.add_argument("--splits-dir", type=Path, default=shared.DEFAULT_SPLITS_DIR)
    parser.add_argument("--check-splits", action="store_true")
    parser.add_argument("--check-atomic-save", action="store_true")
    args = parser.parse_args()

    if Path(args.hpo_run_name).name != args.hpo_run_name or args.hpo_run_name in {"", ".", ".."}:
        parser.error("--hpo-run-name must be a single directory name")
    if args.n_trials < 1 or args.nmf_max_iter < 1:
        parser.error("--n-trials and --nmf-max-iter must be positive")
    if not (args.check_splits or args.check_atomic_save) and args.outer_idx is None:
        parser.error("--outer-idx is required for HPO")
    return args


def load_profiles(csv_path, image_root):
    """Compute notebook-equivalent mean/median over image width (axis 2)."""
    df = pd.read_csv(csv_path)
    if df["LOCAL_FILEPATH"].isna().any() or df["LOCAL_FILEPATH"].eq("").any():
        raise ValueError("Image paths are blank; populate LOCAL_FILEPATH relative to the image root")

    mean_profiles = []
    median_profiles = []
    profile_length = None
    for local_path in df["LOCAL_FILEPATH"]:
        image_path = image_root / local_path
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read B-scan: {image_path}")
        if profile_length is None:
            profile_length = image.shape[0]
        elif image.shape[0] != profile_length:
            raise ValueError(f"Unexpected B-scan height in {image_path}")
        mean_profiles.append(image.mean(axis=1))
        median_profiles.append(np.median(image, axis=1))

    return df, {
        "mean": np.stack(mean_profiles),
        "median": np.stack(median_profiles),
    }


def make_config(params, nmf_max_iter):
    estimator = {
        "type": "LogisticRegression",
        "solver": "saga",
        "random_state": 42,
        "penalty": params["lr_penalty"],
        "C": params["lr_C"],
        "max_iter": params["lr_max_iter"],
    }
    if estimator["penalty"] == "elasticnet":
        estimator["l1_ratio"] = params["lr_l1_ratio"]
    return {
        "axis": "fast_axis",
        "profile_aggregation": params["agg_func"],
        "dim_red": {"type": "NMF", "n_components": params["pca_comps"],
                    "max_iter": nmf_max_iter, "random_state": 42},
        "scaler": {"type": "RobustScaler", "after_dim_red": True},
        "estimator": estimator,
    }


def make_pipeline(config):
    return Pipeline([
        ("dim_red", NMF(
            n_components=config["dim_red"]["n_components"],
            max_iter=config["dim_red"]["max_iter"],
            random_state=config["dim_red"]["random_state"],
        )),
        ("scaler", RobustScaler()),
        ("estimator", LogisticRegression(**{
            key: value for key, value in config["estimator"].items() if key != "type"
        })),
    ])


def suggest_params(trial):
    # Match notebook objective_nmf / generate_nmf_params_from_trial exactly.
    trial.suggest_categorical("agg_func", ["mean", "median"])
    trial.suggest_int("pca_comps", 5, 15)
    penalty = trial.suggest_categorical("lr_penalty", ["l2", "l1", "elasticnet"])
    if penalty == "elasticnet":
        trial.suggest_float("lr_l1_ratio", 0, 1)
    trial.suggest_float("lr_C", 1e-5, 1e3, log=True)
    trial.suggest_int("lr_max_iter", 500, 2500)


def make_objective(train_df, train_profiles, val_df, val_profiles,
                   nmf_max_iter, results_dir):
    y_train = train_df["Condition"].to_numpy()

    def objective(trial):
        suggest_params(trial)
        config = make_config(trial.params, nmf_max_iter)
        aggregation = config["profile_aggregation"]
        pipeline = make_pipeline(config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipeline.fit(train_profiles[aggregation], y_train)
            predictions = pipeline.predict(val_profiles[aggregation])

        pred_df = val_df.copy()
        pred_df["Condition_pred"] = predictions
        id_pred_df = group_preds_by_id(pred_df, id_col="measurement_id")
        metrics_df = plot_id_pred_metrics(id_pred_df, label_names=CLASS_NAMES)
        score = float(metrics_df.loc["f1_score", "macro"])

        for class_name, values in metrics_df.items():
            for metric_name, value in values.items():
                trial.set_user_attr(f"{metric_name}_{class_name}", float(value))

        shared.atomic_json_dump(
            {"trial_number": trial.number, "objective": score, "config": config},
            results_dir / f"trial_{trial.number}.json",
        )
        return score

    return objective


def run_hpo(args):
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {args.image_root}")

    outer_dir = shared.ARTIFACTS_ROOT / args.hpo_run_name / f"outer{args.outer_idx}"
    train_csv, val_csv = shared.prepare_inner_splits(
        args.splits_dir, args.outer_idx, outer_dir
    )
    study_path = outer_dir / "study.pkl"
    best_config_path = outer_dir / "best_config.json"
    study = shared.load_or_create_study(study_path, args.outer_idx)
    previous_nmf_max_iter = study.user_attrs.get("nmf_max_iter")
    if previous_nmf_max_iter is None and best_config_path.exists():
        with open(best_config_path, encoding="utf-8") as fp:
            previous_nmf_max_iter = json.load(fp)["dim_red"]["max_iter"]
    if previous_nmf_max_iter is not None and previous_nmf_max_iter != args.nmf_max_iter:
        raise ValueError(
            f"This study uses nmf_max_iter={previous_nmf_max_iter}; "
            f"cannot resume with {args.nmf_max_iter}"
        )
    study.set_user_attr("nmf_max_iter", args.nmf_max_iter)
    if shared.completed_trial_count(study):
        shared.atomic_json_dump(
            make_config(study.best_params, args.nmf_max_iter), best_config_path
        )

    if shared.completed_trial_count(study) >= args.n_trials:
        print(f"outer{args.outer_idx}: stopping criterion already met "
              f"({shared.completed_trial_count(study)} completed trials)")
        return

    train_df, train_profiles = load_profiles(train_csv, args.image_root)
    val_df, val_profiles = load_profiles(val_csv, args.image_root)
    results_dir = outer_dir / "trial_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    objective = make_objective(
        train_df, train_profiles, val_df, val_profiles, args.nmf_max_iter, results_dir
    )

    while shared.completed_trial_count(study) < args.n_trials:
        study.optimize(objective, n_trials=1)
        shared.atomic_pickle_dump(study, study_path)
        shared.atomic_json_dump(
            make_config(study.best_params, args.nmf_max_iter), best_config_path
        )
        print(f"Saved outer{args.outer_idx}: completed "
              f"{shared.completed_trial_count(study)}/{args.n_trials}, "
              f"best={study.best_value:.6f}")


def main():
    args = parse_args()
    if args.check_splits:
        shared.check_all_splits(args.splits_dir)
    if args.check_atomic_save:
        shared.check_atomic_save()
    if not (args.check_splits or args.check_atomic_save):
        run_hpo(args)


if __name__ == "__main__":
    main()
