import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.trial import TrialState


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for import_path in (REPO_ROOT, SCRIPTS_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from train import main as train_main
from octai.inference import get_pred_df, group_preds_by_id, plot_id_pred_metrics
from octai.utils import read_json


DEFAULT_IMAGE_ROOT = REPO_ROOT / "images"
DEFAULT_SPLITS_DIR = REPO_ROOT / "csv"
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs/tuning/20250312_rot_wd_droppath_tuning_base_config.json"
ARTIFACTS_ROOT = REPO_ROOT / "scripts/tuning/artifacts"
FOLD_COUNT = 5
EXPECTED_PATIENT_COUNT = 48


def parse_args():
    parser = argparse.ArgumentParser(
        description="Leakage-free, per-outer-fold Optuna HPO for OCT classification."
    )
    parser.add_argument("--hpo-run-name", required=True)
    parser.add_argument("--outer-idx", type=int, choices=range(FOLD_COUNT))
    parser.add_argument("--n-trials", type=int, default=40,
                        help="Target total number of completed trials for this outer fold.")
    parser.add_argument("--max-epochs", type=int, default=125)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument(
        "--is-3d",
        action="store_true",
        help="Use the MedicalNet 3D search space (do not tune drop_path_rate).",
    )
    parser.add_argument("--check-splits", action="store_true")
    parser.add_argument("--check-atomic-save", action="store_true")
    args = parser.parse_args()

    if args.n_trials < 1:
        parser.error("--n-trials must be at least 1")
    if args.max_epochs < 1:
        parser.error("--max-epochs must be at least 1")
    if Path(args.hpo_run_name).name != args.hpo_run_name or args.hpo_run_name in {"", ".", ".."}:
        parser.error("--hpo-run-name must be a single directory name")
    if not (args.check_splits or args.check_atomic_save) and args.outer_idx is None:
        parser.error("--outer-idx is required for HPO")

    return args


def fold_csv(splits_dir, fold_idx):
    return Path(splits_dir) / f"test_split{fold_idx}.csv"


def patient_ids(df):
    return set(df["full_id"].unique())


def class_distribution(df):
    labels_per_patient = df.groupby("full_id")["Condition"].nunique()
    assert (labels_per_patient == 1).all(), "A full_id has more than one Condition"
    return df.groupby("full_id")["Condition"].first().value_counts().sort_index().to_dict()


def split_stats(df):
    return {
        "patients": int(df["full_id"].nunique()),
        "measurement_id": int(df["measurement_id"].nunique()),
        "B_scans": int(len(df)),
        "class_distribution": class_distribution(df),
    }


def print_split_stats(name, df):
    stats = split_stats(df)
    print(
        f"  {name}: patients={stats['patients']} "
        f"measurement_id={stats['measurement_id']} B-scans={stats['B_scans']} "
        f"classes={stats['class_distribution']}"
    )


def inner_fold_indices(outer_idx):
    inner_val_idx = (outer_idx + 1) % FOLD_COUNT
    inner_train_indices = [
        idx for idx in range(FOLD_COUNT) if idx not in {outer_idx, inner_val_idx}
    ]
    return inner_train_indices, inner_val_idx


def check_all_splits(splits_dir):
    """Preflight only: this is the sole path that reads every outer-test CSV."""
    folds = [pd.read_csv(fold_csv(splits_dir, idx)) for idx in range(FOLD_COUNT)]
    fold_patient_ids = [patient_ids(df) for df in folds]

    for left in range(FOLD_COUNT):
        for right in range(left + 1, FOLD_COUNT):
            assert not (fold_patient_ids[left] & fold_patient_ids[right]), (
                f"F{left} and F{right} overlap by full_id"
            )

    all_patients = set().union(*fold_patient_ids)
    assert len(all_patients) == EXPECTED_PATIENT_COUNT, (
        f"Expected {EXPECTED_PATIENT_COUNT} patients, found {len(all_patients)}"
    )

    for outer_idx in range(FOLD_COUNT):
        train_indices, val_idx = inner_fold_indices(outer_idx)
        inner_train = pd.concat([folds[idx] for idx in train_indices], ignore_index=True)
        inner_val = folds[val_idx]
        outer_test = folds[outer_idx]

        train_ids = patient_ids(inner_train)
        val_ids = patient_ids(inner_val)
        test_ids = patient_ids(outer_test)
        expected_train_ids = set().union(*(fold_patient_ids[idx] for idx in train_indices))

        assert not (train_ids & val_ids)
        assert not (train_ids & test_ids)
        assert not (val_ids & test_ids)
        assert train_ids | val_ids | test_ids == all_patients
        assert val_ids == fold_patient_ids[val_idx]
        assert train_ids == expected_train_ids

        print(f"outer{outer_idx}: train={train_indices} val=F{val_idx} test=F{outer_idx}")
        print_split_stats("inner_train", inner_train)
        print_split_stats("inner_val", inner_val)
        print_split_stats("outer_test", outer_test)
        print(
            "  intersections: "
            f"train-val={len(train_ids & val_ids)} "
            f"train-test={len(train_ids & test_ids)} "
            f"val-test={len(val_ids & test_ids)}; "
            f"union={len(train_ids | val_ids | test_ids)}"
        )


def prepare_inner_splits(splits_dir, outer_idx, outer_dir):
    """Build inner CSVs without opening this study's outer-test CSV."""
    train_indices, val_idx = inner_fold_indices(outer_idx)
    train_frames = [pd.read_csv(fold_csv(splits_dir, idx)) for idx in train_indices]
    inner_val = pd.read_csv(fold_csv(splits_dir, val_idx))
    inner_train = pd.concat(train_frames, ignore_index=True)

    train_ids = patient_ids(inner_train)
    val_ids = patient_ids(inner_val)
    assert not (train_ids & val_ids), "inner_train and inner_val overlap by full_id"
    assert train_ids == set().union(*(patient_ids(df) for df in train_frames))

    outer_dir.mkdir(parents=True, exist_ok=True)
    inner_train_path = outer_dir / "inner_train.csv"
    inner_val_path = outer_dir / "inner_val.csv"
    inner_train.to_csv(inner_train_path, index=False)
    inner_val.to_csv(inner_val_path, index=False)

    print(
        f"outer{outer_idx} HPO sources: inner_train=F{train_indices}, "
        f"inner_val=F{val_idx}; F{outer_idx} is not read"
    )
    print_split_stats("inner_train", inner_train)
    print_split_stats("inner_val", inner_val)
    return inner_train_path, inner_val_path


def apply_hparams(base_config, params):
    config = json.loads(json.dumps(base_config))
    rot_prob = params["rot_prob"]
    rot_angle = params.get("rot_angle", 0.0) if rot_prob != 0 else 0.0
    config["data"]["dataset"]["transforms"]["rot_angles"] = [-rot_angle, rot_angle]
    config["data"]["dataset"]["transforms"]["p_augs"] = rot_prob
    if "drop_path_rate" in params:
        config["model"]["model"]["drop_path_rate"] = params["drop_path_rate"]
    config["model"]["optimizer"]["weight_decay"] = params["weight_decay"]
    return config


def atomic_pickle_dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as fp:
            temp_path = Path(fp.name)
            pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def atomic_json_dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as fp:
            temp_path = Path(fp.name)
            json.dump(obj, fp, ensure_ascii=False, indent=2)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def check_atomic_save():
    with tempfile.TemporaryDirectory(prefix="octai_atomic_check_") as temp_dir:
        final_path = Path(temp_dir) / "study.pkl"
        atomic_pickle_dump({"completed": 1}, final_path)

        interrupted_path = final_path.parent / f".{final_path.name}.interrupted.tmp"
        with open(interrupted_path, "wb") as fp:
            pickle.dump({"completed": 2}, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.flush()
            os.fsync(fp.fileno())

        with open(final_path, "rb") as fp:
            assert pickle.load(fp) == {"completed": 1}
        interrupted_path.unlink()
        print("Atomic save check passed: pre-replace interruption preserved the prior pickle.")


def completed_trial_count(study):
    return sum(trial.state == TrialState.COMPLETE for trial in study.trials)


def make_objective(args, outer_dir, inner_train_path, inner_val_path, base_config):
    trials_dir = outer_dir / "trials"
    results_dir = outer_dir / "trial_results"
    trials_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        rot_prob = trial.suggest_categorical("rot_prob", [0, 0.5, 1.0])
        if rot_prob != 0:
            trial.suggest_float("rot_angle", 0.5, 15)
        trial.suggest_float("weight_decay", 1e-4, 1, log=True)
        if not args.is_3d:
            trial.suggest_float("drop_path_rate", 0, 0.5)

        trial_config = apply_hparams(base_config, trial.params)
        run_dir = trials_dir / f"trial_{trial.number}_{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        config_path = run_dir / "trial_config.json"
        atomic_json_dump(trial_config, config_path)

        train_args = argparse.Namespace(
            config_fn=str(config_path),
            image_root=str(args.image_root),
            run_dir=str(run_dir),
            train_csv=str(inner_train_path),
            val_csv=str(inner_val_path),
            test_csv=str(inner_val_path),
            gpus=args.gpus,
            max_epochs=args.max_epochs,
            random_state=42,
        )

        succeeded = False
        try:
            train_main(train_args)
            pred_df = get_pred_df(
                str(run_dir), str(inner_val_path), str(args.image_root), v_num=0, use_gpu=True
            )
            id_pred_df = group_preds_by_id(pred_df, id_col="measurement_id")
            metrics_df = plot_id_pred_metrics(
                id_pred_df, label_names=sorted(id_pred_df.real.unique())
            )
            objective_value = float(metrics_df.loc["f1_score", "macro"])

            for class_name, values in metrics_df.items():
                for metric_name, value in values.items():
                    trial.set_user_attr(f"{metric_name}_{class_name}", float(value))

            atomic_json_dump(
                {
                    "trial_number": trial.number,
                    "objective": objective_value,
                    "params": trial.params,
                    "is_3d": args.is_3d,
                    "train_csv": str(inner_train_path),
                    "val_csv": str(inner_val_path),
                    "test_csv": str(inner_val_path),
                    "prediction_csv": str(inner_val_path),
                },
                results_dir / f"trial_{trial.number}.json",
            )
            succeeded = True
            return objective_value
        finally:
            if succeeded:
                try:
                    shutil.rmtree(run_dir)
                except Exception as cleanup_error:
                    print(f"Warning: could not remove successful trial directory {run_dir}: {cleanup_error}")

    return objective


def load_or_create_study(study_path, outer_idx):
    if study_path.exists():
        with open(study_path, "rb") as fp:
            study = pickle.load(fp)
        print(
            f"Loaded {study.study_name}: {len(study.trials)} stored trials, "
            f"{completed_trial_count(study)} completed"
        )
        return study

    return optuna.create_study(
        study_name=f"outer{outer_idx}",
        direction="maximize",
        sampler=TPESampler(seed=42 + outer_idx, n_startup_trials=10),
    )


def save_best_config(study, base_config, best_config_path):
    if completed_trial_count(study) == 0:
        return
    atomic_json_dump(apply_hparams(base_config, study.best_params), best_config_path)


def run_hpo(args):
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {args.image_root}")
    if not args.base_config.is_file():
        raise FileNotFoundError(f"Base config does not exist: {args.base_config}")

    outer_dir = ARTIFACTS_ROOT / args.hpo_run_name / f"outer{args.outer_idx}"
    inner_train_path, inner_val_path = prepare_inner_splits(
        args.splits_dir, args.outer_idx, outer_dir
    )
    base_config = read_json(args.base_config)
    study_path = outer_dir / "study.pkl"
    best_config_path = outer_dir / "best_config.json"
    study = load_or_create_study(study_path, args.outer_idx)

    # Repair or refresh best_config after a process stopped between the two atomic writes.
    save_best_config(study, base_config, best_config_path)
    objective = make_objective(
        args, outer_dir, inner_train_path, inner_val_path, base_config
    )

    while completed_trial_count(study) < args.n_trials:
        study.optimize(objective, n_trials=1)
        atomic_pickle_dump(study, study_path)
        save_best_config(study, base_config, best_config_path)
        print(
            f"Saved outer{args.outer_idx}: completed "
            f"{completed_trial_count(study)}/{args.n_trials}, best={study.best_value:.6f}"
        )

    print(
        f"outer{args.outer_idx} already has the target {completed_trial_count(study)} "
        "completed trials."
    )


def main():
    args = parse_args()
    if args.check_splits:
        check_all_splits(args.splits_dir)
    if args.check_atomic_save:
        check_atomic_save()
    if not (args.check_splits or args.check_atomic_save):
        run_hpo(args)


if __name__ == "__main__":
    main()
