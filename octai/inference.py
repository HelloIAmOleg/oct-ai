import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .module import OctClassifierModule
from .data import OctClassifierDataModule
from .utils import read_json

import torch
from pytorch_lightning import Trainer

from sklearn.metrics import (f1_score, recall_score, precision_score, confusion_matrix,
RocCurveDisplay)

from tbparse import SummaryReader

#-------------PREDICTING_FROM_MODEL_CHECKPOINTS--------------------

def get_ckpt_from_run_dir(run_dir, v_num = 0):
    
    print(run_dir)

    return glob.glob(os.path.join(run_dir, "lightning_logs", f"version_{v_num}", "checkpoints", "epoch*"))[0]

def write_pred_probas_to_df(datamodule, class_names, pred_probas, is_3d = False):
    pred_df = datamodule.predict_dataset.df.copy()
    class_cols = [f"{class_name}_proba" for class_name in class_names]
    pred_df[class_cols] = np.zeros(shape = (len(pred_df), len(class_names)), dtype = np.float32)
    if is_3d:
        measurement_ids = datamodule.predict_dataset.measurement_ids    
        for pat_dir, pred_proba_row in zip(measurement_ids, pred_probas):
            # Q: when some slices are missing, the pred_proba_row must be of different shape, how is it handled?
            pred_df.loc[(pred_df.measurement_id == pat_dir), class_cols] = pred_proba_row 
    else:
        pred_df.loc[:,class_cols] = pred_probas
    
    pred_df["Condition_pred"] = pred_df[class_cols].idxmax(axis = 1).map(lambda s: s.split("_")[0])
    
    return pred_df


def get_pred_df(run_dir, csv_fn, image_root, v_num = 0, use_gpu = True):
    ckpt_path = get_ckpt_from_run_dir(run_dir, v_num = v_num)
    run_config = read_json(os.path.join(run_dir, 'config.json'))

    datamodule = OctClassifierDataModule(image_root = image_root, train_csv = csv_fn, val_csv = csv_fn, test_csv = csv_fn,
                           config = run_config['data'], predict_on = "test")
    module = OctClassifierModule.load_from_checkpoint(ckpt_path, config = run_config['model'])
    
    trainer_kwargs = {}
    if use_gpu:
        trainer_kwargs['accelerator'] = 'gpu'
        trainer_kwargs['devices'] = [0]
        trainer_kwargs['deterministic'] = True
    else:
        trainer_kwargs['accelerator'] = 'cpu'
    
    trainer = Trainer(precision = 32, logger = False, **trainer_kwargs)

    pred_probas = torch.concat(trainer.predict(model = module, datamodule = datamodule, return_predictions = True), dim = 0).detach().cpu().numpy()
    
    print("Pred probas shape:", pred_probas.shape)
    print("Pred df shape:", datamodule.predict_dataset.df.copy().shape)
    
    pred_df = write_pred_probas_to_df(datamodule = datamodule,
                                     class_names = module.label_names, pred_probas = pred_probas,
                                      is_3d = run_config['data'].get('dataset_type', 'normal') in {'MIL', '3D'})
    
    return pred_df

def get_crossval_agg_df(runs_base_dir, csv_template, image_root, v_num = 0, splits_num = 5, use_gpu = True):
    split_dfs = []

    for split in range(splits_num):
        split_dir = os.path.join(runs_base_dir, f'split{split}')

        split_df = get_pred_df(split_dir, csv_template.format(split), image_root,
                               v_num = v_num, use_gpu = use_gpu)
        split_dfs.append(split_df)

    agg_df = pd.concat(split_dfs, axis = 0, ignore_index=True)
    
    return agg_df

#----------------------AGGREGATING_AND_METRICS_CALCULATING-------------------------------

# add option to average probas
DEFAULT_TIE_PRIORITY = ('VLS', 'Early VLS', 'Norm')


def group_preds_by_id(df, id_col = 'full_id', tie_priority=DEFAULT_TIE_PRIORITY):
    """Vote directly over rows; optionally resolve ties by highest-first labels.

    None preserves historical pandas behavior. An explicit priority must contain
    every observed prediction label exactly once (additional labels are allowed).
    Reordering counts changes only equal maxima, never the majority winner.
    """
    if tie_priority is not None:
        tie_priority = list(tie_priority)
        if len(set(tie_priority)) != len(tie_priority):
            raise ValueError('tie_priority must not contain duplicate labels')
        missing = set(df.Condition_pred.dropna().unique()) - set(tie_priority)
        if missing:
            raise ValueError(f'tie_priority missing prediction labels: {sorted(missing)}')

    def vote(group):
        counts = group.Condition_pred.value_counts()
        if tie_priority is not None:
            counts = counts.reindex([label for label in tie_priority if label in counts.index])
        return counts.idxmax()

    preds = df.groupby(id_col).apply(vote, include_groups=False).rename('pred')
    true = df.groupby(id_col).apply(lambda df: df.Condition.iloc[0], include_groups = False).rename('real')
    counts = df.groupby(id_col).apply(lambda df: len(df), include_groups = False).rename('img_count')
    accuracies = df.groupby(id_col).apply(lambda df: df.Condition_pred.value_counts(normalize = True).\
                                                                          get(df.Condition.iloc[0], 0), include_groups=False)\
                                      .rename('accuracy')

    id_pred_df = pd.concat((preds, true, counts, accuracies), axis = 1)
    
    return id_pred_df

def plot_id_pred_metrics(id_pred_df, label_names, add_macro = True,
                         title_prepend = 'Metrics by patient\n', plot = False):
    metrics_to_compute = (f1_score, recall_score, precision_score)

    id_metric_df = pd.DataFrame(np.zeros((3,3)), index = map(lambda f: f.__name__, metrics_to_compute),
                               columns = label_names)

    for metric in metrics_to_compute:
        id_metric_df.loc[metric.__name__] = metric(id_pred_df.real, id_pred_df.pred,
                                                   average = None, labels = label_names)

    if add_macro:
        id_metric_df["macro"] = id_metric_df.mean(axis = 1)

    if plot:
        plt.figure(figsize = (10,7))

        id_metric_df.plot(kind = 'bar', rot = 0, ax = plt.gca())

        plt.ylim(0,1)
        plt.grid(True, lw = 0.2)

        plt.title(title_prepend + f'Macro-F = {id_metric_df.loc["f1_score"].mean():.3f}')

        plt.legend(bbox_to_anchor = (1.,1.02))
        plt.show()

    return id_metric_df

def plot_confusion_matrix(id_pred_df,
                          title = '', ax = None, heatmap_kwargs = {}, label_names = None):
    if ax is None:
        ax = plt.gca()
    
    y_true = id_pred_df.real
    y_pred = id_pred_df.pred
    
    if label_names is None:
        label_names = sorted(y_true.unique())
    
    conf_mat = confusion_matrix(y_true, y_pred, labels = label_names)
    conf_mat_df = pd.DataFrame(conf_mat, columns = label_names, index = label_names)
    
    sns.heatmap(conf_mat_df, cmap = 'Blues', annot = True, fmt = 'd', ax = ax, **heatmap_kwargs)
    
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    
    ax.set_title(title)



# make configurable
def plot_roc_auc(pred_df, class_names, colors, ax = None):
    if ax is None:
        ax = plt.gca()

    y_true_stack = []
    y_score_stack = []

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        y_score = pred_df[f"{class_name}_proba"]
        y_true = (pred_df.Condition == class_name).astype(int)
        
        y_true_stack.append(y_true)
        y_score_stack.append(y_score)

        RocCurveDisplay.from_predictions(
            y_true,
            y_score,
            ax = ax,
            name = class_name,
            color = color
        )
        
    y_true_stack = np.stack(y_true_stack, axis = 1)
    y_score_stack = np.stack(y_score_stack, axis = 1)

    RocCurveDisplay.from_predictions(
        y_true_stack.ravel(),
        y_score_stack.ravel(),
        ax = ax,
        name = r"micro-averaging",
        color = f"C3",
        linestyle = '--',
        plot_chance_level = True
    )
        
    ax.grid(True, lw = 0.2)
        
    ax.set(
        xlabel = 'False Positive Rate',
        ylabel = 'True Positive Rate'
    )

    # ax.xaxis.set_major_formatter(lambda x,pos: f"{x:.1f}" if not x.is_integer() else int(x))
    # ax.yaxis.set_major_formatter(lambda x,pos: f"{x:.1f}" if not x.is_integer() else int(x))

    ax.set_xticks(ticks = np.arange(0,1.2,0.2))
    ax.set_yticks(ticks = np.arange(0,1.2,0.2))

    plt.legend(bbox_to_anchor = (1.01,1.02))


#-----------------------------------TENSORBOARD_READING_AND_VISUALIZING---------------------------------------------------------

def tf_fit_event_from_run_dir(run_dir, v_num = 0):
    return sorted(glob.glob(os.path.join(run_dir, "lightning_logs", f"version_{v_num}", "events*")))[0]

def plot_learning_curve(tb_df, metric_name, early_stop = None, title = '', ax = None):
    """Takes parsed tensorboard dataframe as input and plots learning curve"""
    if ax is None:
        ax = plt.gca()
    
    val_metrics = tb_df[tb_df.tag == f"val/{metric_name}"].sort_values('step').value.reset_index(drop = True)
    train_metrics = tb_df[tb_df.tag == f"train/{metric_name}"].sort_values('step').value.reset_index(drop = True)

    ax.plot(train_metrics, label = 'train')
    ax.plot(val_metrics, label = 'val')
    
    if early_stop is not None:
        ax.axvline(len(val_metrics)-early_stop - 1, c = 'r', ls = '--', lw = 1.5)
    
    ax.set_ylim(top = 1.02)
    ax.grid(True, lw = 0.2)

    ax.set_ylabel(metric_name)
    ax.set_xlabel("Epoch")
    ax.set_title(title)

    ax.legend(loc = 'lower right')

    return ax

def plot_learning_curve_cv(base_run_dir, metric_name = "f1__macro", suptitle = "Кривые обучения на сплитах",
                           early_stop = None, fig = None, splits_num = 5, v_num = 0):
    
    if fig is None:
        fig, axs = plt.subplots(figsize = (5 * 5,4), ncols = splits_num, sharey = "row", squeeze = False)
    else:
        axs = fig.subplots(1, splits_num, sharey = True, squeeze = False)

    axs = axs.flatten()
    
    for split_num, ax in enumerate(axs):
        run_dir = os.path.join(base_run_dir, f"split{split_num}")
        reader = SummaryReader(log_path = tf_fit_event_from_run_dir(run_dir, v_num = v_num))

        tb_df = reader.scalars
        plot_learning_curve(tb_df, metric_name, early_stop = early_stop, title = f"Split {split_num}", ax = ax)

    fig.suptitle(suptitle)
