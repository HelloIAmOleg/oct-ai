import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import json

print(os.getcwd())

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from octai.module import OctClassifierModule
from octai.data import OctClassifierDataModule
from octai.callbacks import get_callbacks
from octai.utils import read_json, dump_json

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config-fn", "-c", required = True)
    parser.add_argument("--image-root", "-ir", required = True)
    parser.add_argument("--train-csv", required = True)
    parser.add_argument("--val-csv", required = True)
    parser.add_argument("--test-csv", required = True)
    parser.add_argument("--run-dir", required = True)
    parser.add_argument("--gpus", type = int, nargs = '+', default = None)
    parser.add_argument("--max-epochs", type = int, default = 100)
    parser.add_argument("--random-state", type = int, default = None)

    return parser.parse_args()

def main(args):
    if args.random_state is not None:
        pl.seed_everything(args.random_state, workers = True)

    os.makedirs(args.run_dir, exist_ok = True)

    config = read_json(args.config_fn)
    dump_json(config, os.path.join(args.run_dir, 'config.json'))
    
    dump_json(vars(args), os.path.join(args.run_dir, 'console_arguments.json'))

    datamodule = OctClassifierDataModule(image_root = args.image_root, train_csv = args.train_csv, val_csv = args.val_csv, test_csv = args.test_csv,
                                         config = config['data'])

    module = OctClassifierModule(config = config['model'])

    trainer = Trainer(default_root_dir = args.run_dir, accelerator = 'gpu', devices = args.gpus, max_epochs = args.max_epochs, log_every_n_steps = 1,
                       precision = 32, callbacks = get_callbacks(config), deterministic = ('warn' if args.random_state is not None else False))

    trainer.fit(model = module, datamodule = datamodule)
    test_metrics = trainer.test(module, datamodule=datamodule)[0]

    dump_json(test_metrics, os.path.join(args.run_dir, 'metrics.json'))



if __name__ == "__main__":
    args = parse_args()
    main(args)

    