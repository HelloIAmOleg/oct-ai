from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler

from .utils import prepare_config

class WarmupFixedScheduler(LRScheduler):
    def __init__(self, optimizer, num_epochs, higher_lr, lr, last_epoch=-1, verbose=False):
        self.num_epochs = num_epochs
        self.higher_lr = higher_lr
        self.lr = lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.num_epochs:
            return [self.higher_lr for _ in self.base_lrs]
        else:
            return [self.lr for _ in self.base_lrs]

class OptimizerConfigurator:

    optimizer_names = {
        "AdamW": AdamW
    }

    default_config = {
        "type": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "betas": (0.9,0.999),
        "amsgrad": True
    }

    @classmethod
    def configure(cls, params, config = None):
        config = prepare_config(cls, config)

        opt_type = config.pop('type')
        if opt_type not in cls.optimizer_names:
            raise KeyError(f"Unknown optimizer type: {opt_type}")
        
        return cls.optimizer_names[opt_type](params, **config)

class SchedulerConfigurator:

    scheduler_names = {
        "reduce_on_plateau": ReduceLROnPlateau,
        "warmup_fixed": WarmupFixedScheduler,
        None: None
    }

    default_config = {
        "type": None,
        "scheduler_params": {},
        "interval": "epoch",
        "frequency": 1,
        "strict": True,
        "monitor": "val/f1__macro",
    }

    @classmethod
    def configure(cls, optimizer, config = None):
        config = prepare_config(cls, config)

        scheduler_type = config.pop("type")
        if scheduler_type not in cls.scheduler_names:
            raise KeyError(f"Unknown scheduler type: {scheduler_type}")
        elif scheduler_type is None:
            return None

        scheduler_config = {"scheduler": cls.scheduler_names[scheduler_type](optimizer = optimizer, **config.pop("scheduler_params"))}
        scheduler_config.update(config)

        return scheduler_config

