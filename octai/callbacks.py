from .utils import prepare_config
from pytorch_lightning.callbacks import (ModelCheckpoint, EarlyStopping, LearningRateMonitor,
                                         Callback)

class FreezeUnfreezeCallback(Callback):
    def __init__(self, final_layer_name: str, num_epochs: int):
        """
        :param final_layer_name: Name of the final layer to remain trainable
        :param num_epochs: Number of epochs to keep other layers frozen. If -1, never unfreeze.
        """
        self.final_layer_name = final_layer_name
        self.num_epochs = num_epochs

    def on_fit_start(self, trainer, pl_module):
        # Freeze all parameters except the final layer
        for name, param in pl_module.named_parameters():
            param.requires_grad = self.final_layer_name in name

    def on_train_epoch_start(self, trainer, pl_module):
        if self.num_epochs == -1:
            return

        current_epoch = trainer.current_epoch
        if current_epoch == self.num_epochs:
            # Unfreeze all layers
            for param in pl_module.parameters():
                param.requires_grad = True
            print(f"Epoch {current_epoch}: All layers are now unfrozen.")

CALLBACKS = {
    "model_checkpoint": ModelCheckpoint,
    "early_stop": EarlyStopping,
    "lr_monitor": LearningRateMonitor,
    "freeze_unfreeze": FreezeUnfreezeCallback
}

def callback_from_config(config):
    type = config.pop("type")

    if type not in CALLBACKS:
        raise KeyError(f"Unknown callback type: {type}")

    return CALLBACKS[type](**config)

def get_callbacks(config):
    callback_configs = config.get("callbacks", [])

    return [callback_from_config(config) for config in callback_configs]