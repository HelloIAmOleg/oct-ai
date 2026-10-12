from torch import tensor
from torch.nn import CrossEntropyLoss, ModuleDict
from torch.nn.functional import softmax
from torch.optim import AdamW
import pytorch_lightning as L

from torchmetrics.classification import (MulticlassRecall, MulticlassF1Score,
                                         MulticlassPrecision, Accuracy)
from torchmetrics import MetricCollection

from torchvision.utils import make_grid

from math import sqrt

from .utils import prepare_config
from .transforms import DEFAULT_NORM_MEANS, DEFAULT_NORM_STDS

from .models import OctClassifier
from .data import CLASS_NAMES
from .optimizer import OptimizerConfigurator, SchedulerConfigurator

METRICS_DICT = {
    'recall': MulticlassRecall,
    'f1': MulticlassF1Score,
    'precision': MulticlassPrecision,
}

class OctClassifierModule(L.LightningModule):

    default_config = {
        "model": None,
        "optimizer": None,
        "scheduler": None
    }

    def __init__(self, config):
        super(OctClassifierModule, self).__init__()

        self.config = prepare_config(self, config)

        self.model = OctClassifier(self.config["model"])
        
        self.loss = CrossEntropyLoss()
        
        self.metrics_logged = METRICS_DICT.keys()
        self.label_names = CLASS_NAMES

        self.metrics_dict = ModuleDict({
            f'{tag}_metrics':MetricCollection({
                metric:METRICS_DICT[metric](num_classes = len(self.label_names), average = None)
                for metric in self.metrics_logged}) 
            for tag in ('train', 'val', 'test')})
        
        self.accuracy_dict = ModuleDict({
            f'{tag}_accuracy':Accuracy(task = 'multiclass', num_classes = len(self.label_names))
                for tag in ('train', 'val', 'test')
        })
            
    def forward(self, x):
        return self.model(x)
    
    def update_metrics(self, tag, preds, labels):
        self.metrics_dict[f'{tag}_metrics'].update(preds, labels)
    
    def log_metrics(self, tag):
        metrics_fn_dict = self.metrics_dict[f'{tag}_metrics']
        
        metrics_dict = metrics_fn_dict.compute()
        
        for metric_name, metric_vals in metrics_dict.items():
            for class_metric_val, class_name in zip(metric_vals, self.label_names):
                self.log(f'{tag}/{metric_name}__{class_name}', class_metric_val)
            self.log(f'{tag}/{metric_name}__macro', metric_vals.mean())
        
        metrics_fn_dict.reset()

    def log_images(self, images, tag):
        # Make usable for both MIL and Instance level
        if self.model.config['model_type'] == '3D':
            BS, C, D, H, W = images.shape
            images = images.permute(0,2,1,3,4)
            images = images.view(-1,C,H,W)
        else:
            *(_), C, H, W = images.shape
            images = images.view(-1,C,H,W)

        images_cpu = images.cpu()

        normalization_mean = tensor(
            data=DEFAULT_NORM_MEANS).view(1, 3, 1, 1)
        normalization_std = tensor(
            data=DEFAULT_NORM_STDS).view(1, 3, 1, 1)

        images_denorm = images_cpu * normalization_std + normalization_mean

        images_grid = make_grid(images_denorm, nrow = int(sqrt(len(images_denorm))))
        images_grid_mono = images_grid[0]

        self.logger.experiment.add_image(tag = f"images/{tag}",
                                         img_tensor = images_grid_mono, dataformats = 'HW')
            
        
    def step(self, batch, batch_idx, tag):
        images, labels = batch
        
        logits = self.model(images)
        preds = logits.argmax(dim = 1)
        
        loss = self.loss(logits, labels)
        self.log(f'{tag}/loss', loss, prog_bar = True)
        
        self.update_metrics(tag, preds, labels)
        
        accuracy_fn = self.accuracy_dict[f'{tag}_accuracy']
        accuracy_fn(preds, labels)
        self.log(f'{tag}/accuracy', accuracy_fn, on_step = True, on_epoch = True, prog_bar = True)

        if (self.current_epoch == 0 or tag == 'test') and (batch_idx == 0):
            self.log_images(images = images, tag = tag)

        return loss 
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'test')
    
    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        images, labels = batch
        logits = self.model(images)

        probas = softmax(logits, dim = 1)

        return probas
    
    def on_train_epoch_end(self):
        self.log_metrics('train')
    def on_test_epoch_end(self):
        self.log_metrics('test')
    def on_validation_epoch_end(self):
        self.log_metrics('val')
    
    def configure_optimizers(self):
        optimizer = OptimizerConfigurator.configure(params = self.model.parameters(), config = self.config['optimizer'])
        scheduler_config = SchedulerConfigurator.configure(optimizer = optimizer, config = self.config['scheduler'])

        if scheduler_config is None:
            return dict(optimizer = optimizer)
        
        return dict(optimizer = optimizer, lr_scheduler = scheduler_config)