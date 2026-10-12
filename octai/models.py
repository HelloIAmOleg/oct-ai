from collections import OrderedDict
from torch import nn
from torchvision import models
from torch import load

import timm

import os

from third_party.MedicalNet.model import generate_model

TIMM_MODEL_NAMES = [
    "timm/resnet34.tv_in1k",
    "timm/resnet34.a1_in1k"
]

from .utils import prepare_config

def _load_ae_resnet34_backbone(model, checkpoint_path):
    """Map Shurrab et al.'s AEResnet(res34=True) state dict to torchvision."""
    state_dict = load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Expected a state dict in {checkpoint_path}")

    translated = {}
    for key, value in state_dict.items():
        if key.startswith('fc.'):
            continue
        parts = key.split('.')
        if (len(parts) == 4 and parts[0] == 'conv1' and
                parts[1] == 'block' and parts[2] in ('0', '1')):
            parts = ['conv1' if parts[2] == '0' else 'bn1', *parts[3:]]
        elif len(parts) == 6 and parts[0] in ('conv2_x', 'conv3_x', 'conv4_x', 'conv5_x'):
            stage = int(parts[0][4]) - 1
            block = {'block1': ('conv1', 'bn1'),
                     'block2': ('conv2', 'bn2'),
                     'downsample': ('downsample.0', 'downsample.1')}.get(parts[2])
            if block is None or parts[3] != 'block' or parts[4] not in ('0', '1'):
                raise ValueError(f"Unexpected AEResnet key: {key}")
            parts = [f'layer{stage}', parts[1], block[int(parts[4])], *parts[5:]]
        else:
            raise ValueError(f"Unexpected AEResnet key: {key}")
        translated_key = '.'.join(parts)
        if translated_key in translated:
            raise ValueError(f"Duplicate mapped AEResnet key: {translated_key}")
        translated[translated_key] = value

    model.fc = nn.Identity()
    model.load_state_dict(translated, strict=True)


def get_backbone(backbone = 'resnet34', drop_path_rate = 0,
                 resnet34_checkpoint_path = None, from_scratch = False):
    """Initialization priority: explicit AE checkpoint, scratch, then ImageNet."""
    if not isinstance(from_scratch, bool):
        raise TypeError('from_scratch must be a boolean')
    if from_scratch and backbone not in ('resnet34', *TIMM_MODEL_NAMES):
        raise ValueError('from_scratch is supported for torchvision/timm ResNet34 backbones')
    if resnet34_checkpoint_path and backbone not in ('resnet34', *TIMM_MODEL_NAMES):
        raise ValueError('resnet34_checkpoint_path requires a ResNet34 backbone')
    if backbone == 'resnet34':
        model = models.resnet34(weights = None if (resnet34_checkpoint_path or from_scratch) else models.ResNet34_Weights.IMAGENET1K_V1)
        out_dim = model.fc.in_features
        if resnet34_checkpoint_path:
            _load_ae_resnet34_backbone(model, resnet34_checkpoint_path)
        else:
            model.fc = nn.Identity()
    elif backbone in TIMM_MODEL_NAMES:
        model = timm.create_model(backbone, pretrained = not (from_scratch or bool(resnet34_checkpoint_path)),
                          drop_path_rate = drop_path_rate)
        out_dim = model.fc.in_features
        if resnet34_checkpoint_path:
            _load_ae_resnet34_backbone(model, resnet34_checkpoint_path)
        else:
            model.fc = nn.Identity()
    else:
        pretrained_models_path = os.path.join(os.path.dirname(__file__), "models_pretrained", f"{backbone}.pth")
        if not os.path.exists(pretrained_models_path):
            raise KeyError(f"Unknown backbone type: {backbone}")
        
        state_dict = load(pretrained_models_path, weights_only=True)['state_dict']

        from types import SimpleNamespace

        opt = SimpleNamespace(
            model = 'resnet',
            model_depth = 34,
            input_W = 224,
            input_H = 224,
            input_D = 1,
            resnet_shortcut = 'A',
            no_cuda = False,
            n_seg_classes = 1,
            gpu_id = [0],
            phase = 'train',
            pretrain_path = pretrained_models_path,
            new_layer_names = ['conv_seg']
        )

        _model, _ = generate_model(opt)

        _model.module.conv_seg = nn.Identity()
        _model.load_state_dict(state_dict)

        model = _model.module
        model = nn.Sequential(OrderedDict([
            ('backbone', model),
            ('avgpool', nn.AdaptiveAvgPool3d((1,1,1))),
            ('flatten', nn.Flatten(end_dim = -1))
        ]))

        out_dim = 512
    return model, out_dim

class OctClassifier(nn.Module):

    default_config = {
        "backbone": "resnet34",
        "resnet34_checkpoint_path": None,
        "from_scratch": False,
        "model_type": "normal",
        "drop_path_rate": 0,
        "attention_hid_dim": 512,
        "out_dim": 3
    }

    def __init__(self, config = None):
        super(OctClassifier, self).__init__()

        self.config = prepare_config(self, config)
        
        self.backbone, backbone_out_dim = get_backbone(self.config["backbone"],
                                                       drop_path_rate = self.config["drop_path_rate"],
                                                       resnet34_checkpoint_path = self.config["resnet34_checkpoint_path"],
                                                       from_scratch = self.config["from_scratch"])
        
        if (self.config['model_type'] == "normal") or (self.config['model_type'] == "3D"):
            self.add_module('head', nn.Linear(backbone_out_dim, self.config["out_dim"]))
        elif self.config['model_type'] == "MIL":
            attention_hid_dim = self.config['attention_hid_dim']
            
            attention_fc = nn.Sequential(
                nn.Linear(backbone_out_dim, attention_hid_dim),
                nn.Tanh(),
                nn.Linear(attention_hid_dim, 1)
            )
            self.add_module('attention_fc', attention_fc)
            
            self.add_module('head', nn.Linear(backbone_out_dim, self.config["out_dim"]))
        else:
            raise KeyError(f"Unknown model type: {self.config['model_type']}")
            
    def forward(self, x):
        if self.config['model_type'] != 'MIL':
            embedding = self.backbone(x)
            return self.head(embedding)
    
        batch_size, num_instances, C, H, W = x.shape
        x = x.view(batch_size * num_instances, C, H, W)  # Flatten instances
        
        features = self.backbone(x)  # Extract features
        features = features.view(batch_size, num_instances, -1)  # [batch_size, num_instances, backbone_out_dim]
        
        # Attention Weights Computation
        attention_weights = self.attention_fc(features).squeeze(-1)  # [batch_size, num_instances]
        attention_weights = nn.functional.softmax(attention_weights, dim=1)  # Normalize
        
        # Weighted Sum (Aggregated Representation)
        aggregated_features = (attention_weights.unsqueeze(-1) * features).sum(dim = 1)

        # Patient-Level Classification
        output = self.head(aggregated_features)
        
        
        return output
