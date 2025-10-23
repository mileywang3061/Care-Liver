import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights,ResNet18_Weights
import torchvision.models as models
from collections import OrderedDict
from arg_config import parse_args

config = parse_args()   


## Renest18 network, modified for classification task
def classification_resnet_model(num_modalities):
    resnet_model = models.resnet18(weights=None)
    resnet_model.conv1 = nn.Conv2d(num_modalities, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet_model.maxpool = nn.Identity()
    resnet_model.layer3 = nn.Identity()
    resnet_model.layer4 = nn.Identity()
    with torch.no_grad():
        dummy_input = torch.randn(1, num_modalities, 16, 16)  # 输入尺寸根据实际任务调整
        x = resnet_model.conv1(dummy_input)
        x = resnet_model.bn1(x)
        x = resnet_model.relu(x)
        x = resnet_model.maxpool(x)
        x = resnet_model.layer1(x)
        x = resnet_model.layer2(x)
        x = resnet_model.layer3(x)
        x = resnet_model.layer4(x)
        x = resnet_model.avgpool(x)
        in_features = x.view(x.size(0), -1).shape[1]
        print('dynamic in_features:', in_features, flush=True)

    resnet_model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, config.num_classes)
    )

    for param in resnet_model.fc.parameters():
        param.requires_grad = True

    return resnet_model



__all__ = ["classification_resnet_model"]