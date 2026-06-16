"""Xây dựng mô hình transfer learning dựa trên MobileNetV2 pretrained ImageNet."""
import torch.nn as nn
from torchvision import models

import config


def build_model(num_classes: int = None, pretrained: bool = True):
    """MobileNetV2: đóng băng backbone, thay lớp phân loại cuối cho 12 mệnh giá."""
    if num_classes is None:
        num_classes = len(config.CLASS_DIRS)

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v2(weights=weights)

    for p in net.features.parameters():
        p.requires_grad = False

    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return net
