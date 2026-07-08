import torch.nn as nn
from torchvision import models

import config


def build_model(num_classes: int = None, pretrained: bool = True,
                freeze_backbone: bool = None, unfreeze_last_n: int = None):
    if num_classes is None:
        num_classes = len(config.CLASS_DIRS)
    if freeze_backbone is None:
        freeze_backbone = config.FREEZE_BACKBONE
    if unfreeze_last_n is None:
        unfreeze_last_n = config.UNFREEZE_LAST_N_BLOCKS

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v2(weights=weights)

    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )

    if pretrained and freeze_backbone:
        _set_finetuning(net, unfreeze_last_n)
    return net


def _set_finetuning(net, unfreeze_last_n: int) -> None:
    for p in net.features.parameters():
        p.requires_grad = False
    if unfreeze_last_n > 0:
        for block in net.features[-unfreeze_last_n:]:
            for p in block.parameters():
                p.requires_grad = True


def param_groups(net, head_lr: float, backbone_lr: float):
    head, backbone = [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        (head if name.startswith("classifier") else backbone).append(p)

    groups = [{"params": head, "lr": head_lr}]
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr})
    return groups
