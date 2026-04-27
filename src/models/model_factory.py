"""Factory for pretrained ResNet50 and EfficientNet-B0 classifiers."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights


def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a torchvision model with optional ImageNet weights and a new head for ``num_classes``.

    ``model_name`` is case-insensitive. Supported: ``\"resnet50\"``, ``\"efficientnet_b0\"``.
    """
    key = model_name.strip().lower()

    if key == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if key == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported model_name {model_name!r}. "
        "Expected 'resnet50' or 'efficientnet_b0'."
    )
