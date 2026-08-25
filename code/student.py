"""Student: torchvision ResNet-50 with adjustable head."""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet50


def build_student(n_classes: int) -> nn.Module:
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def gradcam_target_layer(model: nn.Module):
    """Last conv block of ResNet-50 — recommended for Grad-CAM++."""
    return model.layer4[-1]
