"""DINOv2 teacher wrapper (frozen) + linear probe head."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

# torch.hub's facebookresearch/dinov2 `main` uses Python 3.10+ syntax (`float | None`)
# and won't import on the cluster's Python 3.8. timm ships the same pretrained DINOv2
# backbones and is import-safe on 3.8, so we load through timm instead.
_TIMM_NAMES = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
}


def load_dinov2(name: str = "dinov2_vits14") -> nn.Module:
    """Load a frozen DINOv2 backbone via timm. Returns CLS-token features (B, D)."""
    timm_name = _TIMM_NAMES.get(name, name)
    model = timm.create_model(timm_name, pretrained=True, num_classes=0,
                              img_size=224, dynamic_img_size=True)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


@torch.no_grad()
def dinov2_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward DINOv2 and return the CLS embedding (B, D)."""
    return model(x)


class LinearHead(nn.Module):
    """Linear probe head over a (B, D) feature vector."""
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
