# src/models/resnet.py
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


from src.models.vit import build_vit_cifar, ViTCifarConfig


class ResNetForCIFAR(nn.Module):
    """
    ResNet adapted for CIFAR (32x32):
      - 3x3 conv stem, stride=1, padding=1
      - remove initial maxpool
      - replace final FC to match num_classes
    Returns logits of shape (B, C).
    """

    def __init__(self, num_classes: int = 10, depth: str = "resnet18") -> None:
        super().__init__()
        if depth == "resnet18":
            base = tvm.resnet18(weights=None)
        elif depth == "resnet34":
            base = tvm.resnet34(weights=None)
        else:
            raise ValueError(f"Unsupported depth={depth}. Use 'resnet18' or 'resnet34'.")

        # CIFAR stem: 3x3 conv, stride 1; remove maxpool
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        # Replace classifier
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, num_classes)

        self.net = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        """
        Convenience helper (not used in training): temperature-scaled softmax.
        """
        logits = self.forward(x)
        return torch.softmax(logits / T, dim=-1)




def build_model(name: str, num_classes: int) -> nn.Module:
    """
    Model factory.
    """
    name = name.lower().strip()

    if name in {"resnet18", "cifar_resnet18"}:
        return ResNetForCIFAR(num_classes=num_classes, depth="resnet18")
    if name in {"resnet34", "cifar_resnet34"}:
        return ResNetForCIFAR(num_classes=num_classes, depth="resnet34")

    # --- ViT for CIFAR-10/100 (32x32) ---
    if name in {"vit", "vit_cifar", "cifar_vit", "vit_small", "vit-s"}:
        cfg = ViTCifarConfig(
            image_size=32,
            patch_size=4,
            num_layers=6,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=1536,
            dropout=0.0,
            attention_dropout=0.0,
        )
        return build_vit_cifar(num_classes=num_classes, cfg=cfg)

    raise ValueError(f"Unknown model name: {name}")