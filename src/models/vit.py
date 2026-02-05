# src/models/vit.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    # torchvision >= 0.13 typically
    from torchvision.models.vision_transformer import VisionTransformer
except Exception as e:
    raise ImportError(
        "torchvision VisionTransformer not available. "
        "Please upgrade torchvision (and torch) to a version that includes "
        "torchvision.models.vision_transformer.VisionTransformer."
    ) from e


@dataclass(frozen=True)
class ViTCifarConfig:
    image_size: int = 32
    patch_size: int = 4          # 32/4 = 8 -> 8x8=64 tokens (+ cls)
    num_layers: int = 6
    num_heads: int = 6
    hidden_dim: int = 384
    mlp_dim: int = 1536
    dropout: float = 0.0
    attention_dropout: float = 0.0
    representation_size: Optional[int] = None  # keep None for standard head


def build_vit_cifar(
    num_classes: int,
    *,
    cfg: ViTCifarConfig = ViTCifarConfig(),
) -> nn.Module:
    """
    Vision Transformer adapted for CIFAR (32x32) using torchvision's implementation.

    Notes:
    - patch_size should divide image_size.
    - For CIFAR-10, patch_size=4 is a good default.
    """
    if cfg.image_size % cfg.patch_size != 0:
        raise ValueError(
            f"image_size ({cfg.image_size}) must be divisible by patch_size ({cfg.patch_size})"
        )

    model = VisionTransformer(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        hidden_dim=cfg.hidden_dim,
        mlp_dim=cfg.mlp_dim,
        dropout=cfg.dropout,
        attention_dropout=cfg.attention_dropout,
        num_classes=num_classes,
        representation_size=cfg.representation_size,
        norm_layer=nn.LayerNorm,
    )
    return model
