# src/losses/ce.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Standard cross-entropy loss from logits.

    Args:
        logits: (B,C)
        y: (B,)
    Returns:
        scalar loss
    """
    return F.cross_entropy(logits, y)
