# src/metrics/utils.py
from __future__ import annotations

import torch


def cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine similarity per sample between two tensors of same shape.
    Returns: (B,)
    """
    a_f = a.flatten(start_dim=1)
    b_f = b.flatten(start_dim=1)
    dot = (a_f * b_f).sum(dim=1)
    na = torch.sqrt((a_f * a_f).sum(dim=1) + eps)
    nb = torch.sqrt((b_f * b_f).sum(dim=1) + eps)
    return dot / (na * nb + eps)


def auc_from_curve(y: torch.Tensor) -> float:
    """
    Compute simple normalized AUC from a curve y of shape (S,)
    using trapezoidal rule over uniform x in [0,1].
    """
    if y.ndim != 1:
        raise ValueError(f"curve must be 1D, got {y.shape}")
    if y.numel() < 2:
        return float("nan")
    # uniform spacing
    return float(torch.trapz(y, dx=1.0 / (y.numel() - 1)).item())
