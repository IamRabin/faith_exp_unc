# src/explain/mask.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MaskOutput:
    m: torch.Tensor         # mask in (0,1], same shape as x
    x_tilde: torch.Tensor   # masked input x ⊙ m


def compute_mask(
    mu_g: torch.Tensor,
    std_g: torch.Tensor,
    *,
    beta: float,
    tau: float,
    kappa: float,
) -> torch.Tensor:
    """
    Compute uncertainty-aware mask:
        m = sigmoid(beta * (|mu_g| - tau)) * exp(-kappa * std_g)

    Args:
        mu_g: mean saliency, shape like x
        std_g: saliency std, shape like x (nonnegative)
        beta: sharpness (>0)
        tau: relevance threshold (>=0)
        kappa: uncertainty attenuation (>=0)

    Returns:
        m: mask tensor in (0,1], shape like x
    """
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    if tau < 0:
        raise ValueError(f"tau must be >= 0, got {tau}")
    if kappa < 0:
        raise ValueError(f"kappa must be >= 0, got {kappa}")
    if mu_g.shape != std_g.shape:
        raise ValueError(f"mu_g and std_g must have same shape, got {mu_g.shape} vs {std_g.shape}")

    # relevance gate
    gate = torch.sigmoid(beta * (mu_g.abs() - tau))

    # reliability penalty (in (0,1])
    rel = torch.exp(-kappa * std_g)

    m = gate * rel
    return m


def apply_mask(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    Apply mask via Hadamard product: x_tilde = x ⊙ m
    """
    if x.shape != m.shape:
        raise ValueError(f"x and m must have same shape, got {x.shape} vs {m.shape}")
    return x * m


def compute_mask_and_apply(
    x: torch.Tensor,
    mu_g: torch.Tensor,
    std_g: torch.Tensor,
    *,
    beta: float,
    tau: float,
    kappa: float,
) -> MaskOutput:
    """
    Convenience: compute m and x_tilde.
    """
    std_g = std_g.clamp_min(0.0)
    m = compute_mask(mu_g, std_g, beta=beta, tau=tau, kappa=kappa)
    m = m.to(device=x.device, dtype=x.dtype)
    x_tilde = apply_mask(x, m)
    return MaskOutput(m=m, x_tilde=x_tilde)
