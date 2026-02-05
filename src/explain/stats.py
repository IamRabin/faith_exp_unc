# src/explain/stats.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import random
import torch

from src.explain.saliency import compute_saliency
from src.explain.perturbations import PerturbFn


@dataclass
class SaliencyStats:
    mu: torch.Tensor
    std: torch.Tensor


def compute_saliency_stats(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    K: int,
    create_graph: bool,
    perturbations: Optional[Sequence[PerturbFn]] = None,
    deterministic: bool = True,
    eps: float = 1e-8,
) -> SaliencyStats:
    """
    Compute (mu_g, std_g) under K perturbations.
    perturbations: list of functions mapping x -> xk (can be additive or non-additive).
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")

    if perturbations is None or len(perturbations) == 0:
        raise ValueError("Provide a non-empty list of perturbations.")

    mean = None
    m2 = None

    # choose which perturbation to apply each iteration
    if deterministic and K == len(perturbations):
        indices = list(range(K))
    else:
        indices = [random.randrange(len(perturbations)) for _ in range(K)]

    for k, idx in enumerate(indices, start=1):
        xk = perturbations[idx](x)

        gk = compute_saliency(model, xk, y, create_graph=create_graph)

        if mean is None:
            mean = gk
            m2 = torch.zeros_like(gk)
        else:
            # Welford online update for mean/variance.
            diff = gk - mean
            mean = mean + diff / k
            diff2 = gk - mean
            m2 = m2 + diff * diff2

    var = m2 / K
    std = torch.sqrt(var + eps)
    return SaliencyStats(mu=mean, std=std)
