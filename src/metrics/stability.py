# src/metrics/stability.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import random

import torch
import torch.nn as nn

from src.explain.saliency import compute_saliency
from src.explain.perturbations import PerturbFn
from src.explain.mask import compute_mask
from src.metrics.utils import cosine_similarity_flat


@dataclass
class StabilityMetrics:
    stability: float
    uncertainty_l1: float
    sparsity: float


def compute_stability_uncertainty_sparsity(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    K: int,
    beta: float,
    tau: float,
    kappa: float,
    perturbations: Sequence[PerturbFn],
    deterministic: bool = True,
    mask_threshold: float = 0.5,
    max_batches: int | None = None,
) -> StabilityMetrics:
    """
    Stability:
      E_{(x,y)} E_{pert}[ cos(g(x,y), g(pert(x),y)) ]

    Uncertainty:
      E_{(x,y)} [ || std_g(x,y) ||_1 ]

    Sparsity:
      E_{(x,y)} [ (1/d) * sum_i 1{ m_i > threshold } ]

    Notes:
      - perturbations is a list of functions mapping x -> xk (same shape).
      - If deterministic=True and K == len(perturbations), each perturbation is applied once.
      - Otherwise, perturbations are sampled with replacement for K draws.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if perturbations is None or len(perturbations) == 0:
        raise ValueError("perturbations must be a non-empty sequence of callables")

    model.eval()

    total_stab = 0.0
    total_unc = 0.0
    total_sparse = 0.0
    n_samples = 0

    # Choose perturbation indices for K draws (per batch we reuse this policy)
    if deterministic and K == len(perturbations):
        indices = list(range(K))
    else:
        indices = [random.randrange(len(perturbations)) for _ in range(K)]

    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bs = x.size(0)

        # We need grads w.r.t. input for saliency, so enable grads for this block.
        with torch.set_grad_enabled(True):
            # Base saliency g(x,y)
            g = compute_saliency(model, x, y, create_graph=False)

            # We'll compute:
            # - stability: mean cosine similarity between g and gk
            # - stats (mu,std) of gk using Welford online algorithm
            mean = None
            m2 = None
            stab_sum = 0.0

            for t, idx in enumerate(indices, start=1):
                xk = perturbations[idx](x)

                gk = compute_saliency(model, xk, y, create_graph=False)

                # Stability contribution (scalar)
                stab_sum += float(cosine_similarity_flat(g, gk).mean().item())

                # Welford update for mean/std of gk (tensor stats)
                if mean is None:
                    mean = gk
                    m2 = torch.zeros_like(gk)
                else:
                    diff = gk - mean
                    mean = mean + diff / t
                    diff2 = gk - mean
                    m2 = m2 + diff * diff2

            stab_k = stab_sum / K

            # std from Welford (population variance m2/K)
            var = m2 / K
            std = torch.sqrt(var + 1e-8)

            # compute uncertainty-aware mask
            m = compute_mask(mean, std, beta=beta, tau=tau, kappa=kappa)

        # ---- Metrics computed WITHOUT keeping any autograd graphs ----
        std_det = std.detach()
        m_det = m.detach()

        # Uncertainty: true L1 norm per sample, averaged over batch
        unc_l1 = float(std_det.abs().flatten(start_dim=1).sum(dim=1).mean().item())

        # Sparsity: fraction of elements retained (m > threshold)
        sparse = float((m_det > mask_threshold).float().flatten(start_dim=1).mean(dim=1).mean().item())

        total_stab += stab_k * bs
        total_unc += unc_l1 * bs
        total_sparse += sparse * bs
        n_samples += bs

    return StabilityMetrics(
        stability=total_stab / max(n_samples, 1),
        uncertainty_l1=total_unc / max(n_samples, 1),
        sparsity=total_sparse / max(n_samples, 1),
    )
