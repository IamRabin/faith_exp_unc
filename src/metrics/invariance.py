# src/metrics/invariance.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from src.explain.stats import compute_saliency_stats
from src.explain.mask import compute_mask
from src.explain.perturbations import PerturbFn
from src.losses.kl import kl_pq_from_logits


@dataclass
class InvarianceMetrics:
    delta_inv: float
    delta_sens: float


def compute_invariance_metrics(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    K: int,
    beta: float,
    tau: float,
    kappa: float,
    T: float,
    perturbations: Sequence[PerturbFn],
    deterministic: bool = True,
    max_batches: int | None = None,
) -> InvarianceMetrics:
    """
    Computes:
      Delta_inv  = E[ KL(p_T(x) || p_T(x_tilde)) ]
      Delta_sens = E[ KL(p_T(x) || p_T(x_delete)) ] where x_delete = x ⊙ (1-m)

    Notes:
      - Saliency stats require gradients w.r.t input => we keep grad enabled there.
      - We do NOT need create_graph (no second-order).
      - perturbations: list of callables x -> xk (additive or transform).
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if perturbations is None or len(perturbations) == 0:
        raise ValueError("perturbations must be a non-empty sequence")

    model.eval()

    total_inv = 0.0
    total_sens = 0.0
    n = 0

    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # compute mu/std with gradients enabled
        with torch.set_grad_enabled(True):
            stats = compute_saliency_stats(
                model,
                x,
                y,
                K=K,
                create_graph=False,
                perturbations=perturbations,
                deterministic=deterministic,
            )
            m = compute_mask(stats.mu, stats.std, beta=beta, tau=tau, kappa=kappa)

        m = m.to(dtype=x.dtype, device=x.device).clamp(0, 1)

        # masked / deleted inputs
        x_tilde = x * m
        x_del = x * (1.0 - m)

        with torch.no_grad():
            logits = model(x)
            logits_tilde = model(x_tilde)
            logits_del = model(x_del)

            inv = kl_pq_from_logits(logits, logits_tilde, T=T, reduction="batchmean")
            sens = kl_pq_from_logits(logits, logits_del, T=T, reduction="batchmean")

        bs = x.shape[0]
        total_inv += float(inv.item()) * bs
        total_sens += float(sens.item()) * bs
        n += bs

    return InvarianceMetrics(
        delta_inv=total_inv / max(n, 1),
        delta_sens=total_sens / max(n, 1),
    )
