# src/losses/kl.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def _check_temperature(T: float) -> None:
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")


def log_softmax_T(logits: torch.Tensor, T: float) -> torch.Tensor:
    """
    Temperature-scaled log-softmax: log softmax(logits / T)
    """
    _check_temperature(T)
    return F.log_softmax(logits / T, dim=-1)


def softmax_T(logits: torch.Tensor, T: float) -> torch.Tensor:
    """
    Temperature-scaled softmax: softmax(logits / T)
    """
    _check_temperature(T)
    return F.softmax(logits / T, dim=-1)


def kl_pq_from_logits(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    *,
    T: float,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """
    KL(p||q) where:
      p = softmax(logits_p / T)
      q = softmax(logits_q / T)

    Uses PyTorch's kl_div with log-probs for q and probs for p:
      kl_div(input=log_q, target=p) = sum p * (log p - log q)

    Args:
        logits_p: (B,C)
        logits_q: (B,C)
        T: temperature
        reduction: "batchmean" (recommended), "mean", "sum", "none"
    """
    log_q = log_softmax_T(logits_q, T)     # log q
    p = softmax_T(logits_p, T)             # p (as target)
    return F.kl_div(log_q, p, reduction=reduction)
