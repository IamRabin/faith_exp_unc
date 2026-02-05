# src/losses/full_loss.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from src.explain.stats import compute_saliency_stats
from src.explain.mask import compute_mask_and_apply
from src.explain.perturbations import PerturbFn, cifar_uniform_linf_noise_like
from src.losses.ce import cross_entropy_from_logits
from src.losses.kl import kl_pq_from_logits


@dataclass
class LossOutput:
    total: torch.Tensor
    components: Dict[str, torch.Tensor]
    x_tilde: torch.Tensor
    mask: torch.Tensor
    logits: Optional[torch.Tensor] = None
    logits_tilde: Optional[torch.Tensor] = None


class ExplanationTrainingLoss(nn.Module):
    """
    Full loss:
      CE + λ_sal * ||mu_g||_1 + λ_unc * ||std_g||_1 + λ_faith * T^2 * KL(p_T(x) || p_T(x_tilde))
    """

    def __init__(
        self,
        *,
        K: int,
        eps_raw: float,
        beta: float,
        tau: float,
        kappa: float,
        lambda_sal: float,
        lambda_unc: float,
        lambda_faith: float,
        T: float = 1.0,
        eps_std: float = 1e-8,
        kl_reduction: str = "batchmean",
        perturbations: Optional[Sequence[PerturbFn]] = None,
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        self.K = int(K)
        self.eps_raw = float(eps_raw)

        self.beta = float(beta)
        self.tau = float(tau)
        self.kappa = float(kappa)

        self.lambda_sal = float(lambda_sal)
        self.lambda_unc = float(lambda_unc)
        self.lambda_faith = float(lambda_faith)

        self.T = float(T)
        self.eps_std = float(eps_std)
        self.kl_reduction = kl_reduction

        self.deterministic = bool(deterministic)

        if self.K <= 0:
            raise ValueError("K must be positive")
        if self.eps_raw < 0:
            raise ValueError("eps_raw must be >= 0")
        if self.T <= 0:
            raise ValueError("T must be > 0")

        # Default: use bounded L_inf noise in pixel space (converted to normalized space internally).
        if perturbations is None:
            self.perturbations: Sequence[PerturbFn] = [
                lambda x: x + cifar_uniform_linf_noise_like(x, eps_raw=self.eps_raw)
            ]
        else:
            if len(perturbations) == 0:
                raise ValueError("perturbations must be non-empty if provided")
            self.perturbations = perturbations

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        create_graph: bool,
    ) -> LossOutput:
        """
        Args:
            model: returns logits (B,C)
            x: (B, ...)
            y: (B,)
            create_graph: True for training (backprop through saliency), False for eval metrics
        """
        if y.ndim != 1:
            raise ValueError(f"y must be shape (B,), got {tuple(y.shape)}")
        logits = model(x)  # (B,C)

        # 1) compute saliency statistics (mu_g, std_g)
        stats = compute_saliency_stats(
            model,
            x,
            y,
            K=self.K,
            create_graph=create_graph,
            perturbations=self.perturbations,
            deterministic=(self.deterministic and self.K == len(self.perturbations)),
            eps=self.eps_std,
        )

        # 2) build mask and masked input
        mask_out = compute_mask_and_apply(
            x,
            stats.mu,
            stats.std,
            beta=self.beta,
            tau=self.tau,
            kappa=self.kappa,
        )
        m = mask_out.m
        x_tilde = mask_out.x_tilde

        # 3) forward on masked input
        logits_tilde = model(x_tilde)

        # 4) compute losses
        pred = cross_entropy_from_logits(logits, y)

        sal = stats.mu.abs().mean()
        unc = stats.std.abs().mean()

        faith_kl = kl_pq_from_logits(logits, logits_tilde, T=self.T, reduction=self.kl_reduction)
        faith = (self.T ** 2) * faith_kl

        total = (
            pred
            + self.lambda_sal * sal
            + self.lambda_unc * unc
            + self.lambda_faith * faith
        )

        comps = {
            "pred": pred.detach(),
            "sal": sal.detach(),
            "unc": unc.detach(),
            "faith": faith.detach(),
            "faith_kl": faith_kl.detach(),
        }

        return LossOutput(
            total=total,
            components=comps,
            x_tilde=x_tilde,
            mask=m,
            logits=logits,
            logits_tilde=logits_tilde,
        )
