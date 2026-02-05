# src/train/optim.py
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class OptimConfig:
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True


def build_sgd(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov,
    )


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    max_epochs: int,
    min_lr: float = 0.0,
) -> torch.optim.lr_scheduler._LRScheduler:
    # CosineAnnealingLR expects T_max in epochs
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=min_lr)


def build_multistep_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    milestones: list[int],
    gamma: float = 0.1,
) -> torch.optim.lr_scheduler._LRScheduler:
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
