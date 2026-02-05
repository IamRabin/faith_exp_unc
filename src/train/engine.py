# src/train/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


@dataclass
class TrainStats:
    loss: float
    acc1: float
    components: Dict[str, float]


def train_one_epoch(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_mode: str,
    loss_fn=None,
    log_interval: int = 100,
) -> TrainStats:
    """
    loss_mode:
      - "ce": standard cross-entropy
      - "ours": ExplanationTrainingLoss (expects loss_fn(model,x,y,create_graph=True))
    """
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    # accumulate mean component losses if available
    comp_sums: Dict[str, float] = {}

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if loss_mode == "ce":
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            comps = {}
        elif loss_mode == "ours":
            if loss_fn is None:
                raise ValueError("loss_fn must be provided for loss_mode='ours'")
            out = loss_fn(model, x, y, create_graph=True)
            loss = out.total
            comps = out.components  # dict of detached tensors
            logits = out.logits  # reused computed logits from  loss output
        else:
            raise ValueError(f"Unknown loss_mode: {loss_mode}")

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc1 = accuracy_top1(logits, y)

        total_loss += float(loss.item())
        total_acc += float(acc1)
        n_batches += 1

        for k, v in comps.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + float(v.item())

        if log_interval and step % log_interval == 0:
            pass  # keep silent by default (you can add prints later)

    mean_loss = total_loss / max(n_batches, 1)
    mean_acc = total_acc / max(n_batches, 1)
    mean_comps = {k: v / max(n_batches, 1) for k, v in comp_sums.items()}

    return TrainStats(loss=mean_loss, acc1=mean_acc, components=mean_comps)


@torch.no_grad()
def evaluate_accuracy(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_acc += accuracy_top1(logits, y)
        total_loss += float(loss.item())
        n_batches += 1

    return {
        "acc1": total_acc / max(n_batches, 1),
        "loss": total_loss / max(n_batches, 1),
    }
