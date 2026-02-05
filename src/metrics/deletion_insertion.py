from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics.utils import auc_from_curve


@dataclass
class DeletionInsertionMetrics:
    # Saliency-based (main)
    auc_deletion: float
    auc_insertion: float
    deletion_curve: torch.Tensor   # (S,)
    insertion_curve: torch.Tensor  # (S,)

    # Random-ranking baseline (optional)
    auc_deletion_random: float = float("nan")
    auc_insertion_random: float = float("nan")
    deletion_curve_random: torch.Tensor = torch.empty(0)
    insertion_curve_random: torch.Tensor = torch.empty(0)


@torch.no_grad()
def _confidence_true_class(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Returns p(y|x) for ground-truth class, shape (B,)"""
    p = F.softmax(logits, dim=-1)
    return p.gather(1, y.view(-1, 1)).squeeze(1)


def _rank_features(importance: torch.Tensor) -> torch.Tensor:
    """importance: (B, d_flat) -> indices sorted desc: (B, d_flat)"""
    return torch.argsort(importance, dim=1, descending=True)


def _random_feature_order(B: int, d: int, device: torch.device) -> torch.Tensor:
    """Random permutation per sample: (B, d)"""
    r = torch.rand((B, d), device=device)
    return torch.argsort(r, dim=1, descending=True)


@torch.no_grad()
def _compute_curve_and_auc_from_order(
    *,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    order: torch.Tensor,          # (B, d)
    steps: int,
    baseline_value: float,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """
    Compute deletion/insertion curves for a batch given a fixed feature order.
    Returns (auc_del, auc_ins, del_curve, ins_curve), curves are (steps+1,).
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    model.eval()
    B = x.shape[0]
    x_flat = x.flatten(start_dim=1)
    d = x_flat.shape[1]

    if order.shape != (B, d):
        raise ValueError(f"order must have shape (B,d)=({B},{d}), got {tuple(order.shape)}")

    step_size = max(1, d // steps)

    x_del = x_flat.clone()
    x_ins = torch.full_like(x_flat, fill_value=baseline_value)

    del_curve = []
    ins_curve = []

    # step 0
    del_curve.append(_confidence_true_class(model(x_del.view_as(x)), y).mean())
    ins_curve.append(_confidence_true_class(model(x_ins.view_as(x)), y).mean())

    for s in range(1, steps + 1):
        k = min(s * step_size, d)
        idx = order[:, :k]  # (B,k)

        # deletion: replace top-k with baseline
        x_del.scatter_(1, idx, baseline_value)

        # insertion: copy original top-k into baseline
        vals = x_flat.gather(1, idx)
        x_ins.scatter_(1, idx, vals)

        del_curve.append(_confidence_true_class(model(x_del.view_as(x)), y).mean())
        ins_curve.append(_confidence_true_class(model(x_ins.view_as(x)), y).mean())

    del_curve_t = torch.stack(del_curve)  # (steps+1,)
    ins_curve_t = torch.stack(ins_curve)  # (steps+1,)

    auc_del = auc_from_curve(del_curve_t)
    auc_ins = auc_from_curve(ins_curve_t)
    return auc_del, auc_ins, del_curve_t, ins_curve_t


def evaluate_deletion_insertion(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    importance_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    steps: int = 20,
    baseline_value: float = 0.0,
    max_batches: int = 10,
    compute_random: bool = True,
) -> DeletionInsertionMetrics:
    """
    Evaluate deletion/insertion AUC and curves over a subset of batches.

    Returns:
      - main AUCs/curves averaged across batches
      - optional random-ranking AUCs/curves averaged across batches
    """
    if max_batches is not None and max_batches <= 0:
        raise ValueError(f"max_batches must be positive or None, got {max_batches}")

    model.eval()

    total_del = 0.0
    total_ins = 0.0
    curve_del_sum = None
    curve_ins_sum = None

    total_del_r = 0.0
    total_ins_r = 0.0
    curve_del_r_sum = None
    curve_ins_r_sum = None

    n = 0

    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

        B = x.shape[0]
        d = x.flatten(start_dim=1).shape[1]

        # importance needs gradients
        with torch.set_grad_enabled(True):
            imp = importance_fn(model, x, y)

        imp_flat = imp.flatten(start_dim=1).abs()
        order = _rank_features(imp_flat)

        # main
        with torch.no_grad():
            auc_del, auc_ins, del_curve, ins_curve = _compute_curve_and_auc_from_order(
                model=model,
                x=x,
                y=y,
                order=order,
                steps=steps,
                baseline_value=baseline_value,
            )

        total_del += float(auc_del)
        total_ins += float(auc_ins)

        del_curve = del_curve.detach().cpu()
        ins_curve = ins_curve.detach().cpu()

        if curve_del_sum is None:
            curve_del_sum = del_curve.clone()
            curve_ins_sum = ins_curve.clone()
        else:
            curve_del_sum += del_curve
            curve_ins_sum += ins_curve

        # random baseline
        if compute_random:
            order_r = _random_feature_order(B, d, device=device)
            with torch.no_grad():
                auc_del_r, auc_ins_r, del_curve_r, ins_curve_r = _compute_curve_and_auc_from_order(
                    model=model,
                    x=x,
                    y=y,
                    order=order_r,
                    steps=steps,
                    baseline_value=baseline_value,
                )

            total_del_r += float(auc_del_r)
            total_ins_r += float(auc_ins_r)

            del_curve_r = del_curve_r.detach().cpu()
            ins_curve_r = ins_curve_r.detach().cpu()

            if curve_del_r_sum is None:
                curve_del_r_sum = del_curve_r.clone()
                curve_ins_r_sum = ins_curve_r.clone()
            else:
                curve_del_r_sum += del_curve_r
                curve_ins_r_sum += ins_curve_r

        n += 1

    if n == 0:
        empty = torch.empty(0)
        return DeletionInsertionMetrics(
            auc_deletion=float("nan"),
            auc_insertion=float("nan"),
            deletion_curve=empty,
            insertion_curve=empty,
        )

    out = DeletionInsertionMetrics(
        auc_deletion=total_del / n,
        auc_insertion=total_ins / n,
        deletion_curve=curve_del_sum / n,
        insertion_curve=curve_ins_sum / n,
    )

    if compute_random:
        out.auc_deletion_random = total_del_r / n
        out.auc_insertion_random = total_ins_r / n
        out.deletion_curve_random = curve_del_r_sum / n
        out.insertion_curve_random = curve_ins_r_sum / n

    return out
