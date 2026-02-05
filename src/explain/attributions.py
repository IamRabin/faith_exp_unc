# src/explain/attributions.py
from __future__ import annotations

from typing import Callable, Literal, Optional

import torch
import torch.nn as nn

# Captum explainers
from captum.attr import GradientShap, IntegratedGradients, NoiseTunnel, Saliency


def _target_from_labels(y: torch.Tensor) -> torch.Tensor:
    if y.ndim != 1:
        raise ValueError(f"y must be (B,), got {y.shape}")
    return y


def _make_forward_for_logits(model: nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    # Captum expects forward(inputs) -> logits (B,C)
    def forward(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    return forward


def _build_baseline_batch(
    x: torch.Tensor,
    n_baselines: int,
    mode: Literal["zero", "mean", "noise"],
    dataset_mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Returns baseline batch of shape (n_baselines, C, H, W),
    suitable for Captum SHAP-style methods.

    x: (B, C, H, W)
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x to be (B,C,H,W), got {x.shape}")

    B, C, H, W = x.shape

    if mode == "zero":
        return torch.zeros((n_baselines, C, H, W), device=x.device, dtype=x.dtype)

    if mode == "mean":
        if dataset_mean is None:
            raise ValueError("dataset_mean required for baseline_mode='mean'")
        m = dataset_mean.to(device=x.device, dtype=x.dtype)
        if m.ndim == 1:
            m = m.view(C, 1, 1)
        elif m.ndim == 4 and m.shape[0] == 1:
            m = m.view(C, 1, 1)
        elif m.ndim != 3:
            raise ValueError(f"dataset_mean must be (C,), (C,1,1), or (1,C,1,1), got {m.shape}")
        return m.expand(n_baselines, C, H, W).clone()

    if mode == "noise":
        return torch.randn((n_baselines, C, H, W), device=x.device, dtype=x.dtype) * 0.1

    raise ValueError(f"Unknown baseline mode: {mode}")


def _build_baseline(
    x: torch.Tensor,
    mode: Literal["zero", "mean", "noise"] = "zero",
    *,
    dataset_mean: Optional[torch.Tensor] = None,  # shape (C,) or (1,C,1,1)
) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(x)

    if mode == "mean":
        if dataset_mean is None:
            raise ValueError("dataset_mean must be provided for baseline_mode='mean'")
        m = dataset_mean.to(device=x.device, dtype=x.dtype)
        if m.ndim == 1:  # (C,)
            m = m.view(1, -1, 1, 1)
        return m.expand_as(x).clone()

    if mode == "noise":
        # simple noise baseline; tune stdev if needed
        return torch.randn_like(x) * 0.1

    raise ValueError(f"Unknown baseline mode: {mode}")


@torch.no_grad()
def estimate_dataset_mean_from_loader(loader, *, max_batches: int = 1) -> Optional[torch.Tensor]:
    """
    Quick per-channel mean estimate from a few batches.
    Returns CPU tensor of shape (C,) if inputs are (B,C,H,W), else None.
    """
    means = []
    for bi, (x, _) in enumerate(loader):
        if bi >= max_batches:
            break
        if x.ndim != 4:
            return None
        means.append(x.mean(dim=(0, 2, 3)).detach().cpu())
    if not means:
        return None
    return torch.stack(means, dim=0).mean(dim=0)


def importance_captum_gradient(model, x, y) -> torch.Tensor:
    expl = Saliency(model)
    attr = expl.attribute(x, target=y)
    return attr.abs()


def importance_smoothgrad(
    model,
    x,
    y,
    *,
    n_samples: int = 16,
    stdev: float = 0.1,
):
    expl = Saliency(model)
    nt = NoiseTunnel(expl)
    attr = nt.attribute(
        x,
        nt_type="smoothgrad",
        nt_samples=n_samples,
        stdevs=stdev,
        target=y,
    )
    return attr.abs()


def importance_integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_steps: int = 32,
    baseline_mode: Literal["zero", "mean", "noise"] = "zero",
    dataset_mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Returns |IG| for target class y.
    """
    forward = _make_forward_for_logits(model)
    ig = IntegratedGradients(forward)

    baseline = _build_baseline(x, baseline_mode, dataset_mean=dataset_mean)

    attr = ig.attribute(
        x,
        baselines=baseline,
        target=_target_from_labels(y),
        n_steps=n_steps,
    )
    return attr.abs()


def importance_deepshap(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_baselines: int = 8,
    baseline_mode: Literal["zero", "mean", "noise"] = "zero",
    dataset_mean: Optional[torch.Tensor] = None,
    stdev: float = 0.09,
    n_samples: int = 16,
) -> torch.Tensor:
    """
    SHAP-style attributions using Captum GradientShap (robust).
    Returns |attr| with same shape as x.
    """
    forward = _make_forward_for_logits(model)
    gs = GradientShap(forward)

    baselines = _build_baseline_batch(
        x=x,
        n_baselines=n_baselines,
        mode=baseline_mode,
        dataset_mean=dataset_mean,
    )  # (n_baselines, C, H, W)

    attr = gs.attribute(
        x,
        baselines=baselines,
        target=_target_from_labels(y),
        n_samples=n_samples,
        stdevs=stdev,
    )
    return attr.abs()
