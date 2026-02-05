# src/explain/perturbations.py
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


_CIFAR_STD = (0.2470, 0.2435, 0.2616)
_CIFAR_STD_T = torch.tensor(_CIFAR_STD).view(1, 3, 1, 1)


PerturbFn = Callable[[torch.Tensor], torch.Tensor]


def cifar_uniform_linf_noise_like(x: torch.Tensor, eps_raw: float) -> torch.Tensor:
    """Uniform L_inf noise in pixel space eps_raw, converted to normalized space."""
    if eps_raw < 0:
        raise ValueError(f"eps_raw must be >= 0, got {eps_raw}")
    if eps_raw == 0:
        return torch.zeros_like(x)
    eps_norm = eps_raw / _CIFAR_STD_T.to(device=x.device, dtype=x.dtype)
    return (2 * eps_norm) * torch.rand_like(x) - eps_norm


def brightness_contrast_jitter(x: torch.Tensor, b: float = 0.05, c: float = 0.05) -> torch.Tensor:
    """Apply mild brightness/contrast jitter in normalized space (approx)."""
    B = x.size(0)
    br = 1.0 + (2 * b) * torch.rand(B, 1, 1, 1, device=x.device, dtype=x.dtype) - b
    co = 1.0 + (2 * c) * torch.rand(B, 1, 1, 1, device=x.device, dtype=x.dtype) - c
    mean = x.mean(dim=(2, 3), keepdim=True)
    return (x - mean) * co + mean * br


def random_translation(x: torch.Tensor, max_shift: int = 1) -> torch.Tensor:
    """Random translation via grid_sample. x: (B,C,H,W)."""
    if max_shift <= 0:
        return x
    B, C, H, W = x.shape
    dx = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)
    dy = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)

    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device),
        torch.arange(W, device=x.device),
        indexing="ij"
    )
    grid = torch.stack((xx, yy), dim=-1).float()  # (H,W,2)

    grid_x = (grid[..., 0][None] - dx[:, None, None]).float()
    grid_y = (grid[..., 1][None] - dy[:, None, None]).float()

    grid_x = grid_x / (W - 1) * 2 - 1
    grid_y = grid_y / (H - 1) * 2 - 1
    gridn = torch.stack((grid_x, grid_y), dim=-1)

    return F.grid_sample(x, gridn, mode="bilinear", padding_mode="reflection", align_corners=True)


def mild_gaussian_blur(x: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """Very mild blur using a separable approx kernel."""
    if sigma <= 0:
        return x
    # small fixed kernel for CIFAR; you can replace with torchvision if you prefer
    radius = 1
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel1d = kernel1d / kernel1d.sum()
    kx = kernel1d.view(1, 1, 1, -1).repeat(x.size(1), 1, 1, 1)
    ky = kernel1d.view(1, 1, -1, 1).repeat(x.size(1), 1, 1, 1)
    x = F.conv2d(x, kx, padding=(0, radius), groups=x.size(1))
    x = F.conv2d(x, ky, padding=(radius, 0), groups=x.size(1))
    return x
