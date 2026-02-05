"""
scripts/save_cifar10_perturbations.py

Save CIFAR-10 examples where ALL 4 perturbations are applied to the SAME image:
[orig | linf-noise | photometric | translation | blur]

Key properties:
- Uses the perturbations from src/explain/perturbations.py
- Operates in NORMALIZED space (same as training)
- Saves a 1x5 strip per image
- Upscales with NEAREST neighbor (crisp, no blur introduced by resizing)
- Includes a debug assert understanding that "orig" is not modified in-place

 eps_raw:
  2/255 = 0.0078431373
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid

from src.explain.perturbations import (
    cifar_uniform_linf_noise_like,
    brightness_contrast_jitter,
    random_translation,
    mild_gaussian_blur,
)

# Must match your training pipeline
_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


def unnormalize_cifar(x: torch.Tensor) -> torch.Tensor:
    """x: (B,3,H,W) normalized -> returns approx [0,1]."""
    mean = torch.tensor(_CIFAR_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_CIFAR_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return x * std + mean


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def upsample_nearest(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Nearest-neighbor upsample (crisp blocks, no smoothing)."""
    if scale <= 1:
        return x
    return F.interpolate(x, scale_factor=scale, mode="nearest")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./perturb_viz")
    p.add_argument("--split", type=str, choices=["train", "test"], default="test")
    p.add_argument("--n", type=int, default=20, help="Number of images to export")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0, help="Global seed for reproducibility")

    # Perturbation strengths
    p.add_argument("--eps_raw", type=float, default=2.0 / 255.0, help="L_inf bound in pixel space")
    p.add_argument("--b", type=float, default=0.05, help="brightness jitter strength")
    p.add_argument("--c", type=float, default=0.05, help="contrast jitter strength")
    p.add_argument("--max_shift", type=int, default=1, help="translation in pixels")
    p.add_argument("--blur_sigma", type=float, default=0.5, help="blur sigma")

    # Visualization
    p.add_argument("--scale", type=int, default=6, help="Nearest-neighbor upscaling factor (e.g., 4 or 6)")
    p.add_argument("--no_upsample", action="store_true", help="Disable upsampling even if --scale > 1")
    p.add_argument("--padding", type=int, default=6, help="Padding between tiles in the strip")

    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # Load CIFAR normalized (same domain as your model inputs)
    tfm = T.Compose(
        [
            T.ToTensor(),  # [0,1]
            T.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ]
    )

    ds = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=(args.split == "train"),
        download=True,
        transform=tfm,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    exported = 0
    for xb, yb in loader:
        B = xb.size(0)

        for i in range(B):
            if exported >= args.n:
                break

            # Make per-image randomness reproducible
            torch.manual_seed(args.seed + exported)

            # Clone so "orig" cannot be modified via aliasing
            x0 = xb[i : i + 1].clone()  # (1,3,32,32) normalized
            x0_ref = x0.clone()

            # Apply perturbations to the SAME original x0 (not sequentially)
            x1 = x0 + cifar_uniform_linf_noise_like(x0, eps_raw=args.eps_raw)
            x2 = brightness_contrast_jitter(x0, b=args.b, c=args.c)
            x3 = random_translation(x0, max_shift=args.max_shift)
            x4 = mild_gaussian_blur(x0, sigma=args.blur_sigma)

            # Debug: ensure x0 wasn't changed in-place
            diff = (x0 - x0_ref).abs().max().item()
            assert diff == 0.0, f"Original was modified in-place! max|diff|={diff}"

            # Stack: (5,3,H,W)
            xs = torch.cat([x0, x1, x2, x3, x4], dim=0)

            # Convert to [0,1] for saving
            xs_vis = clamp01(unnormalize_cifar(xs))

            # Upscale with nearest-neighbor to avoid smoothing blur
            if not args.no_upsample:
                xs_vis = upsample_nearest(xs_vis, scale=args.scale)

            # Make a 1x5 strip grid
            grid = make_grid(xs_vis, nrow=5, padding=args.padding)

            label = int(yb[i].item())
            out_path = os.path.join(args.out_dir, f"img_{exported:04d}_y{label}.png")
            save_image(grid, out_path)

            exported += 1

        if exported >= args.n:
            break

    print(f"Saved {exported} strips to: {args.out_dir}")
    print("If the original still looks 'blurred', it is likely viewer interpolation; "
          "try viewing at 100% or use nearest interpolation in your viewer.")


if __name__ == "__main__":
    main()
