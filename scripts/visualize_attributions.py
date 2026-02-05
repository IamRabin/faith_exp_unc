# scripts/visualize_attributions.py
from __future__ import annotations

import argparse
import os
import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.cifar import get_cifar_loaders
from src.models.resnet import build_model
from src.utils.seed import set_seed
from src.utils.device import get_device

from src.explain.saliency import compute_saliency
from src.explain.stats import compute_saliency_stats
from src.explain.perturbations import (
    PerturbFn,
    cifar_uniform_linf_noise_like,
    brightness_contrast_jitter,
    random_translation,
    mild_gaussian_blur,
)

from src.explain.attributions import (
    importance_integrated_gradients,
    importance_deepshap,
    estimate_dataset_mean_from_loader,
)

# -----------------------------
# CIFAR unnormalize helpers
# -----------------------------
_CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
_CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)


def unnormalize_cifar(x_norm: torch.Tensor) -> torch.Tensor:
    """x_norm: (B,3,H,W) normalized with CIFAR mean/std -> returns approx [0,1]."""
    mean = _CIFAR_MEAN.to(x_norm.device, x_norm.dtype)
    std = _CIFAR_STD.to(x_norm.device, x_norm.dtype)
    return x_norm * std + mean


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


# -----------------------------
# Lightning ckpt loader
# -----------------------------
def load_model_from_lightning_ckpt(
    ckpt_path: str,
    *,
    model_name: str,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    model_sd = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            model_sd[k[len("model."):]] = v
    if not model_sd:
        model_sd = state_dict

    model = build_model(model_name, num_classes=num_classes)
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict strict=False: missing={missing} unexpected={unexpected}")

    model = model.to(device)
    model.eval()
    return model


# -----------------------------
# Attribution maps (return abs attribution: (B,3,H,W))
# -----------------------------
@torch.no_grad()
def _predict_labels(model, x: torch.Tensor) -> torch.Tensor:
    return model(x).argmax(dim=1)


def attr_vanilla(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    g = compute_saliency(model, x, y, create_graph=False)
    return g.abs()


def build_perturbations(eps_raw: float) -> List[PerturbFn]:
    return [
        lambda x: x + cifar_uniform_linf_noise_like(x, eps_raw=eps_raw),
        lambda x: brightness_contrast_jitter(x, b=0.05, c=0.05),
        lambda x: random_translation(x, max_shift=1),
        lambda x: mild_gaussian_blur(x, sigma=0.5),
    ]


def attr_mean_saliency(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    K: int,
    perturbations: List[PerturbFn],
) -> torch.Tensor:
    stats = compute_saliency_stats(
        model,
        x,
        y,
        K=K,
        create_graph=False,
        perturbations=perturbations,
        deterministic=True,  # with K==len(list), applies each once
    )
    return stats.mu.abs()


def attr_ig(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int,
    baseline_mode: str,
    dataset_mean: Optional[torch.Tensor],
) -> torch.Tensor:
    return importance_integrated_gradients(
        model,
        x,
        y,
        n_steps=steps,
        baseline_mode=baseline_mode,
        dataset_mean=dataset_mean,
    ).abs()


def attr_deepshap(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_baselines: int,
    baseline_mode: str,
    dataset_mean: Optional[torch.Tensor],
    n_samples: int,
    stdev: float,
) -> torch.Tensor:
    return importance_deepshap(
        model,
        x,
        y,
        n_baselines=n_baselines,
        baseline_mode=baseline_mode,
        dataset_mean=dataset_mean,
        n_samples=n_samples,
        stdev=stdev,
    ).abs()


# -----------------------------
# Heatmap + overlay utilities
# -----------------------------
def heatmap_from_attr(attr: torch.Tensor) -> torch.Tensor:
    """
    attr: (B,3,H,W) or (B,1,H,W) or (B,H,W)
    returns heat: (B,1,H,W) in [0,1] after robust per-image normalization.
    """
    if attr.ndim == 4:
        h = attr.mean(dim=1, keepdim=True)
    elif attr.ndim == 3:
        h = attr.unsqueeze(1)
    else:
        raise ValueError(f"Unexpected attr shape: {tuple(attr.shape)}")

    h_flat = h.flatten(start_dim=1)
    h_sorted, _ = torch.sort(h_flat, dim=1)

    # 1% and 99% indices
    n = h_sorted.size(1)
    lo_i = int(round(0.01 * (n - 1)))
    hi_i = int(round(0.99 * (n - 1)))
    lo_i = max(0, min(lo_i, n - 1))
    hi_i = max(0, min(hi_i, n - 1))

    lo = h_sorted[:, lo_i].unsqueeze(1)
    hi = h_sorted[:, hi_i].unsqueeze(1)

    h_norm = (h_flat - lo) / (hi - lo + 1e-12)
    h_norm = h_norm.clamp(0.0, 1.0).view_as(h)
    return h_norm


def overlay_on_image(img01: torch.Tensor, heat01: torch.Tensor, alpha: float = 0.45) -> torch.Tensor:
    """
    img01: (B,3,H,W) in [0,1]
    heat01: (B,1,H,W) in [0,1]
    returns: (B,3,H,W) overlay image in [0,1]
    """
    heat_rgb = torch.cat([heat01, torch.zeros_like(heat01), torch.zeros_like(heat01)], dim=1)
    out = (1 - alpha) * img01 + alpha * heat_rgb
    return out.clamp(0.0, 1.0)


def upscale_nearest(x: torch.Tensor, scale: int) -> torch.Tensor:
    if scale <= 1:
        return x
    return F.interpolate(x, scale_factor=scale, mode="nearest")


# -----------------------------
# Attribution distribution helpers
# -----------------------------
@dataclass
class AttrStats:
    log10_l1: float
    log10_l2: float
    log10_linf: float
    topk_mass_ratio: float


def _safe_log10(x: float, eps: float = 1e-12) -> float:
    return math.log10(max(x, eps))


def compute_attr_stats(attr: torch.Tensor, topk_percent: float = 5.0) -> List[AttrStats]:
    """
    attr: (B,C,H,W) absolute attribution.
    """
    if attr.ndim < 2:
        raise ValueError(f"attr must be at least 2D (B, ...), got {attr.shape}")

    a = attr.flatten(start_dim=1)  # (B,D)
    l1 = a.sum(dim=1)
    l2 = torch.sqrt((a * a).sum(dim=1) + 1e-12)
    linf = a.max(dim=1).values

    D = a.shape[1]
    k = max(1, int(round((topk_percent / 100.0) * D)))

    topk_vals, _ = torch.topk(a, k=k, dim=1, largest=True, sorted=False)
    mass_ratio = topk_vals.sum(dim=1) / (l1 + 1e-12)

    out: List[AttrStats] = []
    for i in range(a.size(0)):
        out.append(
            AttrStats(
                log10_l1=_safe_log10(float(l1[i].item())),
                log10_l2=_safe_log10(float(l2[i].item())),
                log10_linf=_safe_log10(float(linf[i].item())),
                topk_mass_ratio=float(mass_ratio[i].item()),
            )
        )
    return out


def _plot_two_group_panel(
    values_a: List[float],
    values_b: List[float],
    *,
    label_a: str,
    label_b: str,
    ylabel: str,
    title: str,
    out_png: str,
):
    plt.figure(figsize=(4.2, 3.2))
    ax = plt.gca()

    data = [values_a, values_b]
    pos = [1, 2]

    ax.violinplot(
        data,
        positions=pos,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    ax.boxplot(
        data,
        positions=pos,
        widths=0.25,
        patch_artist=True,
        showfliers=False,
    )
    ax.set_xticks(pos)
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout(pad=0.2)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def _load_png_as_tensor(path: str) -> torch.Tensor:
    """
    Load an RGB PNG saved by matplotlib and return float tensor (3,H,W) in [0,1].
    Uses PIL via torchvision to avoid extra deps.
    """
    from PIL import Image
    img = Image.open(path).convert("RGB")
    t = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float() / 255.0
    return t


def select_images(
    loader,
    *,
    model_ce,
    model_ours,
    device: torch.device,
    n_images: int,
    only_correct: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if only_correct:
            with torch.no_grad():
                pred_ce = _predict_labels(model_ce, x)
                pred_ours = _predict_labels(model_ours, x)
                ok = (pred_ce == y) & (pred_ours == y)
            if not ok.any():
                continue
            x = x[ok]
            y = y[ok]

        xs.append(x)
        ys.append(y)
        if sum(t.size(0) for t in xs) >= n_images:
            break

    if not xs:
        raise RuntimeError("No images selected (try removing --only_correct).")

    x_sel = torch.cat(xs, dim=0)[: n_images]
    y_sel = torch.cat(ys, dim=0)[: n_images]
    return x_sel, y_sel


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_ce", type=str, required=True)
    p.add_argument("--run_ours", type=str, required=True)
    p.add_argument("--ckpt_name", type=str, default="best.ckpt")
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--split", type=str, choices=["val", "test"], default="test")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_size", type=int, default=5000)

    p.add_argument("--n_images", type=int, default=8)
    p.add_argument("--only_correct", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)

    # mean-saliency perturbation params
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--eps_raw", type=float, default=2.0 / 255.0)

    # optional explainers
    p.add_argument("--include_ig", action="store_true")
    p.add_argument("--include_deepshap", action="store_true")
    p.add_argument("--ig_steps", type=int, default=32)
    p.add_argument("--attr_baseline", type=str, default="zero", choices=["zero", "mean", "noise"])
    p.add_argument("--mean_from_loader_batches", type=int, default=1)
    p.add_argument("--shap_baselines", type=int, default=8)
    p.add_argument("--shap_samples", type=int, default=16)
    p.add_argument("--shap_stdev", type=float, default=0.1)

    # visualization
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--upscale", type=int, default=8)
    p.add_argument("--out_dir", type=str, default="figures/qualitative")

    # combined LaTeX-ready figure
    p.add_argument("--make_combined", action="store_true")
    p.add_argument("--dist_metric", type=str, default="topk", choices=["topk", "log10_l1", "log10_l2", "log10_linf"])
    p.add_argument("--topk_percent", type=float, default=5.0)

    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    loaders = get_cifar_loaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
        use_val=True,
        val_size=args.val_size,
        split_seed=args.seed,
    )
    loader = loaders.val if args.split == "val" else loaders.test

    ce_ckpt = os.path.join(args.run_ce, args.ckpt_name)
    ours_ckpt = os.path.join(args.run_ours, args.ckpt_name)
    if not os.path.isfile(ce_ckpt):
        raise FileNotFoundError(f"CE checkpoint not found: {ce_ckpt}")
    if not os.path.isfile(ours_ckpt):
        raise FileNotFoundError(f"OURS checkpoint not found: {ours_ckpt}")

    model_ce = load_model_from_lightning_ckpt(
        ce_ckpt, model_name=args.model, num_classes=loaders.num_classes, device=device
    )
    model_ours = load_model_from_lightning_ckpt(
        ours_ckpt, model_name=args.model, num_classes=loaders.num_classes, device=device
    )

    dataset_mean = None
    if args.attr_baseline == "mean" and (args.include_ig or args.include_deepshap):
        dataset_mean = estimate_dataset_mean_from_loader(
            loader, max_batches=args.mean_from_loader_batches
        )

    perturbations = build_perturbations(args.eps_raw)

    # -----------------------------
    # Select N images
    # -----------------------------
    x_sel, y_sel = select_images(
        loader,
        model_ce=model_ce,
        model_ours=model_ours,
        device=device,
        n_images=args.n_images,
        only_correct=args.only_correct,
    )

    # -----------------------------
    # Compute attributions
    # -----------------------------
    with torch.set_grad_enabled(True):
        # CE
        attr_ce_van = attr_vanilla(model_ce, x_sel, y_sel)
        attr_ce_mean = attr_mean_saliency(
            model_ce, x_sel, y_sel, K=args.K, perturbations=perturbations
        )

        # OURS
        attr_ours_van = attr_vanilla(model_ours, x_sel, y_sel)
        attr_ours_mean = attr_mean_saliency(
            model_ours, x_sel, y_sel, K=args.K, perturbations=perturbations
        )

        # Optional IG/DeepSHAP
        attr_ce_ig = attr_ours_ig = None
        if args.include_ig:
            dm = dataset_mean.to(device) if dataset_mean is not None else None
            attr_ce_ig = attr_ig(
                model_ce, x_sel, y_sel,
                steps=args.ig_steps,
                baseline_mode=args.attr_baseline,
                dataset_mean=dm,
            )
            attr_ours_ig = attr_ig(
                model_ours, x_sel, y_sel,
                steps=args.ig_steps,
                baseline_mode=args.attr_baseline,
                dataset_mean=dm,
            )

        attr_ce_shap = attr_ours_shap = None
        if args.include_deepshap:
            dm = dataset_mean.to(device) if dataset_mean is not None else None
            attr_ce_shap = attr_deepshap(
                model_ce, x_sel, y_sel,
                n_baselines=args.shap_baselines,
                baseline_mode=args.attr_baseline,
                dataset_mean=dm,
                n_samples=args.shap_samples,
                stdev=args.shap_stdev,
            )
            attr_ours_shap = attr_deepshap(
                model_ours, x_sel, y_sel,
                n_baselines=args.shap_baselines,
                baseline_mode=args.attr_baseline,
                dataset_mean=dm,
                n_samples=args.shap_samples,
                stdev=args.shap_stdev,
            )

    # base image (unnormalized)
    img01 = clamp01(unnormalize_cifar(x_sel).detach())

    def make_overlay(attr: torch.Tensor) -> torch.Tensor:
        heat01 = heatmap_from_attr(attr.detach())
        return overlay_on_image(img01, heat01, alpha=args.alpha)

    ce_van_ov = make_overlay(attr_ce_van)
    ce_mean_ov = make_overlay(attr_ce_mean)
    ours_van_ov = make_overlay(attr_ours_van)
    ours_mean_ov = make_overlay(attr_ours_mean)

    ce_ig_ov = ours_ig_ov = None
    if attr_ce_ig is not None:
        ce_ig_ov = make_overlay(attr_ce_ig)
        ours_ig_ov = make_overlay(attr_ours_ig)

    ce_shap_ov = ours_shap_ov = None
    if attr_ce_shap is not None:
        ce_shap_ov = make_overlay(attr_ce_shap)
        ours_shap_ov = make_overlay(attr_ours_shap)

    # -----------------------------
    # Build qualitative grid PNG
    # -----------------------------
    def strip(x_bchw: torch.Tensor) -> torch.Tensor:
        x_up = upscale_nearest(x_bchw, args.upscale)
        return make_grid(x_up, nrow=args.n_images, padding=2)

    strips: List[torch.Tensor] = []

    # Row 1: Original + CE overlays
    strips.append(strip(img01))
    strips.append(strip(ce_van_ov))
    strips.append(strip(ce_mean_ov))
    if ce_ig_ov is not None:
        strips.append(strip(ce_ig_ov))
    if ce_shap_ov is not None:
        strips.append(strip(ce_shap_ov))

    # Row 2: Original + OURS overlays
    strips.append(strip(img01))
    strips.append(strip(ours_van_ov))
    strips.append(strip(ours_mean_ov))
    if ours_ig_ov is not None:
        strips.append(strip(ours_ig_ov))
    if ours_shap_ov is not None:
        strips.append(strip(ours_shap_ov))

    grid = torch.stack(strips, dim=0)  # (R,3,H,W)
    grid = make_grid(grid, nrow=1, padding=6)  # vertical stack

    os.makedirs(args.out_dir, exist_ok=True)
    qual_png = os.path.join(args.out_dir, f"qualitative_{args.split}_N{args.n_images}.png")
    save_image(grid, qual_png)
    print(f"Saved qualitative grid: {qual_png}")

    # Save per-example strips
    per_example_dir = os.path.join(args.out_dir, "per_example")
    os.makedirs(per_example_dir, exist_ok=True)
    for i in range(args.n_images):
        tiles = [
            img01[i : i + 1],
            ce_van_ov[i : i + 1],
            ce_mean_ov[i : i + 1],
            ours_van_ov[i : i + 1],
            ours_mean_ov[i : i + 1],
        ]
        if ce_ig_ov is not None:
            tiles.insert(3, ce_ig_ov[i : i + 1])
            tiles.append(ours_ig_ov[i : i + 1])
        if ce_shap_ov is not None:
            tiles.append(ce_shap_ov[i : i + 1])
            tiles.append(ours_shap_ov[i : i + 1])

        tile = torch.cat([upscale_nearest(t, args.upscale) for t in tiles], dim=0)
        tile_grid = make_grid(tile, nrow=tile.size(0), padding=2)
        save_image(tile_grid, os.path.join(per_example_dir, f"ex_{i:03d}.png"))
    print(f"Saved per-example strips to: {per_example_dir}")

    # -----------------------------
    # Distribution plot on the SAME selected images (CE vs OURS)
    # Default: mean saliency (matches your method)
    # -----------------------------
    stats_ce = compute_attr_stats(attr_ce_mean.abs().float().cpu(), topk_percent=args.topk_percent)
    stats_ours = compute_attr_stats(attr_ours_mean.abs().float().cpu(), topk_percent=args.topk_percent)

    if args.dist_metric == "topk":
        ce_vals = [s.topk_mass_ratio for s in stats_ce]
        ours_vals = [s.topk_mass_ratio for s in stats_ours]
        ylabel = f"Top-{int(args.topk_percent)}% mass ratio"
        metric_tag = f"top{int(args.topk_percent)}_mass"
    elif args.dist_metric == "log10_l1":
        ce_vals = [s.log10_l1 for s in stats_ce]
        ours_vals = [s.log10_l1 for s in stats_ours]
        ylabel = r"$\log_{10}\|a\|_1$"
        metric_tag = "log10_l1"
    elif args.dist_metric == "log10_l2":
        ce_vals = [s.log10_l2 for s in stats_ce]
        ours_vals = [s.log10_l2 for s in stats_ours]
        ylabel = r"$\log_{10}\|a\|_2$"
        metric_tag = "log10_l2"
    else:
        ce_vals = [s.log10_linf for s in stats_ce]
        ours_vals = [s.log10_linf for s in stats_ours]
        ylabel = r"$\log_{10}\|a\|_\infty$"
        metric_tag = "log10_linf"

    dist_png = os.path.join(args.out_dir, f"dist_{args.split}_N{args.n_images}_{metric_tag}.png")
    _plot_two_group_panel(
        ce_vals,
        ours_vals,
        label_a="CE",
        label_b="OURS",
        ylabel=ylabel,
        title=f"{args.dataset.upper()} {args.split} — mean saliency (N={args.n_images})",
        out_png=dist_png,
    )
    print(f"Saved distribution panel: {dist_png}")

    # -----------------------------
    # Combined LaTeX-ready figure: [qualitative grid | distribution panel]
    # Saved as both PNG and PDF.
    # -----------------------------
    if args.make_combined:
        # Load saved images as tensors
        qual_img = _load_png_as_tensor(qual_png)     # (3,H,W)
        dist_img = _load_png_as_tensor(dist_png)     # (3,h,w)

        # Match heights (pad the shorter one)
        Hq = qual_img.shape[1]
        Hd = dist_img.shape[1]
        if Hd != Hq:
            # resize dist panel to match qualitative height (keeps it clean)
            dist_img_b = dist_img.unsqueeze(0)
            dist_img_b = F.interpolate(dist_img_b, size=(Hq, dist_img.shape[2]), mode="bilinear", align_corners=False)
            dist_img = dist_img_b.squeeze(0)

        # Add a small white gap between panels
        gap = 20
        gap_img = torch.ones(3, qual_img.shape[1], gap)

        combined = torch.cat([qual_img, gap_img, dist_img], dim=2)  # concat width

        comb_png = os.path.join(args.out_dir, f"combined_{args.split}_N{args.n_images}_{metric_tag}.png")
        save_image(combined, comb_png)
        print(f"Saved combined PNG: {comb_png}")

        # Save a PDF version too (best for LaTeX)
        comb_pdf = os.path.join(args.out_dir, f"combined_{args.split}_N{args.n_images}_{metric_tag}.pdf")
        plt.figure(figsize=(12, 6))
        plt.axis("off")
        plt.imshow(combined.permute(1, 2, 0).cpu().numpy())
        plt.tight_layout(pad=0.0)
        plt.savefig(comb_pdf, dpi=300, bbox_inches="tight", pad_inches=0.0)
        plt.close()
        print(f"Saved combined PDF: {comb_pdf}")


if __name__ == "__main__":
    main()
