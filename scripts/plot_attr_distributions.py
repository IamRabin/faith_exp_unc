from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from src.data.cifar import get_cifar_loaders
from src.models.resnet import build_model
from src.utils.seed import set_seed
from src.utils.device import get_device

from src.explain.saliency import compute_saliency
from src.explain.stats import compute_saliency_stats

from src.explain.attributions import (
    importance_integrated_gradients,
    importance_deepshap,
    estimate_dataset_mean_from_loader,
)

# NEW: import perturbations you adopted
from src.explain.perturbations import (
    PerturbFn,
    cifar_uniform_linf_noise_like,
    brightness_contrast_jitter,
    random_translation,
    mild_gaussian_blur,
)

import matplotlib.pyplot as plt


# ----------------------------
# Model loading
# ----------------------------
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


# ----------------------------
# Attribution wrappers
# ----------------------------
def importance_vanilla_saliency(model, x, y):
    g = compute_saliency(model, x, y, create_graph=False)
    return g.abs()


def importance_mean_saliency(
    model,
    x,
    y,
    *,
    K: int,
    perturbations: List[PerturbFn],
    deterministic: bool = True,
):
    stats = compute_saliency_stats(
        model,
        x,
        y,
        K=K,
        create_graph=False,
        perturbations=perturbations,
        deterministic=(deterministic and K == len(perturbations)),
    )
    return stats.mu.abs()


# ----------------------------
# Per-image summary statistics
# ----------------------------
@dataclass
class AttrStats:
    log10_l1: float
    log10_l2: float
    log10_linf: float
    topk_mass_ratio: float


def _safe_log10(x: float, eps: float = 1e-12) -> float:
    return math.log10(max(x, eps))


def compute_attr_stats(attr: torch.Tensor, topk_percent: float = 5.0) -> List[AttrStats]:
    if attr.ndim < 2:
        raise ValueError(f"attr must be at least 2D (B, ...), got {attr.shape}")

    B = attr.shape[0]
    a = attr.flatten(start_dim=1)  # (B, D)

    l1 = a.sum(dim=1)
    l2 = torch.sqrt((a * a).sum(dim=1) + 1e-12)
    linf = a.max(dim=1).values

    D = a.shape[1]
    k = max(1, int(round((topk_percent / 100.0) * D)))

    topk_vals, _ = torch.topk(a, k=k, dim=1, largest=True, sorted=False)
    topk_mass = topk_vals.sum(dim=1)
    mass_ratio = topk_mass / (l1 + 1e-12)

    out: List[AttrStats] = []
    for i in range(B):
        out.append(
            AttrStats(
                log10_l1=_safe_log10(float(l1[i].item())),
                log10_l2=_safe_log10(float(l2[i].item())),
                log10_linf=_safe_log10(float(linf[i].item())),
                topk_mass_ratio=float(mass_ratio[i].item()),
            )
        )
    return out


# ----------------------------
# Plotting
# ----------------------------
def violin_boxplot_two_groups(
    values_a: List[float],
    values_b: List[float],
    *,
    label_a: str,
    label_b: str,
    title: str,
    ylabel: str,
    out_pdf: str,
):
    plt.figure(figsize=(4.6, 3.6))
    ax = plt.gca()

    data = [values_a, values_b]
    positions = [1, 2]

    ax.violinplot(
        data,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    ax.boxplot(
        data,
        positions=positions,
        widths=0.25,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(linewidth=2.0),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([label_a, label_b])
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    ax.grid(False)
    plt.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.savefig(out_pdf, dpi=300)
    plt.close()
    print(f"Saved {out_pdf}")


def save_csv(rows: List[Dict[str, Any]], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not rows:
        raise ValueError("No rows to save.")
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_csv}")


def append_rows(
    rows: List[Dict[str, Any]],
    *,
    model_label: str,
    explainer: str,
    stats: List[AttrStats],
    topk_percent: float,
) -> None:
    for s in stats:
        rows.append(
            {
                "model": model_label,
                "explainer": explainer,
                "log10_l1": s.log10_l1,
                "log10_l2": s.log10_l2,
                "log10_linf": s.log10_linf,
                "topk_mass_ratio": s.topk_mass_ratio,
                "topk_percent": topk_percent,
            }
        )


# ----------------------------
# Helper for attribution selection
# ----------------------------
def build_importance_fn(
    *,
    explainer: str,
    K: int,
    perturbations: List[PerturbFn],
    ig_steps: int,
    attr_baseline: str,
    dataset_mean: Optional[torch.Tensor],
    shap_baselines: int,
    shap_samples: int,
    shap_stdev: float,
):
    if explainer == "vanilla":
        return lambda m, x, y: importance_vanilla_saliency(m, x, y)
    if explainer == "mean":
        return lambda m, x, y: importance_mean_saliency(
            m, x, y, K=K, perturbations=perturbations, deterministic=True
        )
    if explainer == "ig":
        return lambda m, x, y: importance_integrated_gradients(
            m, x, y,
            n_steps=ig_steps,
            baseline_mode=attr_baseline,
            dataset_mean=dataset_mean,
        )
    if explainer == "deepshap":
        return lambda m, x, y: importance_deepshap(
            m, x, y,
            n_baselines=shap_baselines,
            baseline_mode=attr_baseline,
            dataset_mean=dataset_mean,
            n_samples=shap_samples,
            stdev=shap_stdev,
        )
    raise ValueError(explainer)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_ce", type=str, required=True)
    parser.add_argument("--run_ours", type=str, required=True)

    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--ckpt_name", type=str, default="best.ckpt")

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_size", type=int, default=5000)

    # attribution / explainer
    parser.add_argument("--explainer", type=str, required=True, choices=["vanilla", "mean", "ig", "deepshap"])
    parser.add_argument("--K", type=int, default=4)

    # NEW meaning: pixel-space epsilon for L_inf noise (e.g., 2/255)
    parser.add_argument("--eps_raw", type=float, default=2.0 / 255.0)

    # IG / SHAP params
    parser.add_argument("--attr_baseline", type=str, default="zero", choices=["zero", "mean", "noise"])
    parser.add_argument("--ig_steps", type=int, default=32)
    parser.add_argument("--shap_baselines", type=int, default=8)
    parser.add_argument("--shap_samples", type=int, default=16)
    parser.add_argument("--shap_stdev", type=float, default=0.1)
    parser.add_argument("--mean_from_loader_batches", type=int, default=1)

    # stats + compute budget
    parser.add_argument("--topk_percent", type=float, default=5.0)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)

    # outputs
    parser.add_argument("--out_dir", type=str, default="figures/attr_dist")

    args = parser.parse_args()

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

    dataset_mean = None
    if args.attr_baseline == "mean" and args.explainer in ["ig", "deepshap"]:
        dataset_mean = estimate_dataset_mean_from_loader(loader, max_batches=args.mean_from_loader_batches)

    # Build perturbation list (matches your adopted set)
    perturbations: List[PerturbFn] = [
        lambda x: x + cifar_uniform_linf_noise_like(x, eps_raw=args.eps_raw),
        lambda x: brightness_contrast_jitter(x, b=0.05, c=0.05),
        lambda x: random_translation(x, max_shift=1),
        lambda x: mild_gaussian_blur(x, sigma=0.5),
    ]

    ce_ckpt = os.path.join(args.run_ce, args.ckpt_name)
    ours_ckpt = os.path.join(args.run_ours, args.ckpt_name)

    if not os.path.isfile(ce_ckpt):
        raise FileNotFoundError(f"CE checkpoint not found: {ce_ckpt}")
    if not os.path.isfile(ours_ckpt):
        raise FileNotFoundError(f"OURS checkpoint not found: {ours_ckpt}")

    model_ce = load_model_from_lightning_ckpt(ce_ckpt, model_name=args.model, num_classes=loaders.num_classes, device=device)
    model_ours = load_model_from_lightning_ckpt(ours_ckpt, model_name=args.model, num_classes=loaders.num_classes, device=device)

    imp_fn = build_importance_fn(
        explainer=args.explainer,
        K=args.K,
        perturbations=perturbations,
        ig_steps=args.ig_steps,
        attr_baseline=args.attr_baseline,
        dataset_mean=dataset_mean,
        shap_baselines=args.shap_baselines,
        shap_samples=args.shap_samples,
        shap_stdev=args.shap_stdev,
    )

    rows: List[Dict[str, Any]] = []
    stats_ce: List[AttrStats] = []
    stats_ours: List[AttrStats] = []

    for bi, (x, y) in enumerate(loader):
        if args.max_batches is not None and bi >= args.max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(True):
            attr_ce = imp_fn(model_ce, x, y).detach()
            attr_ours = imp_fn(model_ours, x, y).detach()

        batch_stats_ce = compute_attr_stats(attr_ce.abs().float().cpu(), topk_percent=args.topk_percent)
        batch_stats_ours = compute_attr_stats(attr_ours.abs().float().cpu(), topk_percent=args.topk_percent)

        stats_ce.extend(batch_stats_ce)
        stats_ours.extend(batch_stats_ours)

    # CSV
    append_rows(
        rows,
        model_label="CE",
        explainer=args.explainer,
        stats=stats_ce,
        topk_percent=args.topk_percent,
    )
    append_rows(
        rows,
        model_label="OURS",
        explainer=args.explainer,
        stats=stats_ours,
        topk_percent=args.topk_percent,
    )

    out_csv = os.path.join(args.out_dir, f"attr_stats_{args.split}_{args.explainer}.csv")
    save_csv(rows, out_csv)

    ce_vals = {
        "log10_l1": [s.log10_l1 for s in stats_ce],
        "log10_l2": [s.log10_l2 for s in stats_ce],
        "log10_linf": [s.log10_linf for s in stats_ce],
        "topk_mass_ratio": [s.topk_mass_ratio for s in stats_ce],
    }
    ours_vals = {
        "log10_l1": [s.log10_l1 for s in stats_ours],
        "log10_l2": [s.log10_l2 for s in stats_ours],
        "log10_linf": [s.log10_linf for s in stats_ours],
        "topk_mass_ratio": [s.topk_mass_ratio for s in stats_ours],
    }

    base_title = f"CIFAR-10 {args.split} — {args.explainer} attribution distribution"

    violin_boxplot_two_groups(
        ce_vals["log10_l1"], ours_vals["log10_l1"],
        label_a="CE", label_b="OURS",
        title=base_title,
        ylabel=r"$\log_{10}\|a\|_1$",
        out_pdf=os.path.join(args.out_dir, f"{args.explainer}_log10_l1.pdf"),
    )
    violin_boxplot_two_groups(
        ce_vals["log10_l2"], ours_vals["log10_l2"],
        label_a="CE", label_b="OURS",
        title=base_title,
        ylabel=r"$\log_{10}\|a\|_2$",
        out_pdf=os.path.join(args.out_dir, f"{args.explainer}_log10_l2.pdf"),
    )
    violin_boxplot_two_groups(
        ce_vals["log10_linf"], ours_vals["log10_linf"],
        label_a="CE", label_b="OURS",
        title=base_title,
        ylabel=r"$\log_{10}\|a\|_\infty$",
        out_pdf=os.path.join(args.out_dir, f"{args.explainer}_log10_linf.pdf"),
    )
    violin_boxplot_two_groups(
        ce_vals["topk_mass_ratio"], ours_vals["topk_mass_ratio"],
        label_a="CE", label_b="OURS",
        title=base_title,
        ylabel=f"Top-{args.topk_percent:.0f}% mass ratio",
        out_pdf=os.path.join(args.out_dir, f"{args.explainer}_top{int(args.topk_percent)}_mass_ratio.pdf"),
    )


if __name__ == "__main__":
    main()
