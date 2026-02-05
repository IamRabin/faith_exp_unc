# eval.py
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Callable, Dict, Optional, Sequence

import torch

from src.data.cifar import get_cifar_loaders
from src.models.resnet import build_model
from src.utils.seed import set_seed
from src.utils.device import get_device

from src.metrics.invariance import compute_invariance_metrics
from src.metrics.stability import compute_stability_uncertainty_sparsity
from src.metrics.deletion_insertion import evaluate_deletion_insertion
from src.explain.saliency import compute_saliency
from src.explain.stats import compute_saliency_stats
from src.train.engine import accuracy_top1

from src.explain.attributions import (
    importance_integrated_gradients,
    importance_deepshap,
    importance_captum_gradient,
    importance_smoothgrad,
    estimate_dataset_mean_from_loader,
)

from src.explain.perturbations import (
    PerturbFn,
    cifar_uniform_linf_noise_like,
    brightness_contrast_jitter,
    random_translation,
    mild_gaussian_blur,
)


def importance_vanilla_saliency(model, x, y):
    g = compute_saliency(model, x, y, create_graph=False)
    return g.abs()


def importance_mean_saliency(
    model,
    x,
    y,
    *,
    K: int,
    perturbations: Sequence[PerturbFn],
    deterministic: bool = True,
):
    stats = compute_saliency_stats(
        model,
        x,
        y,
        K=K,
        create_graph=False,
        perturbations=perturbations,
        deterministic=deterministic,
    )
    return stats.mu.abs()


def _load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@torch.no_grad()
def evaluate_accuracy_top1(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.eval()
    total_acc = 0.0
    n_batches = 0

    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        acc1 = accuracy_top1(logits, y)

        total_acc += acc1
        n_batches += 1

    return total_acc / max(n_batches, 1)


def load_model_from_lightning_ckpt(
    ckpt_path: str,
    *,
    model_name: str,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    """
    Loads the underlying nn.Module weights from a Lightning checkpoint.

    Assumes the LightningModule stores the actual network at attribute `self.model`,
    so ckpt keys look like: "model.<...>"

    Returns:
        model: nn.Module on device in eval mode
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # Extract model.* keys
    model_sd: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            model_sd[k[len("model."):]] = v

    if not model_sd:
        # fallback: maybe checkpoint saved raw model weights without prefix
        model_sd = state_dict

    model = build_model(model_name, num_classes=num_classes)
    missing, unexpected = model.load_state_dict(model_sd, strict=False)

    if missing or unexpected:
        print(f"[WARN] load_state_dict strict=False: missing={missing} unexpected={unexpected}")

    model = model.to(device)
    model.eval()
    return model


def append_csv_row(path: str, row: Dict[str, Any]) -> None:
    """
    Append a single row to a CSV file, writing header if file does not exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    write_header = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_importance_fn(
    *,
    explainer: str,
    K: int,
    perturbations: Sequence[PerturbFn],
    ig_steps: int,
    shap_baselines: int,
    shap_samples: int,
    shap_stdev: float,
    smoothgrad_samples: int,
    smoothgrad_stdev: float,
    attr_baseline: str,
    dataset_mean: Optional[torch.Tensor],
) -> Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]:
    if explainer == "vanilla":
        return lambda m, x, y: importance_vanilla_saliency(m, x, y)
    if explainer == "mean":
        return lambda m, x, y: importance_mean_saliency(
            m,
            x,
            y,
            K=K,
            perturbations=perturbations,
            deterministic=(K == len(perturbations)),
        )
    if explainer == "ig":
        return lambda m, x, y: importance_integrated_gradients(
            m,
            x,
            y,
            n_steps=ig_steps,
            baseline_mode=attr_baseline,
            dataset_mean=dataset_mean,
        )
    if explainer == "captum_grad":
        return lambda m, x, y: importance_captum_gradient(m, x, y)
    if explainer == "smoothgrad":
        return lambda m, x, y: importance_smoothgrad(
            m,
            x,
            y,
            n_samples=smoothgrad_samples,
            stdev=smoothgrad_stdev,
        )
    if explainer == "deepshap":
        return lambda m, x, y: importance_deepshap(
            m,
            x,
            y,
            n_baselines=shap_baselines,
            baseline_mode=attr_baseline,
            dataset_mean=dataset_mean,
            n_samples=shap_samples,
            stdev=shap_stdev,
        )
    raise ValueError(explainer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")

    # Optional overrides
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_size", type=int, default=5000)

    # metric params
    parser.add_argument("--K", type=int, default=None)

    # NOTE: renamed from sigma to eps_raw (pixel-space) for bounded noise perturbations
    parser.add_argument("--eps_raw", type=float, default=None)

    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--T", type=float, default=None)

    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--compute_delins", action="store_true")
    parser.add_argument("--delins_steps", type=int, default=20)
    parser.add_argument("--delins_batches", type=int, default=10)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--ckpt_name", type=str, default="best.ckpt")

    parser.add_argument(
        "--explainer",
        type=str,
        default="mean",
        choices=["vanilla", "mean", "ig", "deepshap", "captum_grad", "smoothgrad"],
        help="Attribution method used for deletion/insertion ranking.",
    )
    parser.add_argument("--ig_steps", type=int, default=32)
    parser.add_argument("--shap_baselines", type=int, default=8)
    parser.add_argument("--attr_baseline", type=str, default="zero", choices=["zero", "mean", "noise"])
    parser.add_argument("--mean_from_loader_batches", type=int, default=1)
    parser.add_argument("--shap_samples", type=int, default=16)
    parser.add_argument("--shap_stdev", type=float, default=0.09)
    parser.add_argument("--smoothgrad_samples", type=int, default=16)
    parser.add_argument("--smoothgrad_stdev", type=float, default=0.10)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    # Load run config if present
    cfg_path = os.path.join(args.run_dir, "config.json")
    cfg = _load_json_if_exists(cfg_path) or {}

    dataset = args.dataset or cfg.get("dataset") or "cifar10"
    model_name = args.model or cfg.get("model") or "resnet18"

    K = args.K if args.K is not None else int(cfg.get("K", 4))
    beta = args.beta if args.beta is not None else float(cfg.get("beta", 10.0))
    tau = args.tau if args.tau is not None else float(cfg.get("tau", 0.0))
    kappa = args.kappa if args.kappa is not None else float(cfg.get("kappa", 2.0))
    T = args.T if args.T is not None else float(cfg.get("T", 2.0))

    # Default eps_raw in pixel space
    eps_raw = args.eps_raw if args.eps_raw is not None else float(cfg.get("eps_raw", 2.0 / 255.0))

    # Data loaders
    loaders = get_cifar_loaders(
        dataset=dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
        use_val=True,
        val_size=args.val_size,
        split_seed=args.seed,
    )
    loader = loaders.val if args.split == "val" else loaders.test

    # Load model from checkpoint
    ckpt_path = os.path.join(args.run_dir, args.ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = load_model_from_lightning_ckpt(
        ckpt_path,
        model_name=model_name,
        num_classes=loaders.num_classes,
        device=device,
    )

    # Define perturbations (K can be != 4; metrics will sample with replacement if needed)
    perturbations: Sequence[PerturbFn] = [
        lambda x: x + cifar_uniform_linf_noise_like(x, eps_raw=eps_raw),
        lambda x: random_translation(x, max_shift=1),
        lambda x: brightness_contrast_jitter(x, b=0.05, c=0.05),
        lambda x: mild_gaussian_blur(x, sigma=0.5),
    ]

    acc1 = evaluate_accuracy_top1(
        model=model,
        loader=loader,
        device=device,
        max_batches=args.max_batches,
    )

    inv = compute_invariance_metrics(
        model=model,
        loader=loader,
        device=device,
        K=K,
        beta=beta,
        tau=tau,
        kappa=kappa,
        T=T,
        perturbations=perturbations,
        deterministic=(K == len(perturbations)),
        max_batches=args.max_batches,
    )

    stab = compute_stability_uncertainty_sparsity(
        model=model,
        loader=loader,
        device=device,
        K=K,
        beta=beta,
        tau=tau,
        kappa=kappa,
        perturbations=perturbations,
        deterministic=(K == len(perturbations)),
        mask_threshold=0.5,
        max_batches=args.max_batches,
    )

    results: Dict[str, Any] = {
        "split": args.split,
        "run_dir": args.run_dir,
        "checkpoint": ckpt_path,
        "dataset": dataset,
        "model": model_name,
        "seed": args.seed,
        "K": K,
        "eps_raw": eps_raw,
        "beta": beta,
        "tau": tau,
        "kappa": kappa,
        "T": T,
        "max_batches": args.max_batches,
        "delta_inv": inv.delta_inv,
        "delta_sens": inv.delta_sens,
        "stability": stab.stability,
        "uncertainty_l1": stab.uncertainty_l1,
        "sparsity": stab.sparsity,
        "top1_acc": acc1,
    }

    # optional deletion/insertion AUC
    if args.compute_delins:
        dataset_mean = None
        if args.attr_baseline == "mean" and args.explainer in ["ig", "deepshap"]:
            dataset_mean = estimate_dataset_mean_from_loader(
                loader, max_batches=args.mean_from_loader_batches
            )

        imp_fn = build_importance_fn(
            explainer=args.explainer,
            K=K,
            perturbations=perturbations,
            ig_steps=args.ig_steps,
            shap_baselines=args.shap_baselines,
            shap_samples=args.shap_samples,
            shap_stdev=args.shap_stdev,
            smoothgrad_samples=args.smoothgrad_samples,
            smoothgrad_stdev=args.smoothgrad_stdev,
            attr_baseline=args.attr_baseline,
            dataset_mean=dataset_mean,
        )

        delins = evaluate_deletion_insertion(
            model=model,
            loader=loader,
            device=device,
            importance_fn=imp_fn,
            steps=args.delins_steps,
            baseline_value=0.0,
            max_batches=args.delins_batches,
            compute_random=True,
        )

        results["auc_deletion"] = float(delins.auc_deletion)
        results["auc_insertion"] = float(delins.auc_insertion)

        results["auc_deletion_random"] = float(delins.auc_deletion_random)
        results["auc_insertion_random"] = float(delins.auc_insertion_random)

        results["delta_auc_deletion_vs_random"] = float(delins.auc_deletion_random - delins.auc_deletion)
        results["delta_auc_insertion_vs_random"] = float(delins.auc_insertion - delins.auc_insertion_random)

        results["delins_steps"] = int(args.delins_steps)
        results["delins_batches"] = int(args.delins_batches)
        results["explainer"] = args.explainer
        results["attr_baseline"] = args.attr_baseline
        results["ig_steps"] = int(args.ig_steps)
        results["shap_baselines"] = int(args.shap_baselines)

        if delins.deletion_curve.numel() > 0:
            results["deletion_curve"] = delins.deletion_curve.detach().cpu().tolist()
            results["insertion_curve"] = delins.insertion_curve.detach().cpu().tolist()
        else:
            results["deletion_curve"] = []
            results["insertion_curve"] = []

        if getattr(delins, "deletion_curve_random", None) is not None and delins.deletion_curve_random.numel() > 0:
            results["deletion_curve_random"] = delins.deletion_curve_random.detach().cpu().tolist()
            results["insertion_curve_random"] = delins.insertion_curve_random.detach().cpu().tolist()
        else:
            results["deletion_curve_random"] = []
            results["insertion_curve_random"] = []

    # Save JSON
    out_json = os.path.join(args.run_dir, f"metrics_{args.split}_{args.explainer}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Also append CSV
    out_csv = os.path.join(args.run_dir, f"eval_metrics_{args.explainer}.csv")
    append_csv_row(out_csv, results)

    print(json.dumps(results, indent=2))
    print(f"Saved metrics to {out_json}")
    print(f"Appended metrics to {out_csv}")

    # Save exact tensors for plotting later
    if args.compute_delins:
        torch.save(
            {
                "deletion_curve": delins.deletion_curve.detach().cpu(),
                "insertion_curve": delins.insertion_curve.detach().cpu(),
                "auc_deletion": float(delins.auc_deletion),
                "auc_insertion": float(delins.auc_insertion),
                "deletion_curve_random": delins.deletion_curve_random.detach().cpu(),
                "insertion_curve_random": delins.insertion_curve_random.detach().cpu(),
                "auc_deletion_random": float(delins.auc_deletion_random),
                "auc_insertion_random": float(delins.auc_insertion_random),
                "steps": int(args.delins_steps),
                "max_batches": int(args.delins_batches),
                "split": args.split,
                "checkpoint": ckpt_path,
            },
            os.path.join(args.run_dir, f"delins_curves_{args.split}_{args.explainer}.pt"),
        )


if __name__ == "__main__":
    main()
