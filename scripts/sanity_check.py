# scripts/sanity_check_full_loss.py
from __future__ import annotations

import argparse

import torch

from src.data.cifar import get_cifar_loaders
from src.models.resnet import build_model
from src.losses.full_loss import ExplanationTrainingLoss
from src.utils.seed import set_seed
from src.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--eps_raw", type=float, default=2.0 / 255.0)

    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--kappa", type=float, default=1.0)

    parser.add_argument("--lambda_sal", type=float, default=0.1)
    parser.add_argument("--lambda_unc", type=float, default=0.1)
    parser.add_argument("--lambda_faith", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=2.0)

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    loaders = get_cifar_loaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_val=True,
        val_size=5000,
        split_seed=args.seed,
    )

    model = build_model(args.model, num_classes=loaders.num_classes).to(device)
    model.train()

    loss_fn = ExplanationTrainingLoss(
        K=args.K,
        eps_raw=args.eps_raw,
        beta=args.beta,
        tau=args.tau,
        kappa=args.kappa,
        lambda_sal=args.lambda_sal,
        lambda_unc=args.lambda_unc,
        lambda_faith=args.lambda_faith,
        T=args.T,
    )

    x, y = next(iter(loaders.train))
    x, y = x.to(device), y.to(device)

    out = loss_fn(model, x, y, create_graph=True)

    model.zero_grad(set_to_none=True)
    out.total.backward()

    any_grad = any(p.grad is not None for p in model.parameters())

    print("=== Sanity Check (Step 7: Full Loss) ===")
    print(f"total={out.total.item():.4f} params_have_grad={any_grad}")
    print("components:", {k: float(v.item()) for k, v in out.components.items()})
    print("mask range:", float(out.mask.min().item()), float(out.mask.max().item()))


if __name__ == "__main__":
    main()
