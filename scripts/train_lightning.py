from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.data.cifar import get_cifar_loaders
from src.models.resnet import build_model
from src.explain.perturbations import (
    cifar_uniform_linf_noise_like,
    brightness_contrast_jitter,
    random_translation,
    mild_gaussian_blur,
    PerturbFn,
)

from src.losses.full_loss import ExplanationTrainingLoss
from src.train.lit_module import LitCifar
from src.utils.seed import set_seed


def force_sdpa_math_for_double_backward() -> None:
    """
    For PyTorch 2.1: efficient/mem-efficient SDPA kernels may not support
    higher-order grads needed when create_graph=True. Force math SDPA.
    """
    if not torch.cuda.is_available():
        return

    # Preferred in PyTorch 2.1+
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        return
    except Exception:
        pass

    # Fallback (older-style toggles; still present in many builds)
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception as e:
        raise RuntimeError(
            "Could not force SDPA math backend. Please check your torch build."
        ) from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model", type=str, choices=["resnet18", "vit"], default="resnet18")
    parser.add_argument("--method", type=str, choices=["standard_ce", "ours"], default="standard_ce")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    # ours-specific hyperparams
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--eps_raw", type=float, default=0.0039)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--lambda_sal", type=float, default=0.1)
    parser.add_argument("--lambda_unc", type=float, default=0.1)
    parser.add_argument("--lambda_faith", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=2.0)

    parser.add_argument("--val_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=0)   # data split seed (keep fixed across runs)

    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--devices", type=int, default=2)  # number of GPUs
    parser.add_argument("--precision", type=str, default="32")  # "16-mixed" etc.
    return parser.parse_args()


def resolve_run_dir(args: argparse.Namespace) -> str:
    if args.run_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join("runs", args.dataset, args.method, f"seed{args.seed}_{ts}")
    return args.run_dir


def build_loss(args: argparse.Namespace):
    if args.method != "ours":
        return None

    eps_raw = args.eps_raw  # max perturbation size measured in raw pixel space, before normalization
    perturbations: list[PerturbFn] = [
        lambda x: x + cifar_uniform_linf_noise_like(x, eps_raw=eps_raw),
        lambda x: brightness_contrast_jitter(x, b=0.05, c=0.05),
        lambda x: random_translation(x, max_shift=1),
        lambda x: mild_gaussian_blur(x, sigma=0.5),
    ]

    return ExplanationTrainingLoss(
        K=args.K,
        eps_raw=args.eps_raw,
        beta=args.beta,
        tau=args.tau,
        kappa=args.kappa,
        lambda_sal=args.lambda_sal,
        lambda_unc=args.lambda_unc,
        lambda_faith=args.lambda_faith,
        T=args.T,
        perturbations=perturbations,
        deterministic=False,  # sample with replacement during training
    )


def should_force_sdpa_math(model_name: str) -> bool:
    return "vit" in model_name.lower()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    # run_dir
    args.run_dir = resolve_run_dir(args)
    os.makedirs(args.run_dir, exist_ok=True)

    # save config
    with open(os.path.join(args.run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # loaders
    loaders = get_cifar_loaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        use_val=True,
        val_size=args.val_size,
        split_seed=args.split_seed,
    )
    
    # NOTE: SDPA math forcing is handled inside LitCifar when needed (ViT + CUDA).
    # if should_force_sdpa_math(args.model):
    #     force_sdpa_math_for_double_backward()
    
    model = build_model(args.model, num_classes=loaders.num_classes)

    loss_fn = build_loss(args)


    lit = LitCifar(
        model,
        method=args.method,
        loss_fn=loss_fn,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
    )

    ckpt = ModelCheckpoint(
        dirpath=args.run_dir,
        filename="best",
        monitor="val/acc1",
        mode="max",
        save_top_k=1,
    )

    logger = CSVLogger(save_dir=args.run_dir, name="lightning_logs")

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices if torch.cuda.is_available() else 1,
        strategy="ddp" if torch.cuda.is_available() and args.devices > 1 else "auto",
        max_epochs=args.epochs,
        precision=args.precision,        
        logger=logger,
        callbacks=[ckpt],
        log_every_n_steps=10,
        enable_checkpointing=True,
    )

    trainer.fit(lit, train_dataloaders=loaders.train, val_dataloaders=loaders.val)


if __name__ == "__main__":
    main()
