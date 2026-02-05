from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import optuna
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from src.data.cifar import get_cifar_loaders
from src.models.resnet import build_model
from src.losses.full_loss import ExplanationTrainingLoss
from src.train.lit_module import LitCifar  # your LightningModule


def seed_everything(seed: int) -> None:
    # your project has set_seed; use it if you want
    try:
        from src.utils.seed import set_seed
        set_seed(seed)
    except Exception:
        L.seed_everything(seed, workers=True)


def build_trainer(
    *,
    run_dir: str,
    max_epochs: int,
    devices: int,
    precision: str,
    monitor_metric: str,
    monitor_mode: str,
    patience: int,
) -> L.Trainer:
    ckpt_cb = ModelCheckpoint(
        dirpath=run_dir,
        filename="best",
        monitor=monitor_metric,
        mode=monitor_mode,
        save_top_k=1,
        save_last=False,
    )

    early_stop = EarlyStopping(
        monitor=monitor_metric,
        mode=monitor_mode,
        patience=patience,
        verbose=False,
    )

    logger = CSVLogger(save_dir=run_dir, name="lightning_logs")

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices if torch.cuda.is_available() else 1,
        strategy="ddp" if torch.cuda.is_available() and devices > 1 else "auto",
        max_epochs=max_epochs,
        precision=precision,  # "32", "16-mixed", "bf16-mixed"
        logger=logger,
        callbacks=[ckpt_cb, early_stop],
        log_every_n_steps=20,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    return trainer


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    # ---------- choose hyperparameters ----------
    # Keep the search space small initially; expand later.
    lr = trial.suggest_float("lr", 0.01, 0.2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)

    # batch size typically per GPU
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    # K is expensive; tune among small options
    K = trial.suggest_categorical("K", [2, 4])

    # L_inf bound in pixel space (consistent with ExplanationTrainingLoss)
    eps_raw = trial.suggest_categorical("eps_raw", [2.0 / 255.0, 4.0 / 255.0, 8.0 / 255.0])
    kappa = trial.suggest_categorical("kappa", [0.5, 1.0, 2.0])

    # Loss weights: keep ranges reasonable
    lambda_faith = trial.suggest_float("lambda_faith", 0.3, 2.0)
    lambda_sal = trial.suggest_float("lambda_sal", 0.02, 0.2)
    lambda_unc = trial.suggest_float("lambda_unc", 0.02, 0.2)

    # Usually fix these unless you’re explicitly ablating them
    beta = args.beta
    tau = args.tau
    T = args.T

    # ---------- trial directory ----------
    trial_dir = os.path.join(args.out_dir, args.study_name, f"trial_{trial.number:05d}")
    os.makedirs(trial_dir, exist_ok=True)

    # Save trial config (very helpful for later)
    trial_cfg: Dict[str, Any] = {
        "trial_number": trial.number,
        "dataset": args.dataset,
        "model": args.model,
        "method": "ours",
        "seed": args.seed,          # training randomness varies per trial
        "split_seed": args.split_seed,             # fixed split
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "val_size": args.val_size,
        "precision": args.precision,
        "devices": args.devices,
        "monitor": args.monitor,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "K": K,
        "eps_raw": eps_raw,
        "beta": beta,
        "tau": tau,
        "kappa": kappa,
        "lambda_sal": lambda_sal,
        "lambda_unc": lambda_unc,
        "lambda_faith": lambda_faith,
        "T": T,
    }
    with open(os.path.join(trial_dir, "trial_config.json"), "w", encoding="utf-8") as f:
        json.dump(trial_cfg, f, indent=2)

    # ---------- seed ----------
    seed_everything(trial_cfg["seed"])

    # ---------- data ----------
    loaders = get_cifar_loaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
        augment=True,
        use_val=True,
        val_size=args.val_size,
        split_seed=args.split_seed,  # FIXED split across trials
    )

    # ---------- model + loss ----------
    model = build_model(args.model, num_classes=loaders.num_classes)

    loss_fn = ExplanationTrainingLoss(
        K=K,
        eps_raw=eps_raw,
        beta=beta,
        tau=tau,
        kappa=kappa,
        lambda_sal=lambda_sal,
        lambda_unc=lambda_unc,
        lambda_faith=lambda_faith,
        T=T,
    )

    lit = LitCifar(
        model,
        method="ours",
        loss_fn=loss_fn,
        lr=lr,
        momentum=args.momentum,
        weight_decay=weight_decay,
        max_epochs=args.epochs,
    )

    # ---------- trainer ----------
    # Choose the metric you optimize:
    # - args.monitor="val/acc1" typically
    # - or optimize explanation metric if you log it in validation (advanced)
    trainer = build_trainer(
        run_dir=trial_dir,
        max_epochs=args.epochs,
        devices=args.devices,
        precision=args.precision,
        monitor_metric=args.monitor,
        monitor_mode="max",
        patience=args.patience,
    )

    trainer.fit(lit, train_dataloaders=loaders.train, val_dataloaders=loaders.val)

    # After fit, get the monitored metric from callback_metrics
    metrics = trainer.callback_metrics
    if args.monitor not in metrics:
        # If it’s missing, return NaN to mark trial as failed/unusable
        return float("nan")

    score = float(metrics[args.monitor].detach().cpu().item())

    # report to optuna
    trial.set_user_attr("run_dir", trial_dir)
    trial.set_user_attr("best_ckpt", getattr(trainer.checkpoint_callback, "best_model_path", ""))
    trial.set_user_attr("score", score)

    return score


def main() -> None:
    parser = argparse.ArgumentParser()

    # experiment basics
    parser.add_argument("--study_name", type=str, default="cifar10_ours")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna.db")  # shared across workers
    parser.add_argument("--out_dir", type=str, default="runs/optuna")

    # dataset/model
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model", type=str, default="resnet18")

    # training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_size", type=int, default=5000)
    parser.add_argument("--momentum", type=float, default=0.9)

    # split vs training randomness
    parser.add_argument("--seed", type=int, default=0)        # base seed (trial adds trial.number)
    parser.add_argument("--split_seed", type=int, default=0)  # fixed split across trials

    # explanation hyperparams you may want fixed initially
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--T", type=float, default=2.0)

    # lightning
    parser.add_argument("--devices", type=int, default=1)     # set 1 for tuning (1 GPU per trial)
    parser.add_argument("--precision", type=str, default="32")

    # tuning controls
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=0)     # seconds, 0 = no timeout
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--monitor", type=str, default="val/acc1")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    # Run trials
    study.optimize(
        lambda t: objective(t, args),
        n_trials=args.n_trials,
        timeout=args.timeout if args.timeout > 0 else None,
        gc_after_trial=True,
    )

    # Save a summary table
    summary_path = os.path.join(args.out_dir, args.study_name, "study_results.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("trial,score,run_dir,best_ckpt,params\n")
        for tr in study.trials:
            if tr.value is None:
                continue
            run_dir = tr.user_attrs.get("run_dir", "")
            best_ckpt = tr.user_attrs.get("best_ckpt", "")
            f.write(
                f"{tr.number},{tr.value},{run_dir},{best_ckpt},{json.dumps(tr.params)}\n"
            )

    print("=== Optuna done ===")
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
