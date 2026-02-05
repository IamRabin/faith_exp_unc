# training.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

# your code
from src.data.cifar import get_cifar_loaders
from src.models.resnet import build_model


# ---------------------------
# SGT loss (self-contained)
# ---------------------------

@dataclass
class SGTParams:
    lam_kl: float = 1.0
    mask_ratio: float = 0.5
    temperature: float = 1.0
    mask_fill: str = "random_uniform"  # "zero" | "random_uniform" | "random_normal"
    create_graph: bool = False         # True = 2nd-order grads, slower; False usually fine


def _fill_like(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(x)
    if mode == "random_normal":
        return torch.randn_like(x)
    if mode == "random_uniform":
        return torch.rand_like(x)
    raise ValueError(f"Unknown mask_fill: {mode}")


def sgt_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    p: SGTParams,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """
    Returns: total_loss, components_dict, logits_on_x
    Implements:
        CE(f(x), y) + lam * KL( p(.|x) || p(.|x_mask) )
    where x_mask masks lowest-saliency pixels based on |d logit_y / d x|.
    """
    # need saliency grads w.r.t. input
    x_req = x.detach()
    x_req.requires_grad_(True)

    logits = model(x_req)
    ce = F.cross_entropy(logits, y)

    idx = torch.arange(logits.size(0), device=logits.device)
    true_logit = logits[idx, y].sum()

    grad_x = torch.autograd.grad(
        outputs=true_logit,
        inputs=x_req,
        create_graph=p.create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]  # [B,C,H,W]

    sal = grad_x.abs().mean(dim=1, keepdim=True)  # [B,1,H,W]

    # build mask keeping top (1-mask_ratio), masking bottom mask_ratio
    B = sal.size(0)
    flat = sal.view(B, -1)
    N = flat.size(1)
    k = int(p.mask_ratio * N)

    if k > 0:
        kk = max(1, min(k, N))
        thresh = torch.kthvalue(flat, kk, dim=1).values.view(B, 1, 1, 1)
        keep = (sal > thresh).float()
    else:
        keep = torch.ones_like(sal)

    keep = keep.expand_as(x_req)

    fill = _fill_like(x_req, p.mask_fill)
    x_mask = x_req * keep + fill * (1.0 - keep)

    logits_m = model(x_mask)

    T = float(p.temperature)
    log_p = F.log_softmax(logits / T, dim=1)
    q = F.softmax(logits_m / T, dim=1)
    kl = F.kl_div(log_p, q, reduction="batchmean") * (T * T)

    total = ce + p.lam_kl * kl

    comps = {
        "ce": ce.detach(),
        "kl": kl.detach(),
        "total": total.detach(),
    }
    return total, comps, logits


# ---------------------------
# Lightning module
# ---------------------------

class CIFARLitModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        method: str = "standard_ce",
        lr: float = 0.1,
        weight_decay: float = 5e-4,
        momentum: float = 0.9,
        max_epochs: int = 200,
        scheduler: str = "cosine",  # "cosine" | "multistep" | "none"
        sgt: Optional[SGTParams] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.method = method
        self.sgt = sgt or SGTParams()

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _acc(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean()

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.method == "standard_ce":
            logits = self(x)
            loss = F.cross_entropy(logits, y)
            comps: Dict[str, torch.Tensor] = {}
        elif self.method == "sgt":
            loss, comps, logits = sgt_loss(self.model, x, y, self.sgt)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        acc = self._acc(logits.detach(), y)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)

        if "ce" in comps:
            self.log("train/ce", comps["ce"], prog_bar=False)
        if "kl" in comps:
            self.log("train/kl", comps["kl"], prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self._acc(logits, y)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self._acc(logits, y)

        self.log("test/loss", loss, prog_bar=True, sync_dist=True)
        self.log("test/acc", acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
            nesterov=True,
        )

        if self.hparams.scheduler == "none":
            return opt

        if self.hparams.scheduler == "cosine":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.hparams.max_epochs
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

        if self.hparams.scheduler == "multistep":
            sch = torch.optim.lr_scheduler.MultiStepLR(
                opt, milestones=[100, 150], gamma=0.1
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

        raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")


# ---------------------------
# CLI / runner
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_size", type=float, default=5000)
    p.add_argument("--split_seed", type=int, default=0)

    # model
    p.add_argument("--model", type=str, default="resnet18")

    # training
    p.add_argument("--method", type=str, default="sgt", choices=["standard_ce", "sgt"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "multistep", "none"])

    # sgt
    p.add_argument("--lam_kl", type=float, default=1.0)
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--mask_fill", type=str, default="random_uniform",
                   choices=["zero", "random_uniform", "random_normal"])
    p.add_argument("--create_graph", action="store_true",
                   help="Use 2nd-order grads for saliency (slow). Usually keep False.")

    # lightning
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", type=str, default="16-mixed")  # or "32-true"
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--log_dir", type=str, default="./runs")
    p.add_argument("--exp_name", type=str, default="cifar_sgt")

    return p.parse_args()


def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

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

    model = build_model(args.model, num_classes=loaders.num_classes)

    sgt_params = SGTParams(
        lam_kl=args.lam_kl,
        mask_ratio=args.mask_ratio,
        temperature=args.temperature,
        mask_fill=args.mask_fill,
        create_graph=args.create_graph,
    )

    lit = CIFARLitModule(
        model=model,
        method=args.method,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        scheduler=args.scheduler,
        sgt=sgt_params,
    )

    logger = CSVLogger(save_dir=args.log_dir, name=args.exp_name)

    ckpt = ModelCheckpoint(
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        filename="{epoch:03d}-{val_acc:.4f}",
    )
    lrmon = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=[ckpt, lrmon],
        log_every_n_steps=50,
    )

    trainer.fit(lit, train_dataloaders=loaders.train, val_dataloaders=loaders.val)
    trainer.test(lit, dataloaders=loaders.test, ckpt_path="best")


if __name__ == "__main__":
    main()
