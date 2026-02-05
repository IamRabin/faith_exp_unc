from __future__ import annotations

from contextlib import nullcontext
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from src.train.engine import accuracy_top1


class LitCifar(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        method: str,
        loss_fn=None,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        max_epochs: int = 10,
    ):
        super().__init__()
        self.model = model
        self.method = method
        self.loss_fn = loss_fn

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        if self.method == "ours" and self.loss_fn is None:
            raise ValueError("loss_fn must be provided when method='ours'")

        # logs hparams (excluding big objects)
        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _should_force_sdpa_math(self) -> bool:
        name = self.model.__class__.__name__.lower()
        return "vit" in name

    def _loss_and_logits(self, x: torch.Tensor, y: torch.Tensor):
        if self.method == "standard_ce":
            logits = self(x)
            loss = F.cross_entropy(logits, y)
            return loss, {}, logits

        if self.loss_fn is None:
            raise ValueError("loss_fn must be provided when method='ours'")

        ctx = nullcontext()
        if torch.cuda.is_available() and self._should_force_sdpa_math():
            from torch.backends.cuda import sdp_kernel
            # Force math SDPA (supports higher-order grads) for ViT attention
            ctx = sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

        with ctx:
            out = self.loss_fn(self.model, x, y, create_graph=True)

        loss = out.total
        comps = out.components
        logits = out.logits if out.logits is not None else self(x)
        return loss, comps, logits

    def training_step(self, batch, batch_idx):
        x, y = batch

        loss, comps, logits = self._loss_and_logits(x, y)
        acc1 = accuracy_top1(logits.detach(), y)

        # Lightning will sync across GPUs if sync_dist=True
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/acc1", acc1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for k, v in comps.items():
            # v should be detached already; detach defensively
            self.log(f"train/{k}", v.detach(), on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc1 = accuracy_top1(logits, y)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc1", acc1, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=0.0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
