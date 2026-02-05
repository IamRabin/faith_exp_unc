# src/train/checkpoint.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
