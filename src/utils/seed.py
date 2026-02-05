# src/utils/seed.py
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set RNG seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Random seed.
        deterministic: If True, enables deterministic algorithms where possible
            (can slow down training and may restrict some ops).
    """
    if seed is None:
        raise ValueError("seed must be an int, got None")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Determinism flags (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch deterministic algorithms (may raise if an op has no deterministic impl)
        torch.use_deterministic_algorithms(True)
    else:
        # Fast default
        torch.backends.cudnn.benchmark = True
