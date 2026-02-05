from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DeviceConfig:
    device: torch.device
    use_amp: bool = False
    amp_dtype: torch.dtype = torch.float16  # can switch to bfloat16 if supported


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Returns a torch.device.
    Args:
        device_str: "cuda", "cpu", "cuda:0", etc. If None, auto-select.
    """
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def get_device_config(
    device_str: Optional[str] = None,
    use_amp: bool = False,
    prefer_bfloat16: bool = True,
) -> DeviceConfig:
    """
    Device + AMP configuration helper.
    """
    device = get_device(device_str)

    amp_dtype = torch.float16
    if prefer_bfloat16 and device.type == "cuda":
        amp_dtype = torch.bfloat16

    return DeviceConfig(device=device, use_amp=use_amp, amp_dtype=amp_dtype)
