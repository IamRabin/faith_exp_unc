# src/utils/config.py
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update dictionary `base` with `updates`.
    """
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _parse_value(s: str) -> Any:
    """
    Parse CLI override values into Python types.
    Examples:
        "10" -> int
        "0.1" -> float
        "true"/"false" -> bool
        "null"/"none" -> None
        "[1,2]" -> list
        "foo" -> "foo"
    """
    sl = s.strip().lower()
    if sl in {"true", "false"}:
        return sl == "true"
    if sl in {"none", "null"}:
        return None

    # try int/float
    try:
        if "." not in s and "e" not in sl:
            return int(s)
        return float(s)
    except ValueError:
        pass

    # try yaml for lists/dicts/strings
    try:
        return yaml.safe_load(s)
    except Exception:
        return s


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[list[str]]) -> Dict[str, Any]:
    """
    Apply overrides in the form ["a.b.c=value", "x=3"].
    """
    if not overrides:
        return cfg

    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, val = item.split("=", 1)
        key = key.strip()
        val_parsed = _parse_value(val)

        # set nested keys
        parts = key.split(".")
        cur = out
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = val_parsed
    return out


def load_config(
    config_path: str,
    default_path: Optional[str] = None,
    overrides: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Load config YAML and optionally merge with default YAML, then apply overrides.

    Merge order:
        defaults -> config -> overrides
    """
    cfg = load_yaml(config_path)
    if default_path is not None:
        defaults = load_yaml(default_path)
        cfg = deep_update(defaults, cfg)
    cfg = apply_overrides(cfg, overrides)
    return cfg
