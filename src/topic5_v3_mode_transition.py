"""Topic 5 V3a mode-transition helpers.

This module intentionally stays on pure configuration for Task 0. Later
tasks add event-window extraction, geometry, dynamics, and avalanche-flux
estimators on top of this config contract. See
docs/superpowers/plans/2026-07-02-topic5-v3a-mode-transition.md for the
full task list; treat this line as exploratory pending the pilot-lock gate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CFG = _ROOT / "config" / "topic5_v3.yaml"


def load_v3_config(path: str | Path | None = None) -> dict:
    """Load the V3a mode-transition YAML config as a plain dict."""
    cfg_path = Path(path) if path is not None else _DEFAULT_CFG
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, Mapping):
        raise ValueError(f"V3a config must be a mapping: {cfg_path}")
    return dict(cfg)
