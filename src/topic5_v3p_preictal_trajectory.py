"""Topic 5 V3p — preictal-only non-axial trajectory (pure math).

READ-ONLY reuse of the V3a stack (src.topic5_v3_mode_transition,
scripts._topic5_v3_io). V3p adds only the preictal-restriction + slope +
residualization + N->N self-sustain layer. See
docs/superpowers/plans/2026-07-03-topic5-v3p-preictal-trajectory.md.
Exploratory; no forecasting.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Mapping, Sequence
import numpy as np
import yaml
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CFG = _ROOT / "config" / "topic5_v3p.yaml"

def load_v3p_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path is not None else _DEFAULT_CFG
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, Mapping):
        raise ValueError(f"V3p config must be a mapping: {cfg_path}")
    return dict(cfg)
