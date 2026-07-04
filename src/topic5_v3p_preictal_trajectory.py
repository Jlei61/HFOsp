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

def _finite_pairs(y, t):
    y = np.asarray(y, float); t = np.asarray(t, float)
    m = np.isfinite(y) & np.isfinite(t)
    return y[m], t[m]

def theil_sen_slope(y, t) -> float:
    y, t = _finite_pairs(y, t)
    if y.size < 2 or np.unique(t).size < 2:
        return float("nan")
    return float(stats.theilslopes(y, t)[0])

def spearman_trend(y, t) -> float:
    y, t = _finite_pairs(y, t)
    if y.size < 3 or np.unique(y).size < 2 or np.unique(t).size < 2:
        return float("nan")
    return float(stats.spearmanr(y, t).correlation)

def slope_over_windows(values, centers, estimator) -> float:
    if estimator == "theil_sen":
        return theil_sen_slope(values, centers)
    if estimator == "spearman":
        return spearman_trend(values, centers)
    if estimator == "ols":
        y, t = _finite_pairs(values, centers)
        if y.size < 2 or np.unique(t).size < 2:
            return float("nan")
        return float(np.polyfit(t, y, 1)[0])
    raise ValueError(f"unknown estimator: {estimator!r}")
