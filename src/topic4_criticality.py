"""Topic 4 M3-v2.2 approach-criticality config loader (Task 0).

Loads the config-of-record `config/topic4_criticality.yaml`: operator units,
verdict thresholds + threshold-sweep, quality-gate floors, branching policy,
mode-selection policy, finite-time-gain horizons, the slow_to_ratefield entry
terminology lock, slow_sensitivity finite-difference steps, atlas grid, and
the virtual_seeg estimator-reuse contract.

This module will be heavily extended by later tasks (spec
docs/superpowers/specs/2026-07-02-topic4-m3v2-2-approach-criticality-design.md);
kept to the config loader only for now.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "topic4_criticality.yaml"


def load_crit_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load the topic4 criticality config YAML as a dict.

    path=None resolves to config/topic4_criticality.yaml relative to the repo root.
    """
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
