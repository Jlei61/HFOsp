from __future__ import annotations
from pathlib import Path
import numpy as np, yaml
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CFG = _ROOT / "config/topic5_v2_phase1.yaml"
def load_phase1_config(path=None) -> dict:
    with open(path or _DEFAULT_CFG) as fh:
        return yaml.safe_load(fh)
