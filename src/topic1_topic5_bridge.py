"""Topic 1 × Topic 5 Bridge — Q1 + Q1b + Q3 implementation.

See docs/superpowers/specs/2026-05-10-topic1-topic5-bridge-design.md.
Q2 is deferred and NOT implemented here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --- Locked constants (see spec §4) -----------------------------------------

ALPHA_WITHIN: float = 0.0167          # α/3 within-subject Bonferroni for 3 features
EFFECT_MIN: float = 0.10              # |ε²| or |r| or Cramér V threshold
P_NULL_BINOMIAL: float = 0.049        # cohort binomial null upper bound
WINDOWS_MIN: List[Tuple[float, float]] = [(-15.0, -1.0), (-30.0, -1.0), (-60.0, -1.0)]
PRIMARY_WINDOW: Tuple[float, float] = (-30.0, -1.0)

COHORT_GAMMA: List[str] = [
    "1073", "1096", "1146", "253", "548",
    "590", "635", "916", "922", "958",
]
SENTINEL_442: str = "442"            # Q1b binary-outlier
SENSITIVITY_BROAD_1084: str = "1084" # broad-band sensitivity
