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


def load_topic5_subtype_labels(
    subject: str,
    band: str,
    results_root: Path,
) -> Dict[str, Any]:
    """Load per-seizure subtype label from topic5 PR-1 z-ER cluster JSON.

    Parameters
    ----------
    subject : str
        Numeric epilepsiae id without prefix, e.g. "442".
    band : str
        Either "gamma_ER" or "broad_ER".
    results_root : Path
        Project results root, typically Path("results") relative to repo root.

    Returns dict with keys:
      - seizure_id_to_subtype : Dict[str, int]   subtype labels (-1 = outlier)
      - n_subtypes : int                          per_band[band]["n_subtypes"]
      - status     : str                          "ok" / "insufficient_n" / ...
    """
    json_path = (
        results_root
        / "data_driven_soz"
        / "layer_a_ictal_er_rank"
        / "seizure_clusters"
        / "per_subject"
        / f"epilepsiae_{subject}__zer_binned.json"
    )
    if not json_path.exists():
        raise FileNotFoundError(f"topic5 PR-1 JSON missing: {json_path}")
    with json_path.open() as fh:
        d = json.load(fh)
    band_d = d["per_band"][band]
    seizure_ids = list(band_d["seizure_ids_kept"])
    labels = list(band_d["subtype_label"])
    if len(seizure_ids) != len(labels):
        raise ValueError(
            f"length mismatch in {json_path}: "
            f"{len(seizure_ids)} ids vs {len(labels)} labels"
        )
    return {
        "seizure_id_to_subtype": dict(zip(seizure_ids, [int(x) for x in labels])),
        "n_subtypes": int(band_d["n_subtypes"]),
        "status": str(band_d["status"]),
    }
