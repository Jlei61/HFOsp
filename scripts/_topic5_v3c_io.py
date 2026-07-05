"""Topic 5 V3c — SOZ join + latency-matrix IO (reuses V3a classifier)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v3_io import CACHE, classify_subject_contacts  # noqa: E402
from src.topic5_v3c_coverage import coverage_metrics  # noqa: E402

SOZ_JSON = {
    "epilepsiae": _ROOT / "results/epilepsiae_soz_core_channels.json",
    "yuquan": _ROOT / "results/yuquan_soz_core_channels.json",
}

# broad = broad-classifiable SOZ subjects (442/958 lack broad cache -> narrow only, spec §3.3)
V3C_SUBJECTS = {
    "broad": ["epilepsiae_139", "epilepsiae_253", "epilepsiae_635", "epilepsiae_1077",
              "epilepsiae_1096", "epilepsiae_1150", "epilepsiae_1146"],
    "narrow": ["epilepsiae_1096", "epilepsiae_1146", "epilepsiae_253",
               "epilepsiae_442", "epilepsiae_958"],
}


def load_soz(dataset: str, subject: str) -> list:
    """Clinical SOZ contact names for one subject; [] if the subject is absent."""
    path = SOZ_JSON[dataset]
    data = json.loads(path.read_text())
    return list(data.get(subject, []))


def axis_soz_join(cls: dict, soz_list: list) -> dict:
    """coverage_metrics(A, S) with S restricted to the all-clean pool; adds soz_in_pool."""
    pool = set(cls["all_clean"])
    soz_in_pool = [n for n in soz_list if n in pool]
    m = coverage_metrics(cls["is_axis"], soz_in_pool)
    m["soz_in_pool"] = soz_in_pool
    return m
