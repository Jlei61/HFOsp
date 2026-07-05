"""Topic 5 V3c — label-space set operations + spatial nulls (PURE, no I/O).

Set language (spec §2): A = interictal axis contacts; S = clinical SOZ ∩ pool.
This module never touches time; latency lives in topic5_v3c_latency.py.
"""
from __future__ import annotations

import numpy as np

from src.topic5_v3_mode_transition import _coerce_rng, label_permute


def coverage_metrics(axis_names: list, soz_names: list) -> dict:
    """Coverage of clinical SOZ S by interictal axis A, plus surplus/jaccard.

    coverage = |A∩S|/|S| (sensitivity); surplus_fraction = |A∖S|/|A| (spec R1:
    near-mechanical for fixed |A|, descriptor only); jaccard = |A∩S|/|A∪S|.
    """
    A = set(axis_names)
    S = set(soz_names)
    covered = sorted(A & S)
    surplus = sorted(A - S)
    missed = sorted(S - A)
    union = A | S
    n_a, n_s = len(A), len(S)
    return {
        "coverage": (len(covered) / n_s) if n_s else float("nan"),
        "surplus_fraction": (len(surplus) / n_a) if n_a else float("nan"),
        "jaccard": (len(covered) / len(union)) if union else float("nan"),
        "n_axis": n_a, "n_soz": n_s,
        "n_covered": len(covered), "n_surplus": len(surplus), "n_missed": len(missed),
        "covered": covered, "surplus": surplus, "missed": missed,
    }


def coverage_null_distribution(
    axis_names: list, all_clean: list, soz_names: list, shaft_by_name: dict,
    *, n_perm: int, rng,
) -> np.ndarray:
    """Same-shaft null: reshuffle the axis label within shafts across all clean
    contacts (preserves |A| and per-shaft axis count), recompute coverage of the
    FIXED soz set. Controls implant geometry (spec §4.2 primary null; R2: proves
    'beyond geometry', not 'beyond HFO-rich' — that needs the rate-matched null).
    """
    rng = _coerce_rng(rng)
    S = set(soz_names)
    n_s = len(S)
    axis_set = set(axis_names)
    nonaxis = [n for n in all_clean if n not in axis_set]
    out = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        new_axis, _ = label_permute(axis_names, nonaxis, shaft_by_name, rng)
        out[i] = (len(set(new_axis) & S) / n_s) if n_s else float("nan")
    return out
