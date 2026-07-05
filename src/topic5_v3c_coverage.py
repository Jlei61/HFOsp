"""Topic 5 V3c — label-space set operations + spatial nulls (PURE, no I/O).

Set language (spec §2): A = interictal axis contacts; S = clinical SOZ ∩ pool.
This module never touches time; latency lives in topic5_v3c_latency.py.
"""
from __future__ import annotations

import numpy as np


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
