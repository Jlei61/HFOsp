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


def _shaft_and_num(name):
    num = "".join(c for c in name if c.isdigit())
    return name[: len(name) - len(num)], (int(num) if num else -1)


def _gini(counts) -> float:
    x = np.sort(np.asarray(counts, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.sum((np.arange(1, n + 1)) * x) - (n + 1) * x.sum()) / (n * x.sum()))


def _mean_min_dist(surplus_names, soz_names, coords_by_name) -> float:
    sc = [coords_by_name[n] for n in soz_names if n in coords_by_name]
    su = [coords_by_name[n] for n in surplus_names if n in coords_by_name]
    if not sc or not su:
        return float("nan")
    sc = np.vstack(sc)
    return float(np.mean([np.min(np.linalg.norm(sc - p[None, :], axis=1)) for p in su]))


def surplus_spatial_metrics(surplus_names, soz_names, coords_by_name, shaft_by_name) -> dict:
    per_shaft = {}
    for n in surplus_names:
        per_shaft.setdefault(shaft_by_name[n], []).append(_shaft_and_num(n)[1])
    max_run = 0
    for nums in per_shaft.values():
        s = sorted(x for x in nums if x >= 0)
        run = best = 1 if s else 0
        for a, b in zip(s, s[1:]):
            run = run + 1 if b == a + 1 else 1
            best = max(best, run)
        max_run = max(max_run, best)
    return {
        "n_shafts_with_surplus": len(per_shaft),
        "shaft_gini": _gini([len(v) for v in per_shaft.values()]),
        "max_contiguous_run": int(max_run),
        "mean_min_dist_to_soz": _mean_min_dist(surplus_names, soz_names, coords_by_name),
    }


def distance_null_distribution(surplus_names, axis_names, soz_names, coords_by_name,
                               shaft_by_name, *, n_perm, rng) -> np.ndarray:
    if not coords_by_name or not any(n in coords_by_name for n in soz_names):
        return np.array([])
    rng = _coerce_rng(rng)
    covered = [n for n in axis_names if n not in set(surplus_names)]
    out = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        new_surplus, _ = label_permute(surplus_names, covered, shaft_by_name, rng)
        out[i] = _mean_min_dist(new_surplus, soz_names, coords_by_name)
    return out
