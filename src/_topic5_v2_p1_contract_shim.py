"""TEMPORARY Phase-1 contract shim (autonomy hedge, 2026-07-01).

Phase 1 (`src/topic5_v2_band_scan.py`) is being built by a concurrent session.
This shim provides ONLY `contact_alignment`, copied VERBATIM from the Phase-1
plan's literal Task-5 code, so the Phase-2 run scripts can compute observed
alignment while Phase 1 is in flight. `contact_alignment` is a deterministic
Spearman/Pearson correlation → its output is IDENTICAL to the real Phase-1
function (both are the same plan code); results computed through it are FINAL,
not provisional.

The science-critical nulls (`spatial_constrained_permute`, `order_null_rank_pair`,
`rebuild_typical_rank`) are NOT shimmed — their construction is the contract and
has no literal code here, so they must come from the real Phase-1 module.

`scripts/_topic5_v2_crit_io.get_contact_alignment()` prefers the real Phase-1
`contact_alignment` and only falls back here, so this file self-retires once
Phase 1 lands Task 5.
"""
from __future__ import annotations

import numpy as np


def contact_alignment(vals_by_name, rank_a_by_name, rank_b_by_name, oriented_template="a"):
    """Verbatim Phase-1 Task-5 contract (fixed-orientation signed + per-template)."""
    from scipy.stats import spearmanr, pearsonr

    def _one(rank_by):
        names = [n for n in vals_by_name if n in rank_by
                 and np.isfinite(vals_by_name[n]) and np.isfinite(rank_by[n])]
        if len(names) < 4:
            return None
        v = np.array([vals_by_name[n] for n in names])
        r = np.array([rank_by[n] for n in names])
        if np.std(v) == 0 or np.std(r) == 0:
            return None
        return {"sp": float(spearmanr(v, r).statistic), "pe": float(pearsonr(v, r)[0]), "n": len(names)}

    a, b = _one(rank_a_by_name), _one(rank_b_by_name)

    def g(o, k, d=float("nan")):
        return o[k] if o else d

    posthoc = max([o for o in (a, b) if o], key=lambda o: abs(o["sp"]), default=None)
    return {
        "signed_pearson_a": g(a, "pe"), "signed_spearman_a": g(a, "sp"),
        "signed_pearson_b": g(b, "pe"), "signed_spearman_b": g(b, "sp"),
        "align_signed_oriented": (g(a, "sp") if oriented_template == "a" else g(b, "sp")),
        "align_signed_posthoc_max": (posthoc["sp"] if posthoc else float("nan")),
        "align_abs_maxab_contact": max([abs(o["sp"]) for o in (a, b) if o], default=float("nan")),
        "n_contacts_a": g(a, "n", 0), "n_contacts_b": g(b, "n", 0),
    }
