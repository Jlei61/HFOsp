"""Family-wise and false-discovery correction for the sign-test families.

Why this exists
---------------
The R0.1 review found 288 exact sign tests inside one analysis file, 48 of them
below 0.05, all reported as ``two_sided_exact_sign_p_unadjusted`` with no
adjusted companion anywhere in the package. The machine evidence card then
lifted one of them (p = 8.2e-4) into a headline position. That p is one of
eleven tied cells and ranks fourth of 288: Holm puts it at 0.234, Benjamini-
Hochberg at q = 0.017. Those two numbers support very different sentences, and
the package offered neither.

The project's own earlier artifact already did this properly --
``results/epi_prssm/v0_1/event_distribution/H2A_EVIDENCE_CARD.json`` carries a
``holm_corrected_primary_family`` block -- so shipping raw p-values here was a
regression against an established internal standard, not an open question.

Both corrections are reported, never one:

* **Holm** controls the family-wise error rate. It is the honest answer to
  "is this one cell significant on its own". It is very conservative here
  because the tests are strongly dependent (same 34 patients, six endpoints
  that share a joint likelihood, six tau values that are smooth in tau), and a
  sign test on 34 patients has a coarse p-value lattice, so many cells tie.
* **Benjamini-Hochberg** controls the false-discovery rate and is the more
  appropriate lens for a grid that is deliberately scanned. It is valid under
  positive dependence, which is what these families have.

Neither replaces the pre-registered reading. Where the spec froze a primary
window set in advance, the direction counts across that frozen set carry the
claim and these values are descriptive support.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def holm(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Holm step-down adjusted p-values, order preserved. ``None`` passes through."""
    idx = [i for i, p in enumerate(p_values) if p is not None]
    out: List[Optional[float]] = [None] * len(p_values)
    if not idx:
        return out
    p = np.asarray([float(p_values[i]) for i in idx], dtype=float)
    order = np.argsort(p, kind="stable")
    m = p.size
    running = 0.0
    adjusted = np.empty(m, dtype=float)
    for rank, position in enumerate(order):
        running = max(running, (m - rank) * p[position])
        adjusted[position] = min(running, 1.0)
    for slot, value in zip(idx, adjusted):
        out[slot] = float(value)
    return out


def benjamini_hochberg(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """BH adjusted p-values (q), order preserved. ``None`` passes through."""
    idx = [i for i, p in enumerate(p_values) if p is not None]
    out: List[Optional[float]] = [None] * len(p_values)
    if not idx:
        return out
    p = np.asarray([float(p_values[i]) for i in idx], dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.minimum(ranked, 1.0)
    for slot, value in zip(idx, adjusted):
        out[slot] = float(value)
    return out


def annotate_family(entries: Iterable[Tuple[Sequence[str], dict]],
                    *, raw_key: str = "two_sided_exact_sign_p_unadjusted",
                    family_name: str) -> Dict[str, object]:
    """Write Holm and BH values back into every summary dict of one family.

    ``entries`` yields ``(path, summary_dict)``; ``path`` is only used to make
    the returned index human-readable. Each summary gains ``holm_adjusted_p``,
    ``bh_adjusted_q`` and ``multiplicity_family`` so a downstream reader cannot
    pick up the raw value without also seeing what it costs.
    """
    items = [(tuple(path), summary) for path, summary in entries]
    raw = [summary.get(raw_key) for _, summary in items]
    holm_values = holm(raw)
    bh_values = benjamini_hochberg(raw)
    tested = [p for p in raw if p is not None]
    for (_, summary), h, q in zip(items, holm_values, bh_values):
        summary["multiplicity_family"] = family_name
        summary["holm_adjusted_p"] = h
        summary["bh_adjusted_q"] = q
    return {
        "family": family_name,
        "n_tests": len(tested),
        "n_raw_below_0p05": int(sum(1 for p in tested if p < 0.05)),
        "n_holm_below_0p05": int(sum(1 for p in holm_values if p is not None and p < 0.05)),
        "n_bh_q_below_0p05": int(sum(1 for q in bh_values if q is not None and q < 0.05)),
        "n_bh_q_below_0p10": int(sum(1 for q in bh_values if q is not None and q < 0.10)),
        "bonferroni_threshold": (0.05 / len(tested)) if tested else None,
        "note": (
            "Holm controls the family-wise error rate and is conservative here "
            "because the tests are strongly dependent and a 34-patient sign test "
            "has a coarse p lattice. BH controls the false-discovery rate and is "
            "the appropriate lens for a deliberately scanned grid. Where the spec "
            "froze a primary window set before results were seen, the direction "
            "counts over that frozen set carry the claim and these values are "
            "descriptive support."
        ),
    }
