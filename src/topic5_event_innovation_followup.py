"""Descriptive follow-up helpers for the frozen Topic 5 V3.0 event-innovation line.

Nothing here re-fits or re-scores the frozen human test.  Two independent
questions are served:

1. **Coverage / timescale census** — why only 17 of 34 patients produced a valid
   innovation, and whether any patient even has the multi-recording structure a
   future cross-recording contract would need.  This is inventory over the frozen
   event streams, not a hypothesis test.
2. **Cohort-stage detectability floor** — given the observed patient-to-patient
   scatter and n=17, how large a true median effect would the frozen cohort rule
   have caught?  This can only bound an already-frozen negative; it cannot raise
   the evidence level.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon


MINIMUM_PATIENTS_FOR_FLOOR = 3


def source_spans(
    abs_time: Sequence[float],
    source_index: Sequence[int],
    record_name: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return one row per continuity unit, ordered by first event time.

    Ordering is by start time rather than by source label because the label is a
    storage index and carries no chronology.
    """

    times = np.asarray(abs_time, dtype=float)
    # Segment labels are recording names (strings), not row indices.
    sources = np.asarray(source_index).astype(str)
    if times.shape != sources.shape:
        raise ValueError("abs_time and source_index must have the same shape")
    if record_name is not None:
        records = np.asarray(record_name)
        if records.shape != times.shape:
            raise ValueError("record_name must have the same shape as abs_time")
    else:
        records = None

    rows = []
    for label in np.unique(sources):
        mask = sources == label
        unit_times = times[mask]
        row: dict[str, Any] = {
            "source_index": str(label),
            "n_events": int(mask.sum()),
            "t_start": float(unit_times.min()),
            "t_end": float(unit_times.max()),
            "duration_seconds": float(unit_times.max() - unit_times.min()),
        }
        if records is not None:
            names = sorted(set(records[mask].tolist()))
            row["record_name"] = (
                names[0] if len(names) == 1 else "MIXED(" + "|".join(names) + ")"
            )
        rows.append(row)
    rows.sort(key=lambda row: (row["t_start"], row["source_index"]))
    return rows


def source_gap_census(
    spans: Sequence[Mapping[str, Any]],
    *,
    min_gap_seconds: float,
    min_events_per_side: int,
) -> dict[str, Any]:
    """Summarise the between-continuity-unit gap structure of one patient.

    A gap "qualifies" only when both flanking units independently carry enough
    events to estimate a state, so a long silence next to a three-event stub does
    not count as usable cross-recording structure.
    """

    ordered = sorted(spans, key=lambda row: (row["t_start"], row["source_index"]))
    n_sources = len(ordered)
    gaps = []
    qualifying = []
    for left, right in zip(ordered, ordered[1:]):
        gap = float(right["t_start"] - left["t_end"])
        gaps.append(gap)
        if (
            gap >= float(min_gap_seconds)
            and int(left["n_events"]) >= int(min_events_per_side)
            and int(right["n_events"]) >= int(min_events_per_side)
        ):
            qualifying.append(gap)
    event_counts = [int(row["n_events"]) for row in ordered]
    return {
        "n_sources": n_sources,
        "n_events_total": int(sum(event_counts)),
        "median_events_per_source": (
            float(np.median(event_counts)) if event_counts else None
        ),
        "max_events_per_source": max(event_counts) if event_counts else None,
        "n_consecutive_gaps": len(gaps),
        "n_qualifying_consecutive_gaps": len(qualifying),
        "max_gap_seconds": max(gaps) if gaps else None,
        "median_gap_seconds": float(np.median(gaps)) if gaps else None,
        "max_qualifying_gap_seconds": max(qualifying) if qualifying else None,
        "total_span_seconds": (
            float(ordered[-1]["t_end"] - ordered[0]["t_start"]) if ordered else None
        ),
        "cross_gap_eligible": bool(qualifying),
    }


def _wilcoxon_two_sided(values: np.ndarray) -> float:
    try:
        return float(wilcoxon(values, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def detectability_floor(
    effects: Sequence[float],
    *,
    deltas: Sequence[float],
    n_draws: int,
    seed: int,
    alpha: float = 0.05,
    smooth: bool = False,
) -> dict[str, Any]:
    """Power of the frozen cohort decision against a shifted patient effect.

    The observed patient effects are centred on their own median to build a
    null-shaped scatter, then re-drawn with a true shift `delta` added.  A draw
    counts as detected when its median is positive **and** the two-sided Wilcoxon
    p-value clears `alpha` — the same conjunction the frozen V3.0 rule applies to
    a route's primary gain.

    Bootstrap resampling duplicates values, which makes SciPy fall back from the
    exact Wilcoxon to the tie-corrected asymptotic form.  The tie fraction is
    reported per delta, and `smooth=True` adds kernel jitter so the exact test is
    used throughout; agreement between the two modes is the robustness check.
    """

    observed = np.asarray(effects, dtype=float)
    observed = observed[np.isfinite(observed)]
    if observed.size < MINIMUM_PATIENTS_FOR_FLOOR:
        raise ValueError(
            f"detectability floor needs at least {MINIMUM_PATIENTS_FOR_FLOOR} patients"
        )
    n = observed.size
    residual = observed - np.median(observed)
    scatter = float(np.std(residual, ddof=1))
    bandwidth = 0.5 * 1.06 * scatter * n ** (-0.2) if smooth else 0.0

    rng = np.random.default_rng(int(seed))
    curve = []
    for delta in deltas:
        drawn = rng.choice(residual, size=(int(n_draws), n), replace=True)
        if bandwidth > 0.0:
            drawn = drawn + rng.normal(scale=bandwidth, size=drawn.shape)
        drawn = drawn + float(delta)
        detected = 0
        tied = 0
        for row in drawn:
            if np.unique(np.abs(row)).size < n or np.any(row == 0.0):
                tied += 1
            if np.median(row) > 0.0 and _wilcoxon_two_sided(row) <= float(alpha):
                detected += 1
        curve.append(
            {
                "delta": float(delta),
                "power": detected / float(n_draws),
                "tie_fraction": tied / float(n_draws),
            }
        )

    delta80 = next(
        (row["delta"] for row in curve if row["power"] >= 0.80),
        None,
    )
    return {
        "n_patients": int(n),
        "observed_median": float(np.median(observed)),
        "residual_sd": scatter,
        "alpha": float(alpha),
        "n_draws": int(n_draws),
        "smooth": bool(smooth),
        "kernel_bandwidth": float(bandwidth),
        "curve": curve,
        "delta80": delta80,
        "rule": (
            "a draw is detected when its median is positive and its two-sided "
            "Wilcoxon p-value is <= alpha"
        ),
    }
