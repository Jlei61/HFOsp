"""Observed-support and horizon-specific split helpers for long future blocks.

Long future blocks are defined in wall-clock time but scored only on time that
was genuinely observable.  A recording gap is therefore neither an event nor
evidence of silence.  Short interruptions can be bridged for *state carry*
without pretending that their missing seconds were observed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MergeAudit:
    original_segments: int
    merged_segments: int
    merged_artificial_cuts: int
    merged_gap_seconds: float
    protected_seizure_gaps: int


@dataclass(frozen=True)
class LongSplitPlan:
    status: str
    horizon_seconds: float
    boundaries: dict[str, float] | None
    exposure_seconds: dict[str, float]
    wall_seconds: dict[str, float]
    requirements_seconds: dict[str, float]
    reason: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _normalise_segments(segments: np.ndarray) -> np.ndarray:
    value = np.asarray(segments, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError("segments must have shape (n,2)")
    value = value[np.argsort(value[:, 0], kind="stable")]
    if value.size and (not np.isfinite(value).all() or np.any(value[:, 1] <= value[:, 0])):
        raise ValueError("segments must be finite positive intervals")
    if value.shape[0] > 1 and np.any(value[1:, 0] < value[:-1, 1] - 1e-6):
        raise ValueError("segments overlap")
    return value


def merge_artificial_cuts(
    segments: np.ndarray,
    seizures: Sequence[Mapping[str, object]] = (),
    *,
    max_gap_seconds: float = 60.0,
) -> tuple[np.ndarray, MergeAudit]:
    """Join short non-seizure cuts into a state-carry interval.

    The returned intervals define where a slow state may propagate without a
    reset.  They MUST NOT replace the original observed-support intervals when
    computing exposure: every missing second remains zero-weight no-event
    evidence even when the state is carried across it.
    """

    source = _normalise_segments(segments)
    if source.shape[0] < 2 or max_gap_seconds <= 0:
        return source.copy(), MergeAudit(source.shape[0], source.shape[0], 0, 0.0, 0)
    seizure_intervals = []
    for row in seizures:
        onset = float(row["onset_epoch"])
        offset = float(row.get("offset_epoch", onset))
        seizure_intervals.append((onset, max(onset, offset)))

    merged: list[list[float]] = [[float(source[0, 0]), float(source[0, 1])]]
    n_merge = 0
    seconds = 0.0
    protected = 0
    for lo_raw, hi_raw in source[1:]:
        lo, hi = float(lo_raw), float(hi_raw)
        gap_lo, gap_hi = merged[-1][1], lo
        gap = max(0.0, gap_hi - gap_lo)
        seizure_inside = any(
            onset < gap_hi + 1e-6 and offset > gap_lo - 1e-6
            for onset, offset in seizure_intervals
        )
        if gap <= max_gap_seconds + 1e-9 and not seizure_inside:
            merged[-1][1] = max(merged[-1][1], hi)
            n_merge += 1
            seconds += gap
        else:
            if gap <= max_gap_seconds + 1e-9 and seizure_inside:
                protected += 1
            merged.append([lo, hi])
    out = np.asarray(merged, dtype=np.float64)
    return out, MergeAudit(source.shape[0], out.shape[0], n_merge, seconds, protected)


def exposure_seconds(segments: np.ndarray, lo: float, hi: float) -> float:
    """Return observed seconds in ``[lo, hi)``."""

    if hi <= lo:
        return 0.0
    support = _normalise_segments(segments)
    return float(np.maximum(0.0, np.minimum(support[:, 1], hi) - np.maximum(support[:, 0], lo)).sum())


def exposure_and_gap_count(
    segments: np.ndarray, starts: np.ndarray, stops: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised audit fields for future windows.

    ``gap_count`` counts exposure holes between observed pieces touched by the
    window.  It is descriptive only and never enters a predictor.
    """

    support = _normalise_segments(segments)
    start = np.asarray(starts, dtype=np.float64)
    stop = np.asarray(stops, dtype=np.float64)
    if start.shape != stop.shape:
        raise ValueError("starts and stops must have identical shape")
    exposure = np.zeros(start.shape, dtype=np.float64)
    pieces = np.zeros(start.shape, dtype=np.int64)
    for lo, hi in support:
        overlap = np.maximum(0.0, np.minimum(stop, hi) - np.maximum(start, lo))
        exposure += overlap
        pieces += overlap > 0
    return exposure, np.maximum(0, pieces - 1)


def _boundary_before(
    segments: np.ndarray, *, hi: float, floor: float, required_exposure: float,
) -> float | None:
    remaining = float(required_exposure)
    support = _normalise_segments(segments)
    for lo_raw, end_raw in support[::-1]:
        lo = max(float(lo_raw), float(floor))
        end = min(float(end_raw), float(hi))
        if end <= lo:
            continue
        width = end - lo
        if width >= remaining - 1e-9:
            return float(end - remaining)
        remaining -= width
    return None


def plan_horizon_specific_split(
    segments: np.ndarray,
    legacy_bounds: Mapping[str, float],
    horizon_seconds: float,
    *,
    inner_horizons: float = 2.0,
    selection_horizons: float = 3.0,
    minimum_fit_horizons: float = 4.0,
) -> LongSplitPlan:
    """Carve a long-horizon INNER and holdout from the safe <80% prefix.

    Boundaries are chosen by *observed exposure*, not by event count or raw
    wall span.  The final holdout contains at least three horizon-equivalents
    of observed time; the preceding INNER contains two.  No development or
    sealed time is opened.
    """

    horizon = float(horizon_seconds)
    if horizon <= 0:
        raise ValueError("horizon_seconds must be positive")
    support = _normalise_segments(segments)
    safe_lo = float(legacy_bounds["20pct"])
    safe_hi = float(legacy_bounds["80pct"])
    requirements = {
        "FIT": minimum_fit_horizons * horizon,
        "INNER": inner_horizons * horizon,
        "SELECTION": selection_horizons * horizon,
    }
    selection_lo = _boundary_before(
        support, hi=safe_hi, floor=safe_lo, required_exposure=requirements["SELECTION"]
    )
    if selection_lo is None:
        return LongSplitPlan("NOT_ESTIMABLE", horizon, None, {}, {}, requirements,
                             "safe prefix lacks the required long holdout exposure")
    inner_lo = _boundary_before(
        support, hi=selection_lo, floor=safe_lo, required_exposure=requirements["INNER"]
    )
    if inner_lo is None:
        return LongSplitPlan("NOT_ESTIMABLE", horizon, None, {}, {}, requirements,
                             "safe prefix lacks the required INNER exposure before holdout")
    fit_exposure = exposure_seconds(support, safe_lo, inner_lo)
    if fit_exposure + 1e-9 < requirements["FIT"]:
        return LongSplitPlan("NOT_ESTIMABLE", horizon, None,
                             {"FIT": fit_exposure}, {}, requirements,
                             "safe prefix lacks four horizon-equivalents for fitting")
    boundaries = {
        "20pct": safe_lo,
        "60pct": float(inner_lo),
        "70pct": float(selection_lo),
        "80pct": safe_hi,
    }
    exposure = {
        "FIT": exposure_seconds(support, safe_lo, inner_lo),
        "INNER": exposure_seconds(support, inner_lo, selection_lo),
        "SELECTION": exposure_seconds(support, selection_lo, safe_hi),
    }
    wall = {
        "FIT": inner_lo - safe_lo,
        "INNER": selection_lo - inner_lo,
        "SELECTION": safe_hi - selection_lo,
    }
    return LongSplitPlan("ESTIMABLE", horizon, boundaries, exposure, wall, requirements)


def phase_for_times(times: np.ndarray, bounds: Mapping[str, float]) -> np.ndarray:
    out = np.full(np.asarray(times).shape, "OUTSIDE", dtype="<U12")
    value = np.asarray(times, dtype=np.float64)
    out[value < float(bounds["20pct"])] = "CALIBRATION"
    out[(value >= float(bounds["20pct"])) & (value < float(bounds["60pct"]))] = "FIT"
    out[(value >= float(bounds["60pct"])) & (value < float(bounds["70pct"]))] = "INNER"
    out[(value >= float(bounds["70pct"])) & (value < float(bounds["80pct"]))] = "SELECTION"
    return out


def matched_wrong_time_donors(
    times: np.ndarray,
    target_rows: np.ndarray,
    donor_rows: np.ndarray,
    *,
    minimum_time_separation: float,
    recent_rate: np.ndarray,
    exposure_fraction: np.ndarray,
    n_donors: int = 5,
    clock_tolerance_seconds: float = 7200.0,
) -> np.ndarray:
    """Choose 5--10 deterministic same-patient wrong-time donors per anchor.

    Matching retains coarse clock time, current short-scale event rate and
    realised observation exposure.  The time separation destroys the precise
    state/future pairing.  Donor selection uses no future target value.
    """

    time = np.asarray(times, dtype=np.float64)
    targets = np.asarray(target_rows, dtype=np.int64)
    donors = np.asarray(donor_rows, dtype=np.int64)
    rate = np.asarray(recent_rate, dtype=np.float64)
    coverage = np.asarray(exposure_fraction, dtype=np.float64)
    if not 5 <= n_donors <= 10:
        raise ValueError("n_donors must be between 5 and 10")
    out = np.full((targets.size, n_donors), -1, dtype=np.int64)
    rate_scale = max(float(np.nanstd(rate[donors])), 1e-6) if donors.size else 1.0
    for i, row in enumerate(targets):
        delta = np.abs(time[donors] - time[row])
        clock = np.abs((time[donors] - time[row]) % 86400.0)
        clock = np.minimum(clock, 86400.0 - clock)
        eligible = (
            (delta >= float(minimum_time_separation))
            & (clock <= float(clock_tolerance_seconds))
            & np.isfinite(rate[donors]) & np.isfinite(coverage[donors])
        )
        candidates = donors[eligible]
        if candidates.size < n_donors:
            continue
        score = (
            np.abs(rate[candidates] - rate[row]) / rate_scale
            + 2.0 * np.abs(coverage[candidates] - coverage[row])
            + 0.1 * np.minimum(
                np.abs((time[candidates] - time[row]) % 86400.0),
                86400.0 - np.abs((time[candidates] - time[row]) % 86400.0),
            ) / max(clock_tolerance_seconds, 1.0)
        )
        order = np.lexsort((candidates, score))[:n_donors]
        out[i] = candidates[order]
    return out
