"""Fixed physical-time risk grid for H2b B1 (discrete-time seizure survival).

Two coverage notions are deliberately kept apart, because collapsing them is
how "we stopped recording" silently becomes "no seizure happened":

``state_spans``
    where the *state* exists -- the group-event artifact coverage that the
    frozen producer can actually be propagated through. Anchors live here.
``monitoring_spans``
    where the patient was *observed* -- the recording coverage that makes a
    documented seizure time trustworthy. Outcomes are censored against this.

Contract clauses pinned by ``tests/test_topic5_h2b_risk_grid.py``:
D1 absolute reproducible grid; D2 anchors need state coverage; D3 no anchor in
an ictal interval; D4 parameterised postictal exclusion (60 min primary);
D5 pre-registered horizon bins; D6 censoring distinguished from beyond-horizon.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

HOUR = 3600.0
MINUTE = 60.0

#: D5 -- pre-registered upper edges of the discrete risk bins.
#: 0-5, 5-15, 15-30, 30-60 min, 1-2 h, 2-6 h; anything past the last edge is
#: "beyond horizon or right-censored".
HORIZON_EDGES_SECONDS: tuple[float, ...] = (
    5 * MINUTE,
    15 * MINUTE,
    30 * MINUTE,
    60 * MINUTE,
    2 * HOUR,
    6 * HOUR,
)

DEFAULT_GRID_SECONDS = 300.0
DEFAULT_POSTICTAL_EXCLUSION_SECONDS = 3600.0
#: Blocks whose seam is smaller than this are one continuous span
#: (v0.1 data contract §12 uses the same 2 s seam tolerance).
DEFAULT_SEAM_TOLERANCE_SECONDS = 2.0


@dataclass(frozen=True)
class RiskRow:
    subject: str
    anchor_epoch: float
    state_span_index: int
    seconds_into_state_span: float
    time_to_next_seizure_sec: float | None
    next_seizure_id: str | None
    outcome_bin: int | None
    censored: bool
    beyond_horizon: bool
    last_observed_bin: int
    observed_horizon_sec: float
    time_since_prev_seizure_sec: float | None
    prev_seizure_id: str | None


def merge_spans(
    spans: Iterable[tuple[float, float]],
    tolerance_seconds: float = DEFAULT_SEAM_TOLERANCE_SECONDS,
) -> tuple[tuple[float, float], ...]:
    """Merge intervals separated by no more than ``tolerance_seconds``."""

    ordered = sorted((float(a), float(b)) for a, b in spans if math.isfinite(a) and math.isfinite(b))
    if not ordered:
        return ()
    out: list[list[float]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - out[-1][1] <= tolerance_seconds:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return tuple((a, b) for a, b in out)


def _covered_until(spans: Sequence[tuple[float, float]], t: float) -> float:
    """Last instant reachable from ``t`` without leaving ``spans``.

    Returns ``t`` itself when ``t`` is not covered at all.
    """

    for start, end in spans:
        if start <= t <= end:
            return end
    return t


def _bin_of(delta: float) -> int | None:
    """Index of the horizon bin containing ``delta`` seconds, else ``None``."""

    for i, edge in enumerate(HORIZON_EDGES_SECONDS):
        if delta <= edge:
            return i
    return None


def _last_fully_observed_bin(observed: float) -> int:
    """Highest bin whose *entire* span was observed (-1 if none)."""

    last = -1
    for i, edge in enumerate(HORIZON_EDGES_SECONDS):
        if observed >= edge:
            last = i
    return last


def build_risk_rows(
    subject: str,
    state_spans: Iterable[tuple[float, float]],
    monitoring_spans: Iterable[tuple[float, float]],
    seizures: Sequence[Mapping[str, object]],
    grid_seconds: float = DEFAULT_GRID_SECONDS,
    postictal_exclusion_seconds: float = DEFAULT_POSTICTAL_EXCLUSION_SECONDS,
    seam_tolerance_seconds: float = DEFAULT_SEAM_TOLERANCE_SECONDS,
) -> list[RiskRow]:
    state = merge_spans(state_spans, seam_tolerance_seconds)
    monitoring = merge_spans(monitoring_spans, seam_tolerance_seconds)

    sz = sorted(
        (
            (float(s["onset_epoch"]), float(s["offset_epoch"]), str(s.get("seizure_id", "")))
            for s in seizures
        ),
        key=lambda t: t[0],
    )
    onsets = [o for o, _, _ in sz]
    offsets_sorted = sorted((off, sid) for _, off, sid in sz)
    offset_times = [o for o, _ in offsets_sorted]

    rows: list[RiskRow] = []
    for span_index, (span_start, span_end) in enumerate(state):
        # D1: anchors are multiples of the step on the absolute epoch axis, so
        # the grid is identical no matter which span or subject produced it.
        first = math.ceil(span_start / grid_seconds) * grid_seconds
        t = first
        while t <= span_end:
            keep = True
            for onset, offset, _sid in sz:
                # D3: no anchor inside an ictal interval.
                # D4: and none inside the postictal exclusion after it. The
                # window is half-open so a new segment resumes *at* offset +
                # exclusion, as "offset 后排除 60 min，再启动新 segment" reads.
                if onset <= t < offset + postictal_exclusion_seconds:
                    keep = False
                    break
            if keep:
                rows.append(
                    _make_row(
                        subject=subject,
                        t=t,
                        span_index=span_index,
                        span_start=span_start,
                        monitoring=monitoring,
                        onsets=onsets,
                        sz=sz,
                        offset_times=offset_times,
                        offsets_sorted=offsets_sorted,
                    )
                )
            t += grid_seconds
    return rows


def _make_row(
    *,
    subject: str,
    t: float,
    span_index: int,
    span_start: float,
    monitoring: Sequence[tuple[float, float]],
    onsets: Sequence[float],
    sz: Sequence[tuple[float, float, str]],
    offset_times: Sequence[float],
    offsets_sorted: Sequence[tuple[float, str]],
) -> RiskRow:
    # Next seizure onset strictly after the anchor.
    j = bisect_right(onsets, t)
    next_onset = onsets[j] if j < len(onsets) else None
    next_id = sz[j][2] if j < len(sz) else None

    # Previous seizure, by offset at or before the anchor.
    k = bisect_left(offset_times, t)
    prev_offset, prev_id = (offsets_sorted[k - 1] if k > 0 else (None, None))

    # D6: how far can we *trust* the record from this anchor?
    horizon_end = t + HORIZON_EDGES_SECONDS[-1]
    covered_to = _covered_until(monitoring, t)
    observed = max(0.0, min(covered_to, horizon_end) - t)
    last_bin = _last_fully_observed_bin(observed)

    delta = (next_onset - t) if next_onset is not None else None
    outcome_bin = _bin_of(delta) if delta is not None else None

    if outcome_bin is not None and delta is not None and delta <= observed:
        # The seizure happened inside the trustworthy window: a real event.
        censored, beyond = False, False
    else:
        outcome_bin = None
        # Beyond horizon only if the *whole* 6 h was actually monitored.
        beyond = observed >= HORIZON_EDGES_SECONDS[-1]
        censored = not beyond

    return RiskRow(
        subject=subject,
        anchor_epoch=t,
        state_span_index=span_index,
        seconds_into_state_span=t - span_start,
        time_to_next_seizure_sec=delta,
        next_seizure_id=next_id if outcome_bin is not None else None,
        outcome_bin=outcome_bin,
        censored=censored,
        beyond_horizon=beyond,
        last_observed_bin=last_bin,
        observed_horizon_sec=observed,
        time_since_prev_seizure_sec=(t - prev_offset) if prev_offset is not None else None,
        prev_seizure_id=prev_id,
    )


def lead_anchor_status(
    anchor_epoch: float,
    state_spans: Iterable[tuple[float, float]],
    seizures: Sequence[Mapping[str, object]],
    postictal_exclusion_seconds: float = DEFAULT_POSTICTAL_EXCLUSION_SECONDS,
    seam_tolerance_seconds: float = DEFAULT_SEAM_TOLERANCE_SECONDS,
) -> str:
    """Is the frozen state readable at an exact ``onset - lead`` instant?

    B2 reads leads at exact offsets rather than snapping to the 5-min grid. A
    missing anchor is reported as such and never back-filled with zeros
    ("6 h 仅在连续 coverage 存在时计入，不用缺失 anchor 补零", H2b spec §1).
    """

    t = float(anchor_epoch)
    spans = merge_spans(state_spans, seam_tolerance_seconds)
    if not any(start <= t <= end for start, end in spans):
        return "no_state_coverage"
    for s in seizures:
        onset = float(s["onset_epoch"])
        offset = float(s["offset_epoch"])
        if onset <= t <= offset:
            return "in_ictal"
        if offset < t < offset + postictal_exclusion_seconds:
            return "in_postictal"
    return "ok"


def group_seizure_episodes(
    seizures: Sequence[Mapping[str, object]],
    gap_seconds: float = DEFAULT_POSTICTAL_EXCLUSION_SECONDS,
) -> list[list[Mapping[str, object]]]:
    """Cluster seizures into independently-predictable episodes (D9).

    A seizure that starts before the previous one's postictal exclusion has
    elapsed joins that episode. Only the *lead* seizure of an episode can ever
    be preceded by a valid grid anchor, so episodes -- not raw seizure rows --
    are the independent unit for the held-out denominator.

    ``gap_seconds`` defaults to the postictal exclusion so the grouping and the
    anchor-dropping rule cannot disagree.
    """

    ordered = sorted(seizures, key=lambda s: float(s["onset_epoch"]))
    episodes: list[list[Mapping[str, object]]] = []
    for s in ordered:
        onset = float(s["onset_epoch"])
        if episodes and onset < float(episodes[-1][-1]["offset_epoch"]) + gap_seconds:
            episodes[-1].append(s)
        else:
            episodes.append([s])
    return episodes
