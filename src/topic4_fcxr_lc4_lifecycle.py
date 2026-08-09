"""Pure adjudication for the FCXR-LC4 continuous lifecycle gate.

The gate deliberately keeps three claims separate:

* an ictal-strength bout ended inside the record;
* returning events in a fixed, late eight-second window match the frozen interictal reference;
* the exact spatial D field at the end of that accepted return remains stable when frozen.

The third claim needs a second simulation and is therefore never inferred here from a scalar D
mean.  This module only decides whether the nominal trajectory is eligible for that confirmation,
and whether the stored confirmation itself passed.
"""
from __future__ import annotations

import numpy as np

from src.topic4_fcxr_lc3_stage import returned_to_reference


SCHEMA = "fcxr-lc4-lifecycle-gate-1.0"
THRESHOLDS = dict(
    PRE_MS=8000.0,
    ICTAL_MIN_MS=1000.0,
    ICTAL_MAX_MS=5000.0,
    RELAPSE_GUARD_MS=2000.0,
    RETURN_MS=8000.0,
    MIN_RETURNING_EVENTS=3,
    REFRACTORY_CEILING_FRAC_MAX=0.01,
)


def _smooth_isolated(regimes):
    """Use the frozen LC1 convention: isolated non-ictal glitches do not split baseline.

    ICTAL is never smoothed.  Keeping this small copy local avoids changing the already executed
    LC1 classifier while preserving its exact interpretation of an interictal neighbourhood.
    """

    out = list(regimes)
    for i in range(1, len(out) - 1):
        if (regimes[i] not in ("ICTAL", "INTERICTAL")
                and regimes[i - 1] == "INTERICTAL"
                and regimes[i + 1] == "INTERICTAL"):
            out[i] = "INTERICTAL"
    return out


def first_ictal_bout(regimes, win_ms, *, min_ms=1000.0):
    """Return the first maximal contiguous ICTAL bout long enough to qualify."""

    need = max(1, int(np.ceil(float(min_ms) / float(win_ms))))
    i = 0
    while i < len(regimes):
        if regimes[i] != "ICTAL":
            i += 1
            continue
        j = i + 1
        while j < len(regimes) and regimes[j] == "ICTAL":
            j += 1
        if j - i >= need:
            return i, j - 1
        i = j
    return None


def refractory_ceiling_fraction(spikes, *, dt_ms, onset_ms, offset_ms,
                                tau_ref_ms, ceiling_fraction=0.9):
    """Fraction of E cells firing near their refractory-limited ceiling in the bout."""

    x = np.asarray(spikes, bool)
    if x.ndim != 2:
        raise ValueError("spikes must be time x E-cell")
    i0 = max(0, int(np.floor(float(onset_ms) / float(dt_ms))))
    i1 = min(x.shape[0], int(np.ceil(float(offset_ms) / float(dt_ms))))
    if i1 <= i0:
        raise ValueError("empty ictal interval")
    duration_s = (i1 - i0) * float(dt_ms) / 1000.0
    hz = x[i0:i1].sum(axis=0) / duration_s
    ceiling_hz = 1000.0 / float(tau_ref_ms)
    return float(np.mean(hz >= float(ceiling_fraction) * ceiling_hz))


def _event_view(events, start_ms, end_ms):
    return [e for e in events
            if bool(e.get("returned", False))
            and float(e["t_on"]) >= float(start_ms)
            and float(e["t_on"]) < float(end_ms)]


def _reference_check(events, start_ms, end_ms, band, *, min_events):
    chosen = _event_view(events, start_ms, end_ms)
    duration_s = (float(end_ms) - float(start_ms)) / 1000.0
    base = returned_to_reference(
        n_returning_after_offset=len(chosen),
        event_rate_hz=(len(chosen) / duration_s if duration_s > 0 else 0.0),
        band=band,
        durations_ms=[float(e["dur_ms"]) for e in chosen],
        participation=[float(e["peak_ext"]) for e in chosen],
    )
    enough = len(chosen) >= int(min_events)
    return dict(
        passed=bool(enough and base["returned"]),
        enough_events=enough,
        window_start_ms=float(start_ms),
        window_end_ms=float(end_ms),
        n_returning=len(chosen),
        event_indices=[int(e.get("index", i + 1)) for i, e in enumerate(chosen)],
        reference=base,
    )


def adjudicate_nominal(*, regimes, win_ms, events, total_ms, reference_band,
                       numerical_safe, refractory_fraction,
                       pre_rate_hz, postictal_rate_hz, T=THRESHOLDS):
    """Adjudicate the single 70 s no-kick trajectory up to the frozen-D confirmation.

    The return window is the final eight seconds, fixed independently of the data.  This prevents
    selecting a flattering transient interval after seeing the trace.  It must begin after the
    two-second relapse/protection guard and consist of the interictal regime after applying only
    the frozen LC1 isolated-window smoothing rule.
    """

    win_ms = float(win_ms)
    total_ms = float(total_ms)
    raw = list(regimes)
    sm = _smooth_isolated(raw)
    bout = first_ictal_bout(sm, win_ms, min_ms=T["ICTAL_MIN_MS"])
    if bout is None:
        return dict(schema=SCHEMA, verdict="F2_NO_QUALIFYING_ICTAL_BOUT", passed=False,
                    eligible_for_frozen_D=False, clauses={"ictal_bout": False}, bout=None)

    b0, b1 = bout
    onset_ms = b0 * win_ms
    bout_end_ms = (b1 + 1) * win_ms
    bout_ms = bout_end_ms - onset_ms
    ended_inside_record = b1 + 1 < len(sm)
    offset_ms = bout_end_ms if ended_inside_record else None

    pre_n = 0
    for r in reversed(sm[:b0]):
        if r != "INTERICTAL":
            break
        pre_n += 1
    pre_ms = pre_n * win_ms
    pre_events = _event_view(events, 0.0, onset_ms)

    guard_n = max(1, int(np.ceil(T["RELAPSE_GUARD_MS"] / win_ms)))
    guard = sm[b1 + 1:min(len(sm), b1 + 1 + guard_n)] if ended_inside_record else []
    guard_observed = len(guard) == guard_n
    no_rapid_relapse = bool(guard_observed and "ICTAL" not in guard)
    postictal_suppressed = bool(no_rapid_relapse
                                and np.isfinite(pre_rate_hz)
                                and np.isfinite(postictal_rate_hz)
                                and float(postictal_rate_hz) < float(pre_rate_hz))

    tail_start = total_ms - float(T["RETURN_MS"])
    tail_i0 = int(round(tail_start / win_ms))
    tail_regimes = sm[tail_i0:] if tail_i0 >= 0 else []
    tail_after_guard = bool(offset_ms is not None
                            and tail_start >= offset_ms + T["RELAPSE_GUARD_MS"])
    tail_is_interictal = bool(tail_regimes and len(tail_regimes) * win_ms >= T["RETURN_MS"]
                              and all(r == "INTERICTAL" for r in tail_regimes))
    ret = _reference_check(events, tail_start, total_ms, reference_band,
                           min_events=T["MIN_RETURNING_EVENTS"])

    clauses = dict(
        numerical_safe=bool(numerical_safe),
        pre_interictal_ms=bool(pre_ms >= T["PRE_MS"]),
        pre_returning_events=bool(len(pre_events) >= T["MIN_RETURNING_EVENTS"]),
        bounded_duration=bool(T["ICTAL_MIN_MS"] <= bout_ms <= T["ICTAL_MAX_MS"]),
        autonomous_offset=bool(ended_inside_record),
        no_rapid_relapse=no_rapid_relapse,
        postictal_suppression=postictal_suppressed,
        not_refractory_plateau=bool(float(refractory_fraction)
                                    <= T["REFRACTORY_CEILING_FRAC_MAX"]),
        return_window_after_guard=tail_after_guard,
        return_window_interictal=tail_is_interictal,
        returning_reference=bool(ret["passed"]),
    )
    passed = bool(all(clauses.values()))
    return dict(
        schema=SCHEMA,
        verdict=("F2_NOMINAL_ELIGIBLE_FOR_FROZEN_D" if passed
                 else "F2_NOMINAL_LIFECYCLE_INCOMPLETE"),
        passed=passed,
        eligible_for_frozen_D=passed,
        clauses=clauses,
        bout=[int(b0), int(b1)],
        onset_ms=float(onset_ms),
        offset_ms=(None if offset_ms is None else float(offset_ms)),
        bout_ms=float(bout_ms),
        pre_ms=float(pre_ms),
        n_returning_before_onset=len(pre_events),
        pre_rate_hz=float(pre_rate_hz),
        postictal_rate_hz=float(postictal_rate_hz),
        refractory_ceiling_fraction=float(refractory_fraction),
        return_window=ret,
    )


def adjudicate_frozen_D(*, regimes, win_ms, events, total_ms, burn_ms,
                        reference_band, numerical_safe, refractory_fraction,
                        T=THRESHOLDS):
    """Decide the 12 s continuation with the exact final spatial D field frozen."""

    sm = _smooth_isolated(list(regimes))
    bout = first_ictal_bout(sm, float(win_ms), min_ms=T["ICTAL_MIN_MS"])
    burn_i = int(np.ceil(float(burn_ms) / float(win_ms)))
    post_burn = sm[burn_i:]
    ref = _reference_check(events, float(burn_ms), float(total_ms), reference_band,
                           min_events=T["MIN_RETURNING_EVENTS"])
    clauses = dict(
        numerical_safe=bool(numerical_safe),
        no_ictal_bout=bout is None,
        low_regime_after_burn=bool(post_burn and all(r == "INTERICTAL" for r in post_burn)),
        not_refractory_plateau=bool(float(refractory_fraction)
                                    <= T["REFRACTORY_CEILING_FRAC_MAX"]),
        returning_reference=bool(ref["passed"]),
    )
    passed = bool(all(clauses.values()))
    return dict(
        schema=SCHEMA,
        verdict=("LC4_CANDIDATE_COMPLETE_LIFECYCLE" if passed
                 else "F2_FROZEN_D_RECOVERY_FAILED"),
        passed=passed,
        clauses=clauses,
        bout=(None if bout is None else [int(bout[0]), int(bout[1])]),
        frozen_window=ref,
        refractory_ceiling_fraction=float(refractory_fraction),
    )
