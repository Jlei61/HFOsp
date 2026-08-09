"""Pure contracts for FCXR-LC4d offset-latency alignment."""
from __future__ import annotations

import numpy as np

from src.topic4_fcxr_lc4_lifecycle import _smooth_isolated, first_ictal_bout
from src.topic4_fcxr_lc4b_deadzone import deadzone_activation


ENTRY_MIN_MS = 8000.0
ENTRY_MAX_MS = 15000.0
ALIGN_AFTER_ONSET_MS = 4000.0
ZERO_PREFIX_MS = 4000.0
MIN_PRE_EVENTS = 3
ICTAL_MIN_MS = 1000.0
ICTAL_MAX_MS = 5000.0
RELAPSE_GUARD_MS = 2000.0


def derive_latency_candidate(base_candidate: dict, entry_record: dict,
                             nominal_record: dict, a_mean_trace,
                             trace_dt_ms: float, interictal_load,
                             executed_current, current_dt_ms: float) -> dict:
    """Apply the single locked time-alignment transfer without a parameter sweep."""
    entry_gate = entry_record.get("gate") or {}
    nominal_gate = nominal_record.get("nominal_gate") or {}
    if not bool(entry_gate.get("passed")):
        raise ValueError("OFFSET_LATENCY_REPAIR_NOT_IDENTIFIABLE: C1 entry did not pass")
    onset = float(nominal_gate.get("onset_ms", np.nan))
    offset = float(nominal_gate.get("offset_ms", np.nan))
    if not (np.isfinite(onset) and np.isfinite(offset) and onset == 11000.0
            and offset == 66000.0):
        raise ValueError("OFFSET_LATENCY_REPAIR_NOT_IDENTIFIABLE: unexpected onset/offset anchor")
    if not (nominal_record.get("no_kick") and nominal_record.get("no_reset")
            and nominal_record.get("no_parameter_step")):
        raise ValueError("OFFSET_LATENCY_REPAIR_NOT_IDENTIFIABLE: anchor was intervened")
    numerical = nominal_record.get("numerical") or {}
    if (not bool(numerical.get("finite", False)) or bool(numerical.get("numerical_unsafe", True))
            or float(numerical.get("clip_frac_max", np.inf)) != 0.0):
        raise ValueError("OFFSET_LATENCY_REPAIR_NOT_IDENTIFIABLE: anchor was unsafe")

    a = np.asarray(a_mean_trace, dtype=float)
    dt = float(trace_dt_ms)
    if a.ndim != 1 or not a.size or not np.all(np.isfinite(a)) or dt <= 0.0:
        raise ValueError("OFFSET_LATENCY_REPAIR_NOT_IDENTIFIABLE: invalid activation trace")
    align_t = onset + ALIGN_AFTER_ONSET_MS
    align_i = int(round(align_t / dt))
    if align_i >= a.size or not np.isclose(align_i * dt, align_t, atol=1e-12, rtol=0.0):
        raise ValueError("OFFSET_LATENCY_REPAIR_NOT_IDENTIFIABLE: missing exact alignment sample")
    a_align = float(a[align_i])
    if a_align <= 0.0:
        raise ValueError("OFFSET_LATENCY_REPAIR_NOT_IDENTIFIABLE: alignment activation is nonpositive")
    target = float(base_candidate["matched_ictal_current"])
    g_new = target / a_align

    quiet_a = deadzone_activation(
        interictal_load,
        deadzone=float(base_candidate["deadzone"]),
        excess_scale=float(base_candidate["K"]),
        n=float(base_candidate["n"]),
    )
    if not np.all(quiet_a == 0.0):
        raise ValueError("OFFSET_LATENCY_REPAIR_NOT_IDENTIFIABLE: dead zone leaks on reference")
    current = np.asarray(executed_current, dtype=float)
    cdt = float(current_dt_ms)
    n0 = int(round(ZERO_PREFIX_MS / cdt)) if cdt > 0.0 else 0
    if current.ndim != 1 or n0 <= 0 or current.size < n0 or not np.all(current[:n0] == 0.0):
        raise ValueError("OFFSET_LATENCY_REPAIR_NOT_IDENTIFIABLE: source current leaks in prefix")

    out = dict(base_candidate)
    prior_g = float(base_candidate["g_m_max"])
    out.update(
        name="lc4d_offset_latency_alignment",
        g_m_max=float(g_new),
        calibration=dict(
            base=base_candidate.get("calibration", {}),
            source_onset_ms=onset,
            source_offset_ms=offset,
            align_after_onset_ms=ALIGN_AFTER_ONSET_MS,
            align_time_ms=align_t,
            a_mean_at_align=a_align,
            prior_g_m_max=prior_g,
            dose_scale=float(g_new / prior_g),
            interictal_activation_max=float(np.max(quiet_a)),
            prefix_current_max=float(np.max(current[:n0])),
        ),
    )
    return out


def adjudicate_latency_screen(*, regimes, win_ms, events, current_trace,
                              current_dt_ms, numerical_safe: bool,
                              refractory_fraction: float, pre_rate_hz: float,
                              post_rate_hz: float) -> dict:
    """Gate one fresh 18 s trajectory before the expensive 70 s lifecycle run."""
    win_ms = float(win_ms)
    sm = _smooth_isolated(list(regimes))
    bout = first_ictal_bout(sm, win_ms, min_ms=ICTAL_MIN_MS)
    if bout is None:
        return dict(
            schema="fcxr-lc4d-latency-screen-1.0",
            verdict="TERMINATOR_PREVENTS_QUALIFYING_ENTRY",
            passed=False,
            clauses={"qualifying_bout": False},
            bout=None,
        )

    b0, b1 = bout
    onset_ms = float(b0 * win_ms)
    bout_end_ms = float((b1 + 1) * win_ms)
    bout_ms = bout_end_ms - onset_ms
    ended = b1 + 1 < len(sm)
    offset_ms = bout_end_ms if ended else None
    pre_events = [e for e in events
                  if bool(e.get("returned", False)) and float(e["t_on"]) < onset_ms]

    current = np.asarray(current_trace, dtype=float)
    prefix_n = min(current.size, int(round(ZERO_PREFIX_MS / float(current_dt_ms))))
    zero_prefix = bool(prefix_n > 0 and np.all(current[:prefix_n] == 0.0))
    guard_n = max(1, int(np.ceil(RELAPSE_GUARD_MS / win_ms)))
    guard = sm[b1 + 1:min(len(sm), b1 + 1 + guard_n)] if ended else []
    guard_observed = len(guard) == guard_n
    no_relapse = bool(guard_observed and "ICTAL" not in guard)
    suppressed = bool(no_relapse and np.isfinite(pre_rate_hz) and np.isfinite(post_rate_hz)
                      and float(post_rate_hz) < float(pre_rate_hz))

    clauses = dict(
        numerical_safe=bool(numerical_safe),
        entry_in_window=bool(ENTRY_MIN_MS <= onset_ms <= ENTRY_MAX_MS),
        at_least_three_pre_events=bool(len(pre_events) >= MIN_PRE_EVENTS),
        first_eight_seconds_clear=bool(onset_ms >= ENTRY_MIN_MS),
        first_four_seconds_zero_current=zero_prefix,
        bounded_duration=bool(ICTAL_MIN_MS <= bout_ms <= ICTAL_MAX_MS),
        autonomous_offset=bool(ended),
        two_second_guard_observed=guard_observed,
        no_rapid_relapse=no_relapse,
        post_rate_suppressed=suppressed,
        not_refractory_plateau=bool(float(refractory_fraction) <= 0.01),
    )
    passed = bool(all(clauses.values()))
    if not ended or bout_ms > ICTAL_MAX_MS:
        verdict = "OFFSET_LATENCY_REPAIR_INSUFFICIENT"
    elif not no_relapse or not suppressed:
        verdict = "SHORT_POSTICTAL_PROTECTION_INSUFFICIENT"
    elif onset_ms < ENTRY_MIN_MS:
        verdict = "L1_ENTRY_TOO_EARLY"
    elif passed:
        verdict = "L1_ENTRY_OFFSET_ALIGNED"
    else:
        verdict = "L1_LATENCY_GATE_FAILED"
    return dict(
        schema="fcxr-lc4d-latency-screen-1.0",
        verdict=verdict,
        passed=passed,
        clauses=clauses,
        bout=[int(b0), int(b1)],
        onset_ms=onset_ms,
        offset_ms=offset_ms,
        bout_ms=bout_ms,
        n_returning_before_onset=len(pre_events),
        pre_rate_hz=float(pre_rate_hz),
        post_rate_hz=float(post_rate_hz),
        first_four_seconds_current_max=(float(np.max(current[:prefix_n]))
                                        if prefix_n else float("nan")),
        refractory_ceiling_fraction=float(refractory_fraction),
    )
