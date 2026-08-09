"""Pure contracts for FCXR-LC4c entry/offset alignment."""
from __future__ import annotations

import numpy as np

from src.topic4_fcxr_lc4_lifecycle import first_ictal_bout
from src.topic4_fcxr_lc4b_deadzone import deadzone_activation


ENTRY_MIN_MS = 8000.0
ENTRY_MAX_MS = 15000.0
ZERO_PREFIX_MS = 4000.0
MIN_PRE_EVENTS = 3


def derive_aligned_candidate(base_candidate: dict, entry_row: dict,
                             a_mean_trace, interictal_load) -> dict:
    """Apply the two locked analytic repairs without searching either axis."""
    if entry_row.get("onset_ms") is None:
        raise ValueError("ENTRY_OFFSET_REPAIR_NOT_IDENTIFIABLE: entry anchor has no onset")
    onset = float(entry_row["onset_ms"])
    if not ENTRY_MIN_MS <= onset <= ENTRY_MAX_MS:
        raise ValueError("ENTRY_OFFSET_REPAIR_NOT_IDENTIFIABLE: entry anchor outside window")
    if int(entry_row.get("n_returning_before_onset", 0)) < MIN_PRE_EVENTS:
        raise ValueError("ENTRY_OFFSET_REPAIR_NOT_IDENTIFIABLE: too few pre-onset events")
    numerical = entry_row.get("numerical") or {}
    if (not bool(numerical.get("finite", False)) or bool(numerical.get("numerical_unsafe", True))
            or float(numerical.get("clip_frac_max", np.inf)) != 0.0):
        raise ValueError("ENTRY_OFFSET_REPAIR_NOT_IDENTIFIABLE: entry anchor is unsafe")
    if not (entry_row.get("no_kick") and entry_row.get("no_reset")
            and entry_row.get("no_parameter_step")):
        raise ValueError("ENTRY_OFFSET_REPAIR_NOT_IDENTIFIABLE: entry anchor is intervened")

    a = np.asarray(a_mean_trace, dtype=float)
    if a.ndim != 1 or not a.size or not np.all(np.isfinite(a)):
        raise ValueError("ENTRY_OFFSET_REPAIR_NOT_IDENTIFIABLE: invalid activation trace")
    a_max = float(np.max(a))
    if a_max <= 0.0:
        raise ValueError("ENTRY_OFFSET_REPAIR_NOT_IDENTIFIABLE: activation never rises")
    target = float(base_candidate["matched_ictal_current"])
    g_new = target / a_max

    quiet_a = deadzone_activation(
        interictal_load,
        deadzone=float(base_candidate["deadzone"]),
        excess_scale=float(base_candidate["K"]),
        n=float(base_candidate["n"]),
    )
    if not np.all(quiet_a == 0.0):
        raise ValueError("ENTRY_OFFSET_REPAIR_NOT_IDENTIFIABLE: dead zone leaks on reference")

    out = dict(base_candidate)
    out.update(
        name="lc4c_theta110_deadzone_dose_transfer",
        theta_h_lc2=float(entry_row["theta_h_lc2"]),
        g_m_max=float(g_new),
        calibration=dict(
            base=base_candidate.get("calibration", {}),
            entry_anchor_onset_ms=onset,
            entry_anchor_pre_events=int(entry_row["n_returning_before_onset"]),
            closed_high_a_mean_max=a_max,
            old_g_m_max=float(base_candidate["g_m_max"]),
            dose_scale=float(g_new / float(base_candidate["g_m_max"])),
            interictal_activation_max=float(np.max(quiet_a)),
        ),
    )
    return out


def adjudicate_entry(*, regimes, win_ms, events, current_trace, current_dt_ms,
                     numerical_safe: bool, refractory_fraction: float) -> dict:
    """Gate the 15 s dynamic entry probe before any 70 s lifecycle run."""
    bout = first_ictal_bout(regimes, float(win_ms), min_ms=1000.0)
    onset_ms = None if bout is None else float(bout[0] * float(win_ms))
    pre_events = [] if onset_ms is None else [
        e for e in events
        if bool(e.get("returned", False)) and float(e["t_on"]) < onset_ms
    ]
    current = np.asarray(current_trace, dtype=float)
    prefix_n = min(current.size, int(round(ZERO_PREFIX_MS / float(current_dt_ms))))
    zero_prefix = bool(prefix_n > 0 and np.all(current[:prefix_n] == 0.0))
    entry_in_window = bool(onset_ms is not None and ENTRY_MIN_MS <= onset_ms <= ENTRY_MAX_MS)
    first_eight_clear = bool(onset_ms is None or onset_ms >= ENTRY_MIN_MS)
    clauses = dict(
        numerical_safe=bool(numerical_safe),
        entry_in_window=entry_in_window,
        at_least_three_pre_events=bool(len(pre_events) >= MIN_PRE_EVENTS),
        first_eight_seconds_clear=first_eight_clear,
        first_four_seconds_zero_current=zero_prefix,
        not_refractory_plateau=bool(float(refractory_fraction) <= 0.01),
    )
    passed = bool(all(clauses.values()))
    if onset_ms is None:
        verdict = "C1_NO_ENTRY"
    elif onset_ms < ENTRY_MIN_MS:
        verdict = "C1_ENTRY_TOO_EARLY"
    elif not passed:
        verdict = "C1_ENTRY_GATE_FAILED"
    else:
        verdict = "C1_ENTRY_ALIGNED"
    return dict(
        schema="fcxr-lc4c-entry-gate-1.0",
        verdict=verdict,
        passed=passed,
        clauses=clauses,
        onset_ms=onset_ms,
        bout=None if bout is None else [int(bout[0]), int(bout[1])],
        n_returning_before_onset=len(pre_events),
        first_four_seconds_current_max=(float(np.max(current[:prefix_n]))
                                        if prefix_n else float("nan")),
        refractory_ceiling_fraction=float(refractory_fraction),
    )
