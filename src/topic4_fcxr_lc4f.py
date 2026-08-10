"""Pure candidate and screen adjudication for LC4f."""
from __future__ import annotations

import numpy as np


def derive_candidate(lc4c_lock: dict, k3: dict, k4: dict, k5: dict,
                     dx: dict, *, y_gate: float) -> dict:
    if lc4c_lock.get("verdict") != "ENTRY_OFFSET_REPAIR_IDENTIFIABLE":
        raise ValueError("LC4f requires the accepted LC4c entry anchor")
    mins = [float(r["x_mean_min"]) for r in (k3, k4, k5)]
    if not (mins[0] <= 0.38 < mins[1] < mins[2]):
        raise ValueError("archived X-depth ordering no longer identifies K_y=3")
    if not bool(dx.get("x_can_terminate_at_observed_D")):
        raise ValueError("archived D/X arbitration lacks termination authority")
    return dict(
        name="lc4f_x_depth_k3", theta_h_lc2=float(lc4c_lock["candidate"]["theta_h_lc2"]),
        use_m=False, y_gate=float(y_gate), K_y=3.0, tau_y=120.0,
        tau_x_down=500.0, tau_x_up=5000.0, x_min=0.1, hill_n=4,
        evidence=dict(k3_x_min=mins[0], k4_x_min=mins[1], k5_x_min=mins[2],
                      termination_boundary=0.38),
    )


def adjudicate_screen(*, regimes, win_ms, events, numerical_safe, refractory_fraction,
                      pre_rate_hz, post_rate_hz, m_current_max):
    from src.topic4_fcxr_lc4_lifecycle import _smooth_isolated, first_ictal_bout

    sm = _smooth_isolated(list(regimes))
    bout = first_ictal_bout(sm, float(win_ms))
    if bout is None:
        return dict(verdict="X_DEPTH_PREVENTS_OR_DELAYS_ENTRY", passed=False,
                    clauses={"qualifying_bout": False}, bout=None)
    b0, b1 = bout
    onset = b0 * float(win_ms)
    ended = b1 + 1 < len(sm)
    offset = (b1 + 1) * float(win_ms) if ended else None
    duration = (b1 - b0 + 1) * float(win_ms)
    pre_events = [e for e in events if e.get("returned") and float(e["t_on"]) < onset]
    guard_n = int(np.ceil(2000.0 / float(win_ms)))
    guard = sm[b1 + 1:b1 + 1 + guard_n] if ended else []
    clauses = dict(
        numerical_safe=bool(numerical_safe), m_current_exactly_zero=float(m_current_max) == 0.0,
        pre_ms=onset >= 8000.0, pre_returning_events=len(pre_events) >= 3,
        bounded_duration=1000.0 <= duration <= 5000.0, autonomous_offset=ended,
        guard_observed=len(guard) == guard_n,
        no_rapid_relapse=len(guard) == guard_n and "ICTAL" not in guard,
        post_rate_suppressed=bool(np.isfinite(post_rate_hz)
                                  and post_rate_hz < pre_rate_hz),
        not_refractory=float(refractory_fraction) <= 0.01,
    )
    if not ended:
        verdict = "X_DEPTH_OFFSET_NEGATIVE"
    elif duration < 1000.0:
        verdict = "X_DEPTH_OVERFAST"
    elif duration > 5000.0:
        verdict = "X_DEPTH_LATE_OFFSET"
    elif not clauses["no_rapid_relapse"]:
        verdict = "X_DEPTH_RAPID_RELAPSE"
    elif all(clauses.values()):
        verdict = "X_DEPTH_OFFSET_CANDIDATE"
    else:
        verdict = "X_DEPTH_SCREEN_INCOMPLETE"
    return dict(verdict=verdict, passed=verdict == "X_DEPTH_OFFSET_CANDIDATE",
                clauses=clauses, bout=[b0, b1], onset_ms=onset, offset_ms=offset,
                bout_ms=duration, n_returning_before_onset=len(pre_events),
                pre_rate_hz=float(pre_rate_hz), post_rate_hz=float(post_rate_hz),
                refractory_ceiling_fraction=float(refractory_fraction))
